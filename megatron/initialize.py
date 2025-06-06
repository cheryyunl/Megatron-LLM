# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""

import random
import time

import numpy as np
import torch
from datetime import timedelta

import megatron
import megatron.fused_kernels
from megatron import get_adlr_autoresume
from megatron import get_tensorboard_writer
from megatron.core import mpu, tensor_parallel

import megatron.arguments

from megatron.checkpointing import load_args_from_checkpoint
from megatron.global_vars import set_global_variables
from megatron.model.transformer import bias_dropout_add_fused_train
from megatron.model.fused_bias_gelu import bias_gelu


def initialize_megatron(extra_args_provider=None,
                        args_defaults={}):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only 
    data processing. In general this arg should not be set unless you know 
    what you are doing.
    """

    # Make sure cuda is available.
    assert torch.cuda.is_available(), 'Megatron requires CUDA.'

    # Parse arguments
    args = megatron.arguments.parse_args(extra_args_provider)

    if args.use_checkpoint_args or args_defaults.get('use_checkpoint_args', False):
        assert args.load is not None, '--use-checkpoints-args requires --load argument'
        load_args_from_checkpoint(args)

    megatron.arguments.validate_args(args, args_defaults)
        
    # set global args, build tokenizer, and set adlr_autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def _finish_mpu_init():
        _initialize_distributed(args)
        
        # Random seeds for reproducibility.
        if args.rank == 0:
            print('> setting random seeds to {} ...'.format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    # Megatron's MPU is the master. Complete initialization right away.
    _finish_mpu_init()
    _init_autoresume()
    # _compile_dependencies(args)

    # No continuation function
    return None


def _compile_dependencies(args):
    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)

    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = args.seq_length
    attn_batch_size = \
        (args.num_attention_heads / args.tensor_model_parallel_size) * \
        args.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = seq_len > 16 and seq_len <= 4096 and \
        seq_len % 4 == 0 and attn_batch_size % 4 == 0

    if not ((args.fp16 or args.bf16) and
            custom_kernel_constraint and
            args.masked_softmax_fusion):
        if args.rank == 0:
            print('WARNING: constraints for invoking optimized'
                  ' fused softmax kernel are not met. We default'
                  ' back to unfused kernel invocations.', flush=True)
    
    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        megatron.fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        megatron.fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)


def _initialize_distributed(args):
    """Initialize torch.distributed and core model parallel."""
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
    # Call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=30)
    )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print('model parallel is already initialized')
        else:
            mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                                          args.pipeline_model_parallel_size,
                                          args.virtual_pipeline_model_parallel_size,
                                          args.pipeline_model_parallel_split_rank)
            if args.rank == 0:
                print(f'> initialized tensor model parallel with size '
                      f'{mpu.get_tensor_model_parallel_world_size()}')
                print(f'> initialized pipeline model parallel with size '
                      f'{mpu.get_pipeline_model_parallel_world_size()}')


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed_))


def write_args_to_tensorboard(args):
    """Write arguments to tensorboard."""
    # NOTE: if we use wandb, then the args are logged on creation, so nothing happens in this 
    # function. 
    if not getattr(args,"wandb_logger",False):
        writer = get_tensorboard_writer()
        if writer:
            for arg in vars(args):
                writer.add_text(arg, str(getattr(args, arg)),
                                global_step=args.iteration)


def set_jit_fusion_options(args):
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function(args)


def _warmup_jit_function(args):
    """ Compile JIT functions before the main training steps """
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(args.ffn_hidden_size // args.tensor_model_parallel_size,
                      dtype=dtype, device='cuda')
    input = torch.rand((args.seq_length, args.micro_batch_size,
                        args.ffn_hidden_size // args.tensor_model_parallel_size),
                       dtype=dtype, device='cuda')
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand((seq_length, args.micro_batch_size, args.hidden_size),
                       dtype=dtype, device='cuda')
    residual = torch.rand((seq_length, args.micro_batch_size, args.hidden_size),
                          dtype=dtype, device='cuda')
    bias = torch.rand((args.hidden_size), dtype=dtype, device='cuda').expand_as(residual)
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()
