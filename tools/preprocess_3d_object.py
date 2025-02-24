"""Processing data for multi-modal {instruction tuning, pre-training}."""

import sys
import json
import time
import itertools
import numpy as np
from pathlib import Path
from typing import Optional, Iterable
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from collections import defaultdict
import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from megatron.tokenizer import build_tokenizer
from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.data.indexed_dataset import make_builder, MMapIndexedDatasetBuilder
from megatron.data.instruction_dataset import Role

# ========
# Handle Multimodal Images
import io
import torch
import base64
import torchvision.transforms as transforms
import math
from PIL import Image

PATCH_SIZE = 512

def process_pointcloud(points_data):
   x_sorted_indices = np.argsort(points_data[:, 0])
   points_sorted = points_data[x_sorted_indices]
   n_patches = math.ceil(len(points_sorted) / PATCH_SIZE)
   patches = []
   for i in range(n_patches):
       start_idx = i * PATCH_SIZE
       end_idx = start_idx + PATCH_SIZE
       
       if end_idx <= len(points_sorted):
           patch = points_sorted[start_idx:end_idx]
       else:
           patch = points_sorted[start_idx:]
           remaining = PATCH_SIZE - len(patch)
           patch = np.concatenate([patch, patch[:remaining]])
           
       patches.append(patch)
   return patches


def load_image_to_base64(image_path: str) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def load_base64_to_PILImage(base64_string: str) -> Image:
    # convert data:image/jpeg;base64, to jpeg
    base64_string = base64_string.split(",")[1]
    decoded_string = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded_string)).convert('RGB')

# ========

# def format_message(message: str, role: str) -> str:
#     return f"<|im_start|>{role}\n{message}<|im_end|>\n"
MESSAGE_PREFIX = "{role}\n"
MESSAGE_SUFFIX = "\n"
NON_VISION_TOKEN = -1

class Encoder(object):
    tokenizer: Optional[AbstractTokenizer] = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)
        
        Encoder.pointcloud_start_token = Encoder.tokenizer.vocab[self.args.pointcloud_start_token]
        Encoder.pcd_patch_token = Encoder.tokenizer.vocab[self.args.pcd_patch_token]
        Encoder.z_sep_token = Encoder.tokenizer.vocab[self.args.z_sep_token]
        Encoder.row_sep_token = Encoder.tokenizer.vocab[self.args.row_sep_token]
        Encoder.pointcloud_end_token = Encoder.tokenizer.vocab[self.args.pointcloud_end_token]
        
        if not self.args.do_pretrain:
            Encoder.chat_start_token = Encoder.tokenizer.vocab[self.args.chat_start_token]
            Encoder.chat_end_token = Encoder.tokenizer.vocab[self.args.chat_end_token]

    def encode(self, line: str) -> tuple[int, list[int], list[int], np.ndarray]:
        try:
            return self._encode(line)
        except Exception as e:
            print(f"Error encountered, skipping: {e}")
            return None
    
    def _encode(self, line: str) -> tuple[int, list[int], list[int], np.ndarray]:
        # get data
        assert Encoder.tokenizer is not None

        data = json.loads(line)

        # each line can be:
        if isinstance(data, list):
            # 1 conversations: [{"role": "human", "content": [{"type": "text", "text": "XXX"}]}]
            conversations = data
        elif isinstance(data, dict) and "conversations" in data:
            # 2 a dict with "id": {"id": "XXX", "conversations": [{"role": "human", "content": [{"type": "text", "text": "XXX"}]}]}
            assert isinstance(data, dict), "Data must be a list or a dictionary."
            # _id = data["id"]
            conversations = data["conversations"]
        elif isinstance(data, dict) and "content" in data and "role" in data:
            # 3 a dict with {"role": "human", "content": [{"type": "text", "text": "XXX"}]}
            conversations = [data]
        else:
            raise ValueError(f"Unknown data format: {data}")

        # tokenize and get roles
        tokens = []
        roles = []
        vision_patch_indices = [] # same shape as tokens, NON_VISION_TOKEN=-1 for non-vision tokens
        vision_patches = [] # list of C*PATCH_H*PATCH_W = 3*32*32 = 3072
        n_images = 0

        for turn in conversations:
            role = turn["role"]
            if not isinstance(turn["content"], list):
                assert isinstance(turn["content"], str), "Content must be a string (text) if not a list."
                turn["content"] = [{
                    "type": "text",
                    "text": turn["content"]
                }]

            # add format prefix/suffix if not pre-training
            if not self.args.do_pretrain:
                prefix_tokens = [Encoder.chat_start_token] + Encoder.tokenizer.tokenize(MESSAGE_PREFIX.format(role=role))
                tokens += prefix_tokens
                roles += [Role[role].value]*len(prefix_tokens)
                vision_patch_indices += [NON_VISION_TOKEN]*len(prefix_tokens)

            for item in turn["content"]:
                if item["type"] == "text":
                    tokenized_text = Encoder.tokenizer.tokenize(item["text"])
                    tokens += tokenized_text
                    roles += [Role[role].value]*len(tokenized_text)
                    vision_patch_indices += [NON_VISION_TOKEN]*len(tokenized_text)
                
                elif item["type"] == "image_url":
                    pcd_content = item["image_url"]["url"]
                    #assert "base64" in pcd_content, "Only base64 point cloud is currently supported."
                    points_data = np.frombuffer(base64.b64decode(pcd_content),dtype=np.float64)
                    assert points_data.shape == (6144,), f"Invalid shape {points_data.shape}"
                    points_data = points_data.reshape(-1, 6)
                    assert points_data.shape == (1024, 6), f"Invalid shape {points_data.shape}"
                    cur_vision_patches = process_pointcloud(points_data)
                    assert cur_vision_patches[0].shape == (512, 6), f"Invalid shape {cur_vision_patches[0].shape}"
                    #f.write(f"Number of patches created: {len(cur_vision_patches)}\n")
                    #f.write(f"Single patch shape: {cur_vision_patches[0].shape}\n")
                    n_patches = len(cur_vision_patches)
                    cur_vision_patches = [patch.reshape(-1) for patch in cur_vision_patches]
                    cur_vision_patches = np.stack(cur_vision_patches)
                    n_images += 1

                    dummy_vision_tokens = []
                    dummy_roles = []
                    cur_patch_indices = []

                    for i in range(n_patches):
                        dummy_vision_tokens.append(Encoder.pcd_patch_token)
                        dummy_roles.append(Role.image.value)  # Role.pointcloud.value
                        cur_patch_indices.append(len(vision_patches) + i)

                    # Update data
                    tokens += [Encoder.pointcloud_start_token] + dummy_vision_tokens + [Encoder.pointcloud_end_token]
                    roles += [Role[role].value] + dummy_roles + [Role[role].value]
                    
                    vision_patch_indices += [NON_VISION_TOKEN] + cur_patch_indices + [NON_VISION_TOKEN]
                    vision_patches.extend([p.astype(np.float16) for p in cur_vision_patches])

                #else:
                    #raise ValueError(f"Unknown content type (only 'text' and 'image_url' are supported): {item['type']}")

            if not self.args.do_pretrain:
                suffix_tokens = [Encoder.chat_end_token] + Encoder.tokenizer.tokenize(MESSAGE_SUFFIX)
                tokens += suffix_tokens
                roles += [Role[role].value]*len(suffix_tokens)
                vision_patch_indices += [NON_VISION_TOKEN]*len(suffix_tokens)

        assert len(vision_patches) == len(list(filter(lambda r: r == Role.image.value, roles))), \
            f"Number of image patches should be equal to the number of image tokens."
        # vision_patches = np.array(vision_patches)
        assert len(tokens) == len(vision_patch_indices)
        assert len(tokens) == len(roles)
        return len(line), tokens, roles, vision_patches, vision_patch_indices, n_images

    @property
    def special_tokens(self) -> dict:
        return self.tokenizer._special_tokens


class DatasetWriter:
    def __init__(self, prefix: str, vocab_size: int, dataset_impl: str = "mmap",
                 feature: str = "text"):
        self.vocab_size = vocab_size
        self.dataset_impl = dataset_impl
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, tokens: list[int]):
        self.builder.add_item(torch.IntTensor(tokens))

    def __enter__(self):
        self.builder = make_builder(self.bin_fname, impl=self.dataset_impl,
                                    vocab_size=self.vocab_size)
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None

class ImageDatasetWriter:
    def __init__(self, prefix: str, feature: str = "image_patch"):
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, list_of_image_patches: list[np.ndarray]):
        self.builder.add_item(torch.HalfTensor(list_of_image_patches))

    def __enter__(self):
        self.builder = MMapIndexedDatasetBuilder(
            self.bin_fname,
            dtype=np.float16,
        )
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None

def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs="+",
                       help='Path(s) to input JSON file(s)')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer', 'FalconTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_prefix', type=Path, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk_size', type=int, default=32,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--no_mp', action='store_true',
                       help='Disable multiprocessing')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('--chat_start_token', type=str, default='<|im_start|>')
    group.add_argument('--chat_end_token', type=str, default='<|im_end|>')
    group.add_argument('--pointcloud_start_token', type=str, default='<pointcloud>')
    group.add_argument('--pcd_patch_token', type=str, default='<pcd_patch>')
    group.add_argument('--z_sep_token', type=str, default='<z_sep>')
    group.add_argument('--row_sep_token', type=str, default='<row_sep>')
    group.add_argument('--pointcloud_end_token', type=str, default='</pointcloud>')
    group.add_argument('--vocab_extra_ids', type=int, default=0)
    group.add_argument('--vocab_extra_ids_list', type=str, default=None,
                       help='comma separated list of special vocab ids to add to the tokenizer')
    group.add_argument("--no_new_tokens", action="store_false", dest="new_tokens",
                       help=("Whether to add special tokens (e.g. CLS, MASK, etc) "
                             "in the sentencepiece tokenizer or not"))
    group.add_argument("--do_packing", action="store_true",
                       help=("Whether to pack documents into sequences of max_seq_length."))
    group.add_argument("--do_pretrain", action="store_true",
                       help=("Whether to format data for pretraining by removing all the chat format."))
    group.add_argument("--max_seq_length", type=int, default=4096)
    group.add_argument("--target_size", type=int, default=None)
    args = parser.parse_args()
    args.keep_empty = False


    if args.do_packing:
        assert args.max_seq_length, "Must specify max_seq_length when packing documents."
        print(f"Packing documents into sequences of max_seq_length {args.max_seq_length}.")

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    return args

def pack_docs(docs: Iterable, tokenizer, max_seq_length, stats_counter, keep_data_after_truncation=True):
    current_pack_tokens = []
    current_pack_roles = []
    current_pack_vision_patches = []
    current_pack_vision_patch_indices = []

    current_pack_n_images = 0
    current_seq_length = 0
    current_size = 0
    pbar = tqdm(desc="Packing documents")
    for content in docs:
        if content is None:
            stats_counter["error_skipped"] += 1
            continue

        size, tokens, roles, vision_patches, vision_patch_indices, n_images = content

        stats_counter["n_images"] += n_images

        # Check if adding the current text (with separator) will exceed max_seq_length
        if current_seq_length + len(tokens) + 1 <= max_seq_length:
            if current_pack_tokens:  # not the first sentence in pack
                # Add separator token
                current_pack_tokens.append(tokenizer.bos)
                current_pack_roles.append(Role.PACK_SEP.value)
                current_pack_vision_patch_indices.append(NON_VISION_TOKEN)
                
                # increment `vision_patch_indices` by the number of existing visual patches in the pack
                vision_patch_indices = [
                    idx if idx == NON_VISION_TOKEN else idx + len(current_pack_vision_patches)
                    for idx in vision_patch_indices
                ]

                current_seq_length += 1
                current_size += len(tokenizer._inv_special_tokens[tokenizer.bos])

            current_pack_tokens.extend(tokens)
            current_pack_roles.extend(roles)
            current_pack_vision_patches.extend(vision_patches)
            current_pack_vision_patch_indices.extend(vision_patch_indices)

            current_pack_n_images += n_images
            current_seq_length += len(tokens)
            current_size += size

        elif current_seq_length == 0:
            # The only possible reason for this is len(tokens) >= max_seq_length 
            assert len(current_pack_tokens) == len(current_pack_roles) == 0
            assert len(tokens) >= max_seq_length
            assert len(tokens) == len(roles)
            assert len(tokens) == len(vision_patch_indices)
            # We truncate the tokens to max_seq_length AND treat it as a single pack
            # packed_docs.append((size, tokens[:max_seq_length], roles[:max_seq_length]))
            stats_counter["n_total_packed_doc"] += 1
            stats_counter["n_total_tokens"] += len(current_pack_tokens)
            yield (
                size,
                tokens[:max_seq_length],
                roles[:max_seq_length],
                vision_patches, # no need to truncate vision patches since it is indexed by vision_patch_indices
                vision_patch_indices[:max_seq_length],
                n_images * (max_seq_length / len(tokens)) # use fraction of images to approximate the number of images
            )

            if keep_data_after_truncation:
                # Put the remaining tokens into the next pack
                current_pack_tokens = tokens[max_seq_length:]
                current_pack_roles = roles[max_seq_length:]
                current_pack_vision_patches = vision_patches # no need to truncate vision patches
                current_pack_vision_patch_indices = vision_patch_indices[max_seq_length:]

                current_pack_n_images = n_images * (len(current_pack_tokens) / len(tokens)) # use fraction of images to approximate the number of images
                current_seq_length = len(current_pack_tokens)
                current_size = size * (len(current_pack_tokens) / len(tokens))

        else:
            # Finish the current pack and start a new one
            assert len(current_pack_tokens) > 0
            assert current_size > 0
            assert len(current_pack_tokens) == len(current_pack_roles)
            assert len(current_pack_tokens) == len(current_pack_vision_patch_indices)
            # packed_docs.append((current_size, current_pack_tokens, current_pack_roles))
            stats_counter["n_total_packed_doc"] += 1
            stats_counter["n_total_tokens"] += len(current_pack_tokens)
            yield (
                current_size,
                current_pack_tokens,
                current_pack_roles,
                current_pack_vision_patches,
                current_pack_vision_patch_indices,
                current_pack_n_images
            )

            current_pack_tokens = tokens
            current_pack_roles = roles
            current_pack_vision_patches = vision_patches
            current_pack_vision_patch_indices = vision_patch_indices
            
            current_pack_n_images = n_images
            current_seq_length = len(tokens)
            current_size = size
        
        stats_counter["n_total_doc"] += 1
        pbar.update(1)
        # add status update
        pbar.set_postfix(stats_counter)
    pbar.close()

    # Add any remaining packed sequences
    if current_pack_tokens:
        assert len(current_pack_tokens) > 0, "Should not have empty pack."
        assert len(current_pack_tokens) == len(current_pack_roles)
        assert len(current_pack_tokens) == len(current_pack_vision_patch_indices)
        # packed_docs.append((current_size, current_pack_tokens, current_pack_roles))
        stats_counter["n_total_packed_doc"] += 1
        stats_counter["n_total_tokens"] += len(current_pack_tokens)
        yield (
            current_size,
            current_pack_tokens,
            current_pack_roles,
            current_pack_vision_patches,
            current_pack_vision_patch_indices,
            current_pack_n_images
        )
    
    print(f"Packed {stats_counter['n_total_doc']} documents into {stats_counter['n_total_packed_doc']} documents ({stats_counter['n_total_tokens']} tokens)")


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    
    # These tokens should already in the tokenizer
    assert args.pointcloud_start_token in tokenizer.vocab
    assert args.pcd_patch_token in tokenizer.vocab
    assert args.z_sep_token in tokenizer.vocab
    assert args.row_sep_token in tokenizer.vocab
    assert args.pointcloud_end_token in tokenizer.vocab

    def smart_open(f):
        if ".gz" in f:
            import gzip
            print(f"Detected .gz file ({f}), will use gzip.open to read them.")
            return gzip.open(f)
        elif ".lz4" in f:
            import lz4.frame
            print(f"Detected .lz4 file ({f}), will use lz4.frame.open to read them.")
            return lz4.frame.open(f)
        else:
            return open(f)

    fs = map(smart_open, args.input)

    output_jsonl = f"{args.output_prefix}.jsonl"
    with DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "text") as token_writer, \
            DatasetWriter(args.output_prefix, 16, args.dataset_impl,
                          "role") as role_writer, \
            ImageDatasetWriter(args.output_prefix, "vision_patch") as vision_patches_writer, \
            DatasetWriter(args.output_prefix, None, args.dataset_impl,
                          "vision_patch_indices") as vision_patch_indices_writer, \
            open(output_jsonl, "w") as output_file:

        f = itertools.chain(*fs)
        if not args.no_mp:
            print(f"Using multiprocessing with {args.workers} workers.")
            pool = Pool(args.workers, initializer=encoder.initializer)
            docs = pool.imap(encoder.encode, f, args.chunk_size)
        else:
            encoder.initializer()
            docs = (encoder.encode(i) for i in f)

        stats_counter = defaultdict(int)
        if args.do_packing:
            # make sure it works when docs is a generator
            # print(f"Sorting loaded documents by length for efficient packing. This can be slow for large dataset.")
            # docs = sorted(docs, key=lambda x: len(x[1]), reverse=True)
            docs = pack_docs(docs, tokenizer, args.max_seq_length, stats_counter)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, content in enumerate(docs, start=1):
            if args.target_size and i >= args.target_size:
                print(f"Target size reached. Stopping.")
                break

            if content is None:
                continue

            (size, tokens, roles, vision_patches, vision_patch_indices, n_images) = content
            total_bytes_processed += size
            if len(tokens) == 0:
                print("WARNING: Encountered empty document, skipping.")
                exit(1)
            assert size > 0
            assert len(tokens) == len(roles)
            assert len(tokens) > 0

            if len(vision_patches) == 0:
                # Add a dummy patch IF no vision patches are found
                # This is to ensure that the MMAP in the training code works
                vision_patches = [
                    np.zeros((6 * PATCH_SIZE), dtype=np.float16)
                ]

            token_writer.add_item(tokens)
            role_writer.add_item(roles)
            vision_patches_writer.add_item(np.array(vision_patches))
            vision_patch_indices_writer.add_item(vision_patch_indices)
            stats = {
                "n_tokens": len(tokens),
                "n_pointcloud_tokens": sum(list(map(lambda r: r == Role.image.value, roles))),
                "n_objects": n_images,
            }
            output_file.write(json.dumps(stats) + "\n")

            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print(f"Done processing {i} documents! Now finalizing.")

    # Save stats
    with open(f"{args.output_prefix}_stats.json", "w") as stats_file:
        stats = dict(
            total_bytes_processed=total_bytes_processed,
            total_docs_processed=i,
            total_time=time.time() - startup_start,
            total_time_startup=startup_end - startup_start,
            total_time_processing=time.time() - proc_start,
            **stats_counter
        )
        stats_file.write(json.dumps(stats))
    Path(f"{args.output_prefix}_DONE").touch()

    for f in fs:
        f.close()

    if not args.no_mp:
        pool.close()
        pool.join()
    
if __name__ == '__main__':
    main()
