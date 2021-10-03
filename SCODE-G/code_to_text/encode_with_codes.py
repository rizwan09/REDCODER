#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from tqdm import tqdm

from multiprocessing import Pool
import sentencepiece as spm
from python_tokenizer import python_code_tokenize


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.keep_empty = args.keep_empty
        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len
        self.sp = spm.SentencePieceProcessor(model_file=args.model_file)

    def initializer(self):
        pass

    def _encode(self, line):
        return self.sp.encode(line, out_type=str)

    def _decode(self, tokens):
        return self.sp.decode(tokens)

    def encode(self, example):
        assert isinstance(example, dict)
        assert 'src' in example and 'tgt' in example
        if len(example['src']) == 0 and not self.keep_empty:
            return None
        if len(example['tgt']) == 0 and not self.keep_empty:
            return None
        src_tokens = self._encode(example['src'])[:self.max_src_len]
        tgt_tokens = self._encode(example['tgt'])[:self.max_tgt_len]
        return {'src': " ".join(src_tokens), 'tgt': " ".join(tgt_tokens)}


def load_data(input_file, src_field, tgt_field, src_lang, top_k, WITH_WITHOUT_SUFFIX):
    data = []
    with open(input_file) as f:
        retrieved_code = json.load(f)

    for idx, ex in enumerate(tqdm(retrieved_code, total=len(retrieved_code))):
            try:
                src = ex[src_field]
            except:
                ex=retrieved_code[ex]
            assert src_field in ex and tgt_field in ex
            src = ex[src_field]
            src = src.replace('\n', ' ').strip()
            tgt = ex[tgt_field]
            if isinstance(tgt, list):
                tgt = " ".join(tgt)
            tgt = tgt.replace('\n', ' ').strip()

            if top_k > 0:
                inserted = 0
                for rank, ctx in enumerate(ex['ctxs']):
                    if WITH_WITHOUT_SUFFIX=="with":
                        src += ' _NL_SEP_ ' + (ctx["text"])
                        inserted += 1
                        if inserted >= top_k:
                            break
                    else:
                        if tgt.strip()!=ctx["text"].strip(): #for retrieving without ref code but includes other codes in the test corpus
                            src += ' _NL_SEP_ ' + (ctx["text"])
                            inserted += 1
                            if inserted >= top_k:
                                break


            src = src.replace('\n', ' ').strip()
            data.append({'src': src, 'tgt': tgt})

    return data


def process(args):
    dataset = load_data(args.input_file, args.src_field, args.tgt_field, args.src_lang, top_k=args.top_k, WITH_WITHOUT_SUFFIX=args.WITH_WITHOUT_SUFFIX)

    encoder = MultiprocessingEncoder(args)
    pool = Pool(args.workers, initializer=encoder.initializer)

    processed_dataset = []
    with tqdm(total=len(dataset), desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(encoder.encode, dataset, 100)):
            pbar.update()
            processed_dataset.append(ex)

    out_src = os.path.join(args.output_dir, '{}.spm.{}'.format(args.pref, args.src_lang))
    out_tgt = os.path.join(args.output_dir, '{}.spm.{}'.format(args.pref, args.tgt_lang))
    with open(out_src, 'w', encoding='utf-8') as src_writer, \
            open(out_tgt, 'w', encoding='utf-8') as tgt_writer:
        for ex in processed_dataset:
            if ex is not None:
                src_writer.write(ex['src'] + '\n')
                tgt_writer.write(ex['tgt'] + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        help='path to *.model file',
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=['-'],
        help="input files (.jsonl) to filter/encode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=['_'],
        help="path of the output directory",
    )
    parser.add_argument(
        "--src_field",
        type=str,
        default=['_'],
        help="field name to be considered as the source",
    )
    parser.add_argument(
        "--tgt_field",
        type=str,
        default=['_'],
        help="field name to be considered as the source",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default=['_'],
        help="name of the source language",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default=['_'],
        help="name of the target language",
    )
    parser.add_argument(
        "--pref",
        type=str,
        default=['_'],
        help="file prefix",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=512)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--WITH_WITHOUT_SUFFIX", type=str, default="with")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()
