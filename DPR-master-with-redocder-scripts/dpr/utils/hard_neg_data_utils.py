#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""

import json
import logging
import math
import pickle
import random
from typing import List, Iterator, Callable

from torch import Tensor as T

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info('Reading file %s', path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info('Aggregated data size: {}'.format(len(results)))
    logger.info('Total data size: {}'.format(len(results)))
    return results


def read_data_from_json_files(paths: List[str], upsample_rates: List = None, TopCodeR=True, text_to_code=True, concode_with_code=True, dataset = "CONCODE", valid=False) -> List:
    logger.info("Text to Code:  ", str(text_to_code))

    results = []
    if upsample_rates is None:
        upsample_rates = [1] * len(paths)

    assert len(upsample_rates) == len(paths), 'up-sample rates parameter doesn\'t match input files amount'

    for i, path in enumerate(paths):
        if not TopCodeR:
            with open(path, 'r', encoding="utf-8") as f:
                logger.info('Reading file %s' % path)
                data = json.load(f)
                upsample_factor = int(upsample_rates[i])
                data = data * upsample_factor
                results.extend(data)
                logger.info('Aggregated data size: {}'.format(len(results)))
        else:
            if dataset == 'conala':
                logger.info("Parsing CoNaLa dataset")
                p = (15000+2379)//10
                if path.endswith("json"):
                    with open(path) as reader:
                        data = json.load(reader)
                        if valid:
                            data=data[-p:]
                        else:
                            data = data[:-p]

                        for d in data:
                            try:
                                results.append({"question": d["rewritten_intent"], "hard_negative_ctxs": [], "negative_ctxs": [],
                                  "positive_ctxs": [{"text":d["snippet"], "title":None}], "label": "1"} )
                            except:
                                results.append({"question": d["intent"], "hard_negative_ctxs": [],
                                                "negative_ctxs": [],
                                                "positive_ctxs": [{"text": d["snippet"], "title": None}],
                                                "label": "1"})


                # import ipdb
                # ipdb.set_trace()

            elif dataset == 'CONCODE':
                logger.info("Parsing CONCODE dataset")
                source_str = "nl"
                target_str = "code"
                if not text_to_code:
                    source_str = "code"
                    target_str = "nl"
                with open(path) as reader:
                    for row in reader:
                        line = json.loads(row)
                        q=line[source_str]
                        if not concode_with_code:
                            q = q.split("concode_field_sep")[0]


                        ctx = {"text": line[target_str], "title": None, "answers": [line[target_str]]}
                        object = {"question": q, "hard_negative_ctxs": [], "negative_ctxs": [],
                                      "positive_ctxs": [ctx], "label": "1"}
                        results.append(object)
                        if len(results)<5:
                            logger.info("----------------------")
                            logger.info("Source/q: %s", q)
                            logger.info("Traget/context: %s", ctx["text"])
                            logger.info("----------------------")
            elif dataset == "KP20k":
                logger.info("Parsing KP20k dataset")
                with open(path) as reader:
                    for row in reader:
                        line = json.loads(row)
                        q = line["keyword"]
                        text = line["title"]+ ' </s> ' + line["abstract"]
                        ctx = {"text": text, "title": None, "answers": [ text ]}
                        object = {"question": q, "hard_negative_ctxs": [], "negative_ctxs": [],
                                  "positive_ctxs": [ctx], "label": "1"}
                        results.append(object)
                        if len(results) < 5:
                            logger.info("----------------------")
                            logger.info("Source/q: %s", q)
                            logger.info("Traget/context: %s", ctx["text"])
                            logger.info("----------------------")


            elif dataset == "ICLR" and not text_to_code:
                logger.info("Parsing ICLR dataset")
                with open(path) as reader:
                    for row in reader:
                        line = json.loads(row)
                        q = line["function"]
                        text = line["summary"]
                        ctx = {"text": text, "title": None, "answers": [ text ]}
                        object = {"question": q, "hard_negative_ctxs": [], "negative_ctxs": [],
                                  "positive_ctxs": [ctx], "label": "1"}
                        results.append(object)
                        if len(results) < 5:
                            logger.info("----------------------")
                            logger.info("Source/q: %s", q)
                            logger.info("Traget/context: %s", ctx["text"])
                            logger.info("----------------------")

            elif path.endswith("txt"):
                if text_to_code: logger.info("Text-to-Code")
                else: logger.info("Code-to-Text")
                with open(path, "r", encoding='utf-8') as f:
                    logger.info('Reading file %s' % path)
                    for line in f.readlines():
                        line = line.strip().split('<CODESPLIT>')
                        if len(line) != 5:
                            continue
                        label = line[0]

                        source_str=3
                        target_str=4
                        if not text_to_code:
                            source_str = 4
                            target_str = 3
                        # ctx = {"text":line[4], "title":None, "answers":[line[4]]}
                        # if label=="0": object = {"question": line[3], "hard_negative_ctxs": [ctx], "negative_ctxs":[], "positive_ctxs":[], "label": label}
                        # else: object = {"question": line[3], "positive_ctxs":[ctx], "hard_negative_ctxs": [], "negative_ctxs":[], "label": label}
                        # results.append(object)

                        ctx = {"text": line[target_str], "title": None, "answers": [line[target_str]]}
                        if label == "0":
                            object = {"question": line[source_str], "hard_negative_ctxs": [ctx], "negative_ctxs": [],
                                      "positive_ctxs": [], "label": label}
                        else:
                            object = {"question": line[source_str], "positive_ctxs": [ctx], "hard_negative_ctxs": [],
                                      "negative_ctxs": [], "label": label}
                        results.append(object)

            elif path.endswith("jsonl"):
                with open(path, "r", encoding='utf-8') as f:
                    logger.info('Reading csnet file %s' % path)
                    for line in f.readlines():
                        line = json.loads(line)
                        label = str("1")
                        source_str="docstring_tokens"
                        target_str="function_tokens"
                        target="function"

                        if not text_to_code:
                            source_str = "function_tokens"
                            target_str = "docstring_tokens"

                        try:
                            ctx = {"text":' '.join(line[target_str]), "title":None, "answers":[' '.join(line[target_str])]}
                            if label=="0": object = {"question": ' '.join(line[source_str]), "hard_negative_ctxs": [ctx], "negative_ctxs":[], "positive_ctxs":[], "label": label}
                            else: object = {"question": ' '.join(line[source_str]), "positive_ctxs":[ctx], "hard_negative_ctxs": [], "negative_ctxs":[], "label": label}
                            results.append(object)
                        except:
                            if "function" in source_str:
                                source_str="code_tokens"
                            else:
                                target_str = "code_tokens"
                                target = 'code'
                            try:
                                ctx = {"text": ' '.join(line[target_str]), "title": None, "answers": [' '.join(line[target_str])]}
                                if label == "0":
                                    object = {"question": ' '.join(line[source_str]), "hard_negative_ctxs": [ctx],
                                              "negative_ctxs": [], "positive_ctxs": [], "label": label}
                                else:
                                    object = {"question": ' '.join(line[source_str]), "positive_ctxs": [ctx],
                                              "hard_negative_ctxs": [], "negative_ctxs": [], "label": label}
                                results.append(object)
                            except:
                                logger.info("could  not parse: %s ", line)

            elif path.endswith("json"):
                with open(path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    c=0
                    for id, line in data.items():
                        assert line['query']==line['question']
                        hard_neg_ctx = None
                        for ctx in line['ctxs']:
                            if ctx['document_title']!=id:
                                hard_neg_ctx = ctx
                                hard_neg_ctx['title'] = ctx['document_title']
                                break
                        if hard_neg_ctx:
                            object = {"question": line['query'], "positive_ctxs":[{"text": line['answers'], "title": id}], "hard_negative_ctxs": [hard_neg_ctx], "negative_ctxs":[], "label": "1"}
                            results.append(object)
                        # if c<10: print(object)
                        # c+=1
    logger.info('Aggregated data size: {}'.format(len(results)))
    return results










class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """
    def __init__(self, data: list, shard_id: int = 0, num_shards: int = 1, batch_size: int = 1, shuffle=True,
                 shuffle_seed: int = 0, offset: int = 0,
                 strict_batch_size: bool = False
                 ):

        self.data = data
        total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        samples_per_shard = math.ceil(total_size / self.shards_num)

        self.shard_start_idx = self.shard_id * samples_per_shard

        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.debug(
            'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d', samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    def total_data_len(self) -> int:
        return len(self.data)

    def iterate_data(self, epoch: int = 0) -> Iterator[List]:
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(self.data)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations

        max_iterations = self.max_iterations - self.iteration

        shard_samples = self.data[self.shard_start_idx:self.shard_end_idx]
        for i in range(self.iteration * self.batch_size, len(shard_samples), self.batch_size):
            items = shard_samples[i:i + self.batch_size]
            if self.strict_batch_size and len(items) < self.batch_size:
                logger.debug('Extending batch to max size')
                items.extend(shard_samples[0:self.batch_size - len(items)])
            self.iteration += 1
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            batch = shard_samples[0:self.batch_size]
            yield batch

        logger.debug('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.data:
            visitor_func(sample)


def normalize_question(question: str) -> str:
    if question[-1] == '?':
        question = question[:-1]
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError
