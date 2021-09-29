#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import os
import csv
import glob
import json
import gzip
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
import re

import numpy as np
import torch
from torch import Tensor as T
from torch import nn

from dpr.data.qa_validation import calculate_matches
from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint
from dpr.indexer.faiss_indexers import DenseIndexer, DenseHNSWFlatIndexer, DenseFlatIndexer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(self, question_encoder: nn.Module, ctx_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
        self.ctx_encoder = ctx_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       questions[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded queries %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded queries tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def generate_ctx_vectors(self, ctxs: List[str]) -> T:
        n = len(ctxs)
        bsz = self.batch_size
        query_vectors = []

        self.ctx_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [self.tensorizer.text_to_tensor(q) for q in
                                       ctxs[batch_start:batch_start + bsz]]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.ctx_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info('Encoded ctxs %d', len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info('Total encoded ctxs tensor %s', query_tensor.size())

        assert query_tensor.size(0) == len(ctxs)
        return query_tensor

    def get_top_docs(self, query_vectors: np.array, ctxs_vectors: np.array, top_docs: int = 100, ncandidates=2) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        logger.info('index search starting at: %f sec.', time0)
        results = []
        for i in range(len(query_vectors)):
            q_bz = query_vectors[i]
            c_bz = [ ((i*ncandidates)+j, ctxs_vectors[(i*ncandidates)+j]) for j in range(ncandidates) ]
            self.index.reset()
            self.index._index_batch(c_bz)
            results.extend(self.index.search_knn(np.array([q_bz]), top_docs))
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_csv_file(location, dataset=None, CSNET_ADV=False, concode_with_code=False) -> Iterator[Tuple[str, List[str]]]:
    if CSNET_ADV:
        with open(location) as tsvfile:
            for line in tsvfile:
                row_json = json.loads(line)
                yield row_json["docstring"], row_json["function"]

    elif not dataset:
        with open(location) as reader:
            for row in reader:
                row_json = json.loads(row)
                yield ' '.join(row_json["docstring_tokens"]), ' '.join(row_json["code_tokens"])
    elif dataset=='CONCODE':
        logger.info("Parsing CONCODE dataset")
        with open(location) as reader:
            for row in reader:
                row_json = json.loads(row)
                q=row_json["nl"]
                if not concode_with_code:
                    q = q.split("concode_field_sep")[0]
                yield q, row_json["code"]

    elif dataset=='KP20k':
        logger.info("Parsing KP20k dataset")
        with open(location) as reader:
            for row in reader:
                line = json.loads(row)
                q = line["keyword"]
                try:
                    text = line["title"] + ' </s> ' + line["abstract"]
                except:
                    text = "N/A"
                yield q, text



def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, biencoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    print_args(args)

    biencoder, _ = setup_for_distributed_mode(biencoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16,
                                            args.fp16_opt_level)

    biencoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(biencoder)
    logger.info('Loading saved model state ...')
    model_to_load.load_state_dict(saved_state.model_dict)

    encoder = model_to_load.question_model
    ctx_encoder = model_to_load.ctx_model

    vector_size = encoder.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)


    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer, args.faiss_gpu)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer, args.faiss_gpu)

    retriever = DenseRetriever(encoder, ctx_encoder, args.batch_size, tensorizer, index)

    # get questions & answers
    questions = []
    ctxs = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file, dataset=args.dataset, CSNET_ADV=args.CSNET_ADV, concode_with_code=args.concode_with_code):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    ctxs_1 = [x.strip() for x in open(args.ctx_file_1, 'r', encoding='utf-8').readlines()]
    ctxs_2 = [x.strip() for x in open(args.ctx_file_2, 'r', encoding='utf-8').readlines()]
    # ctxs_3 = [x.strip() for x in open(args.ctx_file_3, 'r', encoding='utf-8').readlines()]
    for cx1, cx2 in zip(ctxs_1, ctxs_2):
        ctxs.append(cx1)
        ctxs.append(cx2)
        # ctxs.append(cx3)

    questions_tensor = retriever.generate_question_vectors(questions)
    ctxs_tensor = retriever.generate_ctx_vectors(ctxs)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), ctxs_tensor.numpy(), args.n_docs)




    if args.dataset=="CONCODE":
        logger.info('Writing retrivals reanked prediction to: %s', str(args.out_file))
        import jsonlines
        with open(args.out_file, 'w')  as wf:
            for id, top_ids_scores  in enumerate(top_ids_and_scores):

                if top_ids_scores[0][0] == 0:
                    cand = ctxs_1[id]
                if top_ids_scores[0][0] == 1:
                    cand = ctxs_2[id]
                # else:
                #     cand = ctxs_3[id]


                wf.write(cand + "\n")






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Text and corresponding code file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file_1', required=True, type=str, default=None,
                        help="All codes file in the txt format: function")
    parser.add_argument('--ctx_file_2', required=True, type=str, default=None,
                        help="All codes file in the txt format: function")
    # parser.add_argument('--ctx_file_3', required=True, type=str, default=None,
    #                     help="All codes file in the txt format: function")
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .json file path to write results to ')
    parser.add_argument('--dataset', type=str, default=None,
                        help=' to build correct dataset parser ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string', 'exact'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=100, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=200000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--faiss_gpu", action='store_true', help='If enabled, save use  faiss_gpu')
    parser.add_argument("--CSNET_ADV", action='store_true',
                        help="Whether to parse CSNET_ADV.")
    parser.add_argument("--concode_with_code", action='store_true',
                        help="Whether to parse CSNET_ADV.")


    args = parser.parse_args()


    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
