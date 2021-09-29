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
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer, index: DenseIndexer):
        self.question_encoder = question_encoder
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

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        logger.info('index search starting at: %f sec.', time0)
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info('index search time: %f sec.', time.time() - time0)
        return results


def parse_qa_csv_file(location, dataset=None, CSNET_ADV=False, concode_with_code=False) -> Iterator[Tuple[str, List[str]]]:
    with open(args.ctx_file, 'r') as pf:
        data = pf.readlines()
        for idx, d in enumerate(data):
            line_a = json.loads(d)
            doc_token = ' '.join(line_a['docstring_tokens'])
            # code_token = ' '.join(line_a['code_tokens'])
            code_token = line_a['url']
            yield doc_token, code_token



def load_passages(ctx_files: str, CSNET_ADV=False, dataset = None) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_files)
    with gzip.open(args.ctx_file, 'r') as pf:
        data = pf.readlines()
        for idx, d in enumerate(data):
            line_a = json.loads(str(d, encoding='utf-8'))
            code_token = ' '.join(line_a['code_tokens'])
            docs[line_a['url']] = (code_token, None)
    return docs



def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info('Reading file %s', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def main(args):
    saved_state = load_states_from_checkpoint(args.model_file)
    set_encoder_params_from_state(saved_state.encoder_params, args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu,
                                            args.local_rank,
                                            args.fp16)
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info('Loading saved model state ...')

    prefix_len = len('question_model.')
    question_encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                              key.startswith('question_model.')}
    model_to_load.load_state_dict(question_encoder_state)
    vector_size = model_to_load.get_out_size()
    logger.info('Encoder vector_size=%d', vector_size)

    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer, args.faiss_gpu)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer, args.faiss_gpu)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)


    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = [path for path in glob.glob(ctx_files_pattern) if not path.endswith(".index.dpr") and not path.endswith(".index_meta.dpr")]
    logger.info("input_paths=%s ", input_paths)
    index_path = "_".join(input_paths[0].split("_")[:-1])

    if args.save_or_load_index and (os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")):
        retriever.index.deserialize_from(index_path)
    else:
        logger.info('Reading all passages data from files: %s', input_paths)
        retriever.index.index_data(input_paths)
        if args.save_or_load_index:
            retriever.index.serialize(index_path)
    # get questions & answers
    questions = []
    question_answers = []

    for ds_item in parse_qa_csv_file(args.qa_file, dataset=args.dataset, CSNET_ADV=args.CSNET_ADV, concode_with_code=args.concode_with_code):
        question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)
    ranks=[]

    idxs = np.arange(len(questions))
    np.random.seed(2)  # set random seed so that random things are reproducible

    ranks = []
    for idxxx, top_ids_scores in enumerate(top_ids_and_scores):
        # real_rank = top_ids_scores[0].index(question_answers[idx])
        # ranks.append(1/real_rank)
        rank = 0
        find = False
        for url in top_ids_scores[0][:1000]:
            if find is False:
                rank += 1
            if question_answers[idxxx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)
    logger.info("Ranks: %s", ranks)
    mean_mrr = np.mean(np.array(ranks))
    print("mean mrr: {}".format(mean_mrr))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--qa_file', required=True, type=str, default=None,
                        help="Text and corresponding code file of the format: question \\t ['answer1','answer2', ...]")
    parser.add_argument('--ctx_file', required=True, type=str, default=None,
                        help="All codes file in the txt format: function")
    parser.add_argument('--encoded_ctx_file', type=str, default=None,
                        help='Glob path to encoded passages (from generate_dense_embeddings tool)')
    parser.add_argument('--out_file', type=str, default=None,
                        help='output .json file path to write results to ')
    parser.add_argument('--dataset', type=str, default=None,
                        help=' to build correct dataset parser ')
    parser.add_argument('--match', type=str, default='string', choices=['regex', 'string', 'exact'],
                        help="Answer matching logic type")
    parser.add_argument('--n-docs', type=int, default=100, help="Amount of top docs to return")
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for question encoder forward pass")
    parser.add_argument('--index_buffer', type=int, default=200000,
                        help="Temporal memory data buffer size (in samples) for indexer")
    parser.add_argument("--hnsw_index", action='store_true', help='If enabled, use inference time efficient HNSW index')
    parser.add_argument("--save_or_load_index", action='store_true', help='If enabled, save index')
    parser.add_argument("--faiss_gpu", action='store_true', help='If enabled, save use  faiss_gpu')
    parser.add_argument("--CSNET_ADV", action='store_true',
                        help="Whether to parse CSNET_ADV.")
    parser.add_argument("--concode_with_code", action='store_true',
                        help="Whether to use concode_with_code.")


    args = parser.parse_args()

    if args.dataset=='CONCODE':
        args.encoded_ctx_file+='*.pkl'
        args.ctx_file+='*.json'

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
