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


def parse_qa_csv_file(location, dataset=None, CSNET_ADV=False, concode_with_code=False, valid=False) -> Iterator[Tuple[str, List[str]]]:
    if CSNET_ADV:
        with open(location) as tsvfile:
            for line in tsvfile:
                row_json = json.loads(line)
                yield row_json["docstring"], row_json["function"]

    # For CoNaLa
    # We consider the first 90/% (2141) data as train the rest 10/% as test.
    # if valid:
    #     data = data[-238:]
    # else:
    #     data = data[:-238]
    # We implement this by commenting line commenting apropriately for the corresponding run
    elif dataset == 'conala_train':
        logger.info("Parsing CoNaLa train dataset")
        p = int(0.1 * (15000 + 2379))
        if location.endswith("json"):
            with open(location) as reader:
                data = json.load(reader)
                data = data[:-p] #train
                for d in data:
                    try:
                        if d['rewritten_intent'] and d['rewritten_intent']!="":
                            yield d["rewritten_intent"], d["snippet"]
                    except:
                        if d['intent'] and d['intent']!="":
                            yield d["intent"], d["snippet"]
    elif dataset == 'conala_valid':
        logger.info("Parsing CoNaLa valid dataset")
        p = int(0.1 * (15000 + 2379))
        if location.endswith("json"):
            with open(location) as reader:
                data = json.load(reader)
                data = data[-p:]  #valid
                for d in data:
                    try:
                        if d['rewritten_intent'] and d['rewritten_intent'] != "":
                            yield d["rewritten_intent"], d["snippet"]
                    except:
                        if d['intent'] and d['intent'] != "":
                            yield d["intent"], d["snippet"]
    elif dataset == 'conala_test':
        logger.info("Parsing CoNaLa test dataset")

        if location.endswith("json"):
            with open(location) as reader:
                data = json.load(reader)

                for d in data:
                    if d['rewritten_intent']:
                        yield d["rewritten_intent"], d["snippet"]
                    else:
                        yield d["intent"], d["snippet"]


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

    elif dataset == "ICLR":
        logger.info("Parsing ICLR dataset")
        with open(location) as reader:
            for row in reader:
                line = json.loads(row)
                q = line["function"]
                text = line["summary"]
                yield text, q

    elif location.endswith('gz'):
        logger.info('Reading file %s' % location)
        with gzip.open(location, 'r') as pf:
            data = pf.readlines()
            for idx, d in enumerate(data):
                line_a = json.loads(str(d, encoding='utf-8'))
                doc_token = ' '.join(line_a['docstring_tokens'])
                code_token = ' '.join(line_a['code_tokens'])
                yield doc_token, code_token
    elif location.endswith("txt"):
        with open(location, "r", encoding='utf-8') as f:
            logger.info('Reading file %s' % location)
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                label = line [0]
                if label=='1':
                    source_str = 3
                    target_str = 4
                    yield line[source_str], line[target_str]
    elif location.endswith("jsonl"):
        with open(location, "r", encoding='utf-8') as f:
            logger.info('Reading csnet file %s' % location)
            for line in f.readlines():
                line = json.loads(line)
                label = str("1")
                source_str = "docstring_tokens"
                target_str = "function_tokens"

                try:
                    yield ' '.join(line[source_str]), ' '.join(line[target_str])
                except:
                    if "function" in source_str:
                        source_str = "code_tokens"
                    else:
                        target_str = "code_tokens"
                    try:
                        yield ' '.join(line[source_str]), ' '.join(line[target_str])
                    except:
                        logger.info("could  not parse: %s ", line)
    elif not dataset:
        with open(location) as reader:
            for row in reader:
                row_json = json.loads(row)
                yield ' '.join(row_json["docstring_tokens"]), ' '.join(row_json["code_tokens"])

def validate(passages: Dict[object, Tuple[str, str]], answers: List[List[str]],
             result_ctx_ids: List[Tuple[List[object], List[float]]],
             workers_num: int, match_type: str) -> List[List[bool]]:
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return match_stats.questions_doc_hits


def load_passages(ctx_files: str, CSNET_ADV=False, dataset = None) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info('Reading data from: %s', ctx_files)
    for ctx_file in glob.glob(ctx_files):
        if CSNET_ADV:
            with open(ctx_file) as tsvfile:
                for line in tsvfile:
                    row =json.loads(line)
                    docs[row["idx"]] = (' '.join(row["function_tokens"]), None)
        elif dataset == "CONCODE":
            with open(ctx_file) as tsvfile:
                idx=0
                for line in tsvfile:
                    js = json.loads(line)
                    docs[ctx_file + "_" + str(idx)] = (js["code"], js["nl"])
                    idx += 1
        elif dataset == "KP20k":
            with open(ctx_file) as tsvfile:
                logger.info( "Reading %s", ctx_file)
                idx=0
                for line in tsvfile:
                    line = json.loads(line)
                    text = line["title"] + ' </s> ' + line["abstract"]
                    docs[str(idx)] = (line["abstract"], line["title"])
                    idx += 1
        elif ctx_file.endswith(".pkl"):
            definitions = pickle.load(open(args.ctx_file, 'rb'))
            for idx, d in enumerate(definitions):
                docs[args.ctx_file + "_" + str(idx)] = (' '.join(d["function_tokens"]), None)

        elif args.ctx_file.endswith('deduplicated.summaries.txt'):
            with open(args.ctx_file) as f:
                for idx, line in enumerate(f):
                    docs[args.ctx_file + "_" + str(idx)] = (line, None)

        elif ctx_file.endswith(".gz"):
            # with gzip.open(ctx_file, 'rt') as tsvfile:
            #     reader = csv.reader(tsvfile, delimiter='\t', )
            #     # file format: doc_id, doc_text, title
            #     for row in reader:
            #         if row[0] != 'id':
            #             docs[row[0]] = (row[1], row[2])

            with gzip.open(args.ctx_file, 'r') as pf:
                data = pf.readlines()
                for idx, d in enumerate(data):
                    line_a = json.loads(str(d, encoding='utf-8'))
                    code_token = ' '.join(line_a['code_tokens'])
                    docs[args.ctx_file + "_" + str(idx)] = (code_token, None)


        elif not ctx_file.endswith(".index.dpr") and not ctx_file.endswith(".index_meta.dpr"):
            with open(ctx_file) as tsvfile:
                idx=0
                for line in tsvfile:
                    for w in ['DEDENT', 'INDENT', "NEW_LINE"]: line=line.replace(w, "")
                    docs[args.ctx_file + "_" + str(idx)] =  (line, None)
                    idx += 1
    return docs


def save_results(passages: Dict[object, Tuple[str, str]], questions: List[str], answers: List[List[str]],
                 top_passages_and_scores: List[Tuple[List[object], List[float]]], per_question_hits: List[List[bool]],
                 out_file: str
                 ):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    ctx_lens = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        merged_data.append({
            'question': q,
            'answers': q_answers,
            'ctxs': [
                {
                    'id': results_and_scores[0][c],
                    'title': docs[c][1],
                    'text': docs[c][0],
                    'score': scores[c],
                    'has_answer': hits[c],
                } for c in range(ctxs_num)
            ]
        })
        for c in range(ctxs_num):
            ctx_lens.append(len(docs[c][0].split()))

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info('Saved results * scores  to %s', out_file)
    try:
        logger.info('Avg retrieved code len %d max len %d min len %d', np.mean(ctx_lens), max(ctx_lens), min(ctx_lens))
    except:
        import ipdb
        ipdb.set_trace()


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
    logger.info('input_paths: %s',str(input_paths))
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

    for ds_item in parse_qa_csv_file(args.qa_file, dataset=args.dataset, CSNET_ADV=args.CSNET_ADV,\
                                     concode_with_code=args.concode_with_code):
        if args.code_to_text:
            answers, question = ds_item
        else:
            question, answers = ds_item
        questions.append(question)
        question_answers.append(answers)

    questions_tensor = retriever.generate_question_vectors(questions)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

    all_passages = load_passages(args.ctx_file, args.CSNET_ADV, dataset=args.dataset)

    if len(all_passages) == 0:
        raise RuntimeError('No passages data found. Please specify ctx_file param properly.')

    if args.CSNET_ADV:
        import jsonlines
        with open(args.qa_file) as tsvfile, jsonlines.open("predictions.jsonl", "w") as wf:
            for line, top_ids_score  in zip(tsvfile, top_ids_and_scores):
                row_json = json.loads(line)
                wf.write({"url":row_json["url"], "answers":top_ids_score[0]})
        import ipdb
        ipdb.set_trace()
    if args.dataset=="KP20k":
        logger.info('Writing retrivals to: predictions_KP20k.jsonl')
        import jsonlines
        with open(args.qa_file) as tsvfile, jsonlines.open("predictions_KP20k.jsonl", "w") as wf:
            idx=0
            for line, top_ids_score  in zip(tsvfile, top_ids_and_scores):
                wf.write({"id":str(idx), "answers":top_ids_score[0]})
                idx+=1
        import ipdb
        ipdb.set_trace()


    questions_doc_hits = validate(all_passages, question_answers, top_ids_and_scores, args.validation_workers,
                                  args.match)

    if args.out_file:
        save_results(all_passages, questions, question_answers, top_ids_and_scores, questions_doc_hits, args.out_file)


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
    parser.add_argument("--code_to_text", action='store_true')


    args = parser.parse_args()

    if args.dataset=='CONCODE':
        args.encoded_ctx_file+='*.pkl'
        args.ctx_file+='*.json'

    assert args.model_file, 'Please specify --model_file checkpoint to init model weights'

    setup_args_gpu(args)
    print_args(args)
    main(args)
