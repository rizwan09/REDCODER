import os
import re
import json
import logging
import argparse
from tqdm import tqdm

from elasticsearch import Elasticsearch
from retrieval.bm25.doc_db import DocDB


def build_index(db_path, domain, es_index_settings, port=9200, overwrite=False):
    db = DocDB(db_path)

    # initialize the elastic search
    config = {'host': 'localhost', 'port': port}
    es = Elasticsearch([config])
    index_name = "{}_search_test".format(domain)
    if es.indices.exists(index=index_name):
        if overwrite is True:
            es.indices.delete(index=index_name)
        else:
            print(
                "index {} have already been created! Please set overwrite True or use different index name.".format(
                    index_name
                )
            )
            return None

    index_settings = es_index_settings

    es.indices.create(
        index=index_name,
        body={
            "settings": index_settings["settings"]
        }
    )
    # populate index
    # load DB and index in Elastic Search
    es.ping()
    doc_ids = db.get_doc_ids()
    count = 0

    tracer = logging.getLogger('elasticsearch')
    tracer.setLevel(logging.CRITICAL)  # or desired level
    tracer.addHandler(logging.FileHandler('indexer.log'))

    for doc_id in tqdm(doc_ids):
        doc_text = db.get_doc_text(doc_id)
        rec = {"document_text": doc_text, "document_title": doc_id}
        try:
            index_status = es.index(
                index=index_name, id=count, body=rec
            )
            count += 1
        except:
            print(f'Unable to load document {doc_id}.')

    n_records = es.count(index=index_name)['count']
    print(f'Successfully loaded {n_records} into {index_name}')
    return es


def search_es(es_obj, index_name, question_text, n_results=5):
    # construct query
    query = {
        'query': {
            'match': {
                'document_text': question_text
            }
        }
    }

    res = es_obj.search(index=index_name, body=query, size=n_results)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, required=True, help='Path to sqlite db holding document texts')
    parser.add_argument('--domain', type=str, required=True, help='Domain name')
    parser.add_argument('--config_file_path', type=str, required=True, help='path to the congig file')
    parser.add_argument('--port', type=int, required=True, help='port number')
    parser.add_argument('--overwrite', action='store_true', help='overwrite previous index')
    parser.add_argument('--code_to_text', action="store_true")

    args = parser.parse_args()
    question_text = "natural language processing"
    es = build_index(
        args.db_path,
        args.domain,
        json.load(open(args.config_file_path)),
        port=args.port,
        overwrite=args.overwrite
    )
    if es is not None:
        res = search_es(
            es_obj=es,
            index_name="{}_search_test".format(args.domain),
            question_text=question_text,
            n_results=10
        )
        print(res)


if __name__ == '__main__':
    main()
