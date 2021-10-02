import json
import argparse
import jsonlines

from tqdm import tqdm
from pathlib import Path
from elasticsearch import Elasticsearch


def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def search_es(es_obj, index_name, query_text, n_results=5):
    # construct query
    query = {
        'query': {
            'match': {
                'document_text': query_text
            }
        }
    }

    res = es_obj.search(index=index_name, body=query, size=n_results)

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', type=str, required=True, help='Path to index')
    parser.add_argument('--input_data_file', type=str, required=True, help='Path to index')
    parser.add_argument('--port', type=int, required=True, help='Port number')
    parser.add_argument('--output_fp', type=str, required=True)
    parser.add_argument('--only_query', action='store_true')
    parser.add_argument('--n_docs', type=int, default=100)
    parser.add_argument('--code_to_text', action="store_true")

    args = parser.parse_args()
    input_data = read_jsonlines(args.input_data_file)
    config = {'host': 'localhost', 'port': args.port}
    es = Elasticsearch([config])
    result = {}

    idx = 0
    fname = Path(args.input_data_file).stem
    for item in tqdm(input_data):
        if 'concode' in args.input_data_file:
            answers = item["code"]
            if args.only_query:
                query = item["nl"].split('concode_elem_sep')[0]
            else:
                query = item["nl"]
        else:
            query = ' '.join(item['docstring_tokens'])
            answers = ' '.join(item["code_tokens"])

        if args.code_to_text:
            temp = query
            query = answers
            answers = temp
            if idx<10: print("query: ", query, "answers: ", answers)

        if len(query) == 0:
            continue

        res = search_es(
            es_obj=es,
            index_name=args.index_name,
            query_text=query,
            n_results=args.n_docs
        )
        result[fname + '.' + str(idx)] = {
            "ctxs": res["hits"]["hits"],
            "query": query,
            "question": query,
            "found": False,
            "answers": answers
        }
        idx += 1

    # evaluate top n accuracy
    for q_id in result:
        hits = result[q_id]["ctxs"]
        for hit in hits:
            if q_id == hit["_source"]["document_title"]:
                result[q_id]["found"] = True
                break
        # filtering fields to store less data
        result[q_id]["ctxs"] = [
            {
                'document_title': h["_source"]["document_title"],
                '_score': h["_score"],
                'text': h["_source"]["document_text"]
            }
            for h in hits
        ]

    with open(args.output_fp, 'w') as outfile:
        json.dump(result, outfile, indent=True)

    top_n_accuracy = len([q_id for q_id, item in result.items() if item["found"]]) / len(result)
    print(top_n_accuracy)


if __name__ == '__main__':
    main()
