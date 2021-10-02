import json
import argparse


def MAP(inputfile):
    with open(inputfile) as f:
        examples = json.load(f)
    map = 0
    max_k = 0

    # scores = [[0 for i in range(max_k)] for j in range(len(retrieved_code))]
    # for idx, ex in enumerate(retrieved_code):
    #     target = ex['answers'].strip()
    #     for rank, ctx in enumerate(ex['ctxs']):
    #         cand = ctx["text"].strip()
    #         if cand == target:
    #             for j in range(rank, max_k): scores[idx][j] = 1
    #
    #
    # for i in range(max_k):
    #     EM = sum([score[i] for score in scores]) / len(retrieved_code)
    #     print("At top ", i, " EM/Recall: ", EM * 100)

    for idx, (id, ex) in enumerate(examples.items()):
        average_precision = 0
        if max_k==0:
            max_k = len(ex["ctxs"])
            scores = [[0 for i in range(max_k)] for j in range(len(examples))]
        if ex["found"]:
            for j, hit in enumerate(ex["ctxs"]):
                if hit["document_title"] == id:
                    average_precision = 1 / (j + 1)
                    for k in range(j, max_k): scores[idx][k] = 1
                    break
        map += average_precision


    for i in range(max_k):
        EM = sum([score[i] for score in scores]) / len(examples)
        print("At top ", i, " EM/Recall: ", EM * 100)

    return map / len(examples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Log JSON file')
    args = parser.parse_args()

    map = MAP(args.input_file)
    print("MAP - ", map)
