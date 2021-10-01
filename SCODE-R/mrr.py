# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from https://github.com/microsoft/CodeXGLUE/blob/main/Text-Code/NL-code-search-Adv/evaluator/evaluator.py
import logging
import sys, json
import numpy as np


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            answers[str(idx)] = str(idx)
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            predictions[str(idx)] = js['answers']
    return predictions


def calculate_scores(answers, predictions):
    scores = []
    for key in answers:
        # import ipdb
        # ipdb.set_trace()
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        flag = False
        for rank, idx in enumerate(predictions[key]):
            if idx == answers[key]:
                scores.append(1 / (rank + 1))
                flag = True
                break
        if flag is False:
            scores.append(0)
    result = {}
    result['MRR'] = round(np.mean(scores), 4)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for POJ-104 dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    print("reading gold answers")
    answers = read_answers(args.answers)
    print("reading predcited answers")
    predictions = read_predictions(args.predictions)
    print("computing scores")
    scores = calculate_scores(answers, predictions)
    print(scores)


if __name__ == '__main__':
    main()
    # python mrr.py -a /home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.test.jsonl -p /home/rizwan/DPR/predictions_KP20k.jsonl





