import re
import argparse
import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import nltk
import numpy as np
import random


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').split(' ')
    return int(num[0])


def main(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(args.retrieved_code_file) as f:
        retrieved_code = json.load(f)

    src_writer = open('{}/{}.source'.format(args.out_dir, args.split), 'w', encoding='utf8')
    tgt_writer = open('{}/{}.target'.format(args.out_dir, args.split), 'w', encoding='utf8')

    print("topk: ", args.top_k, " args.UNION: ", args.UNION, flush=True)


    for idx, ex in enumerate(tqdm(retrieved_code, total=len(retrieved_code))):

        try:
            source = ex['question']
        except:
            ex = retrieved_code[ex]

        source = ex['question']
        target = ex['answers']
        # assert len(ex['ctxs']) >= args.top_k


        if (args.top_k<0 or args.NO_CONCODE_VARS) :
             source = source.split('concode_field_sep')[0]
        if args.top_k > 0:
            inserted =  0

            for rank, ctx in enumerate(ex['ctxs']):
                # if "test.json" not in ctx["id"] and target.strip()!=ctx["text"].strip(): #for retrieving without test corpus
                if args.WITH_OR_WITHOUT_REF=="with":  #for retrieving without ref code but includes other codes in the test corpus
                    source += ' _CODE_SEP_ ' + (ctx["text"])
                    inserted+=1
                    if inserted>= args.top_k:
                        break
                else:
                    if target.strip() != ctx["text"].split('_NL_')[0].strip():
                        source += ' _CODE_SEP_ ' + ctx["text"]
                        inserted += 1
                        if inserted >= args.top_k:
                            break


        source = re.sub("[\n\r\t ]+", " ", source)
        target = re.sub("[\n\r\t ]+", " ", target)
        src_writer.write(source + '\n')
        tgt_writer.write(target + '\n')

    src_writer.close()
    tgt_writer.close()

    print ("written to: ", src_writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved_code_file", required=True, help='path to .json file')
    parser.add_argument("--split", required=True, choices=['train', 'valid', 'test'])
    parser.add_argument("--top_k", type=int, default=0, help='number of retrieved code to consider, -1 means only NL before concode_field_sep')
    parser.add_argument("--out_dir", required=True, help='directory path to save data')
    parser.add_argument("--mask_rate", type=float, help='masking words ratio', default=0.0)
    parser.add_argument("--NO_CONCODE_VARS", action="store_true", help='Do not use concode_filed_sep variables')
    parser.add_argument("--UNION", action="store_true", help='UNION of Retrieved and gold vars')
    parser.add_argument("--ONLY_RETRIEVAL", action="store_true", help='ONLY_RETRIEVED CODE will be used as source')
    parser.add_argument("--dag", action="store_true", help='ONLY_RETRIEVED CODE will be used as source')
    parser.add_argument("--WITH_OR_WITHOUT_REF", type=str, help='WITH_OR_WITHOUT_REF')
    args = parser.parse_args()
    main(args)