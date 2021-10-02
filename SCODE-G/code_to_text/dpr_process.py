import re
import argparse
import subprocess
import json
from tqdm import tqdm
from pathlib import Path
import nltk
import numpy as np
import random
import sys

sys.path.append(".")
sys.path.append("..")

from tokenizer.tokenizer import tokenize_java, tokenize_python


try:
    vocabs = [line.split()[0].strip() for line in open('/local/wasiahmad/codebart/vocab').readlines()][:40000]
except:
    vocabs = [line.split()[0].strip() for line in open('/home/rizwan/DPR_models/vocab').readlines()][:40000]
print("last 20 vocabs: ", vocabs[-20:])
print("#vocab: ", len(vocabs))

def get_masked_sent(sent, mask_rate=0.15, debug=False):
    """
    Adapted from https://github.com/huggingface/transformers/blob/f9cde97b313c3218e1b29ea73a42414dfefadb40/examples/lm_finetuning/simple_lm_finetuning.py#L276-L301
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    if debug:
        import ipdb
        ipdb.set_trace()
    tokens = sent.split()
    # print('unmasked tokens: ', tokens)
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_rate:
            prob /= mask_rate

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(vocabs)[0].replace('@', '')

            # -> rest 10% randomly keep current token

        else:
            pass
    # print('masked tokens: ', tokens)
    return ' '.join(tokens)

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

    def write(source1, target, src_writer=src_writer, tgt_writer=tgt_writer, source2=None):
        source = re.sub("[\n\r\t ]+", " ", source1)
        target = re.sub("[\n\r\t ]+", " ", target)
        src_writer.write(source + '\n')
        tgt_writer.write(target + '\n')

        if source2:
            src_writer.write(source2 + '\n')
            tgt_writer.write(target + '\n')



    for idx, ex in enumerate(tqdm(retrieved_code, total=len(retrieved_code))):
        source = ''
        try:
            target = ex['answers']
        except:
            ex=retrieved_code[ex]
            target = ex["answers"]
        # print("target: ", target, flush=True)

        assert len(ex['ctxs']) >= args.top_k

        if (args.top_k<0 or args.NO_CONCODE_VARS) :
             source = source.split('concode_field_sep')[0]
        if args.top_k > 0:
            inserted =  0
            for rank, ctx in enumerate(ex['ctxs']):
                if args.WITH_WITHOUT_SUFFIX != "with":
                    if target.strip() != ctx[
                        "text"].strip():  # for retrieving without ref code but includes other codes in the test corpus
                            source = ctx["text"]
                            break
                else:
                    source = ctx["text"]
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
    parser.add_argument("--mask_rate", type=float, help='masking words ratio', default=0.15)
    parser.add_argument("--NO_CONCODE_VARS", action="store_true", help='Do not use concode_filed_sep variables')
    parser.add_argument("--UNION", action="store_true", help='UNION of Retrieved and gold vars')
    parser.add_argument("--ONLY_RETRIEVAL", action="store_true", help='ONLY_RETRIEVED CODE will be used as source')
    parser.add_argument("--dag", action="store_true", help='ONLY_RETRIEVED CODE will be used as source')
    parser.add_argument("--WITH_WITHOUT_SUFFIX",type=str, default='no', help='WITH_WITHOUT_SUFFIX')
    parser.add_argument("--lang",type=str, default='python', help='WITH_WITHOUT_SUFFIX')
    args = parser.parse_args()
    # print("args: ", flush=True)
    # print("--"*50, flush=True)
    # print(args, flush=True)
    # print("--" * 50, flush=True)
    main(args)
