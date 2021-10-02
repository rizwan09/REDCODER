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


def main(args):

    input_source_file=args.input_data_dir+"/test.source"
    output_source_file=args.output_data_dir+"/test.source"
    input_target_file=args.input_data_dir+"/test.target"
    output_target_file=args.output_data_dir+"/test.target"

    line_c = 0
    with open(input_source_file) as input_source_f, open(input_target_file) as input_target_f, \
            open(output_source_file, 'w') as output_source_f, open(output_target_file, 'w') as output_target_f:
        for source, target in zip(input_source_f, input_target_f):
            source = source.split('_CODE_SEP_')[1]
            output_source_f.write(source.strip()+"\n")
            output_target_f.write(target.strip()+"\n")
            line_c+=1


    print('line_c: ', line_c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", required=True, help='path to .json file')
    parser.add_argument("--output_data_dir", required=True, help='path to .json file')
    args = parser.parse_args()
    main(args)
