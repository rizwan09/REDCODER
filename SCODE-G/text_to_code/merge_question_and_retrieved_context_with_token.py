import argparse
import json
import os
import shutil
from subprocess import check_output
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--q', help='Input Question file', required=True)
    parser.add_argument('--r', help='Input Retrievd results file', required=True)
    parser.add_argument('--o', help='Output merged Json file', required=True)
    parser.add_argument('--t', help='top_k', type=int, required=True)

    args = parser.parse_args()

    q_json = args.q
    ctx_json = args.r

    q_data = json.load(open(q_json))
    o_data = json.load(open(ctx_json))
    len_=  len(q_data)
    assert len(q_data) == len_
    with tqdm(total=len_, desc='Processing') as pbar:
        for i in range(len_):
            o_data[i]["question"] = q_data[i]["question"]
            o_data[i]["ctxs"] = o_data[i]["ctxs"][:args.t]
            pbar.update()

    output_file = open(args.o, 'w')
    json.dump(o_data, output_file)
    output_file.close()
    pass

# python merge_question_and_retrieved_context_with_token.py  \
#   --q /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/test_10.json \
#   --r /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/test_100.json_2.json \
#   --o /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/test_parsed_top_2.json --t 2

#
#
# python merge_question_and_retrieved_context_with_token.py  \
#   --q /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/train_10.json \
#   --r /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/train_100.json_2.json \
#   --o /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/train_parsed_top_2.json --t 2

#
# python merge_question_and_retrieved_context_with_token.py  \
#   --q /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/valid_10.json \
#   --r /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/valid_100.json_2.json \
#   --o /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/valid_parsed_top_2.json --t 2
#



