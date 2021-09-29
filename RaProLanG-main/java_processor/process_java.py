from multiprocessing import Pool, cpu_count, current_process

import argparse
import json
import os
import shutil
from subprocess import check_output
from tqdm import tqdm

tmp_dir_name = "/local/rizwan/tmp_code_process"
TIME_LIMIT_FOR_EACH_RETRIEVED_CODE_PARSE = 3  # (in seconds)


def extract_identifiers(code_text, file_name=None):
    if not os.path.exists(tmp_dir_name):
        os.mkdir(tmp_dir_name)
    java_file_path = os.path.join(tmp_dir_name, 'A.java' if file_name is None else file_name)
    java_file = open(java_file_path, 'w')
    java_code = "class A { \n" + code_text + "\n}"
    java_file.write(java_code)
    java_file.close()
    command = ["java", "-jar", "Tokenizer.jar", java_file_path]
    stdout = check_output(command, timeout=TIME_LIMIT_FOR_EACH_RETRIEVED_CODE_PARSE)
    output = stdout.decode('utf-8')
    if 'timed out' in output:
        os.remove(java_file_path)
        raise TimeoutError('Time Out')
    tokens = output.split("\t")
    return_tokens = []
    for token in tokens:
        parts = token.split(" ")
        if len(parts) == 2:
            # First element is type, second element is token
            return_tokens.append([parts[0].strip(), parts[1].strip()])
            pass
    os.remove(java_file_path)
    return return_tokens


class MultiprocessingEncoder(object):
    def __init__(self, top_k):
        self.topk = top_k
        pass

    def initializer(self):
        pass

    def extract_tokens(self, data_pointwith_id):
        assert isinstance(data_pointwith_id, tuple)
        i = data_pointwith_id[0]
        data_point = data_pointwith_id[1]
        retrieved_codes = data_point['ctxs']
        # print("top_k: ", self.topk, flush=True)
        for j in range(min(self.topk, len(retrieved_codes))):
            code_point = retrieved_codes[j]
            code_text = code_point['text']
            file_name = "A_" + str(i) + "_" + str(j) + ".java"
            try:
                identifiers = extract_identifiers(code_text, file_name)
            except Exception as ex:
                print('Error in example %d retrieved code %d' % (i, j), ex, flush=True)
                identifiers = []
            code_point['identifiers'] = identifiers
            retrieved_codes[j] = code_point
            pass
        data_point['ctxs'] = retrieved_codes[:self.topk]
        return data_point


if __name__ == '__main__':
    if os.path.exists(tmp_dir_name):
        shutil.rmtree(tmp_dir_name)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input Json file', required=True)
    parser.add_argument('--top_k', type=int, help='top-K retrieved passages to process. '
                                        'By default (-1) the code will process all', default=-1)
    parser.add_argument('--output', help="Output path\nLeave Black, fi you want to replace the input", default="")
    parser.add_argument('--workers', default=cpu_count())
    args = parser.parse_args()
    input_json = args.input
    output_json = args.output
    if output_json == "":
        output_json = input_json + "_"+str(args.top_k)+".json"
        pass
    input_data = [(i, dp) for i, dp in enumerate(json.load(open(input_json)))]
    encoder = MultiprocessingEncoder(args.top_k)
    pool = Pool(args.workers, initializer=encoder.initializer)
    processed_dataset = []
    len_=len(input_data)
    with tqdm(total=len_, desc='Processing') as pbar:
        for i, ex in enumerate(pool.imap(encoder.extract_tokens, input_data, 100)):
            pbar.update()
            processed_dataset.append(ex)
    output_file = open(output_json, 'w')
    json.dump(processed_dataset, output_file)
    output_file.close()
    if os.path.exists(tmp_dir_name):
        shutil.rmtree(tmp_dir_name)
    pass

#     python process_java.py --input /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/train_100.json --top_k 2 &
#     python process_java.py --input /local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/test_100.json --top_k 2 &

