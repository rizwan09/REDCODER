import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

lang='iclr_c'


CHECKPOINT='/home/rizwan/DPR_models/biencoder_graphcode_models_iclr_code_text/dpr_biencoder.14.5270'

pretrained_model="/home/rizwan/graphcodebert-base"


OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_graphcode_models_iclr_code_text/"



OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/biencoder_graphcode_models_iclr_code_text/"+'encoddings_iclr_only_deduplicated.summaries.txt_0.pkl'
splits=["in_domain_train", "in_domain_valid", "overall_test", "in_domain_test", "out_domain_test"]

# if split=="test": qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/iclr_c_sum_data/overall_test.jsonl"
# if split=="train": qa_file = "/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/iclr_c_sum_data/in_domain_train.jsonl"
# if split=="valid": qa_file ="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/iclr_c_sum_data/in_domain_valid.jsonl"

for split in splits:

    qa_file ="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/iclr_c_sum_data/"+split+".jsonl"



    CTX_FILE =  FILE = './iclr_only.deduplicated.summaries.txt'


    print(qa_file)

    DEVICES = [5]
    CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])


    top_k =  100

    command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
              ' python dense_retriever.py  --model_file '+ CHECKPOINT + \
              '  --ctx_file  '+ CTX_FILE + \
              '  --qa_file '+ qa_file + \
              '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILES + \
              '  --out_file  '+ OUTPUT_DIR_PATH+"iclr_code_text_retrieval_iclr_only_dedup_"+split+"_"+str(top_k)+".json" \
              '  --n-docs  '+ str(top_k) + ' --sequence_length 256 --save_or_load_index --code_to_text --dataset ICLR'
    # print (command)
    print(command, flush=True)
    os.system(command)



