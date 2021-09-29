import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]

lang='python'
lang='java'

CHECKPOINT="/home/rizwan/DPR_models/biencoder_models/python/dpr_biencoder.2.6441"
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models/java/dpr_biencoder.1.7101"

pretrained_model="microsoft/codebert-base"


GITHUB_DB="/local/wasiahmad/github_data/"+lang+"/" #local in NLP10
ENCODDING_DIR ='/local/rizwan/DPR_models/github_encoddings/'+lang+"/"
OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models/"+lang+"/"
BASE_COMMENT_DIR='/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/code-to-text/'+lang+"/"


pretrained_model="microsoft/codebert-base"

files = 'test.functions_class.tok ' \
        'test.functions_standalone.tok ' \
        'valid.functions_class.tok  ' \
        'valid.functions_standalone.tok ' \
        'train.0.functions_class.tok ' \
        'train.1.functions_class.tok ' \
        'train.2.functions_class.tok ' \
        'train.3.functions_class.tok ' \
        'train.4.functions_class.tok ' \
        'train.5.functions_class.tok ' \
        'train.6.functions_class.tok ' \
        'train.7.functions_class.tok ' \
        'train.0.functions_standalone.tok ' \
        'train.1.functions_standalone.tok ' \
        'train.2.functions_standalone.tok ' \
        'train.3.functions_standalone.tok ' \
        'train.4.functions_standalone.tok ' \
        'train.5.functions_standalone.tok ' \
        'train.6.functions_standalone.tok ' \
        'train.7.functions_standalone.tok'.split()



GITHUB_RAW_FILES = GITHUB_DB+'*7.functions_standalone.tok'

# GITHUB_RAW_FILES = '/local/rizwan/DPR_models/github_encoddings/python/'+'*7.functions_standalone.tok'

OUTPUT_ENCODED_FILES = ENCODDING_DIR + 'github_encoddings_*.pkl'  #+file_name
# OUTPUT_ENCODED_FILES = '/local/rizwan/DPR_models/github_encoddings/python/github_encoddings_*'  #+file_name


qa_file_suffix=["train", "valid", "test"][2]
qa_file = BASE_COMMENT_DIR+str(qa_file_suffix)+'.jsonl'
print(qa_file)

DEVICES = [0,1,2,5,6,3,4]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

top_k =  100
OUT_TOP_K_FILE = OUTPUT_DIR_PATH+CHECKPOINT.split()[-1]

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python  -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
          ' dense_retriever.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ GITHUB_RAW_FILES + \
          '  --qa_file '+ qa_file + \
          '  --encoded_ctx_file ' + OUTPUT_ENCODED_FILES + \
          '  --out_file  '+ OUTPUT_DIR_PATH+str(qa_file_suffix)+"_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --sequence_length 256 --save_or_load_index '
# print (command)
print(command, flush=True)
os.system(command)




