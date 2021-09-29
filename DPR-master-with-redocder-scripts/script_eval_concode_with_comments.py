import os

lang='java'


# CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/dpr_biencoder.5.1786"
CHECKPOINT="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/java/dpr_biencoder.6.2084"
CHECKPOINT="//home/rizwan/DPR_models/biencoder_models_with_concode_tokens_graphcodebert/java/dpr_biencoder.9.12500"

pretrained_model="microsoft/codebert-base"

# OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/"
OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_concode_with_code_tokens/java/"
OUTPUT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_concode_with_code_tokens_graphcodebert/java/"
BASE_COMMENT_DIR='/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/'
# OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/github_encoddings_conode/java/concode_train.json_0.pkl"
# OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/github_encoddings_conode/java/concode_with_code_token_train.json_"
OUTPUT_ENCODED_FILES = "/local/rizwan/DPR_models/github_encoddings_conode/java/concode_with_code_token_"
# ctx_file="/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/train.json"
ctx_file="/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode/"



dataset='CONCODE'


# pretrained_model="microsoft/codebert-base"
pretrained_model="/home/rizwan/graphcodebert-base/"


qa_file_suffix=["train", "valid", "test"][2]
qa_file = BASE_COMMENT_DIR+str(qa_file_suffix)+'.json'
print(qa_file)

DEVICES = [2]
CUDA_VISIBLE_DEVICES = ','.join([str(i) for i in DEVICES])

top_k =  100

command = 'CUDA_VISIBLE_DEVICES=' + CUDA_VISIBLE_DEVICES + \
          ' python ' + \
          ' dense_retriever_with_comments.py  --model_file '+ CHECKPOINT + \
          '  --ctx_file  '+ str(ctx_file) + \
          '  --qa_file '+ qa_file + \
          '  --encoded_ctx_file ' + str(OUTPUT_ENCODED_FILES) + \
          '  --out_file  '+ OUTPUT_DIR_PATH+str(qa_file_suffix)+"_"+str(top_k)+".json" \
          '  --n-docs  '+ str(top_k) + ' --batch_size 64 --match exact --sequence_length 256 --concode_with_code --save_or_load_index --dataset '+dataset
# print (command)
# \ # ' -m torch.distributed.launch --nproc_per_node='+str(len(DEVICES)) + \
print(command, flush=True)
os.system(command)


