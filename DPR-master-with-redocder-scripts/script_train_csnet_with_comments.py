import os

# lang = ["go",  "java",  "javascript",  "php",  "python",  "ruby"]
lang = "python"
# lang = "java"
# lang = "javascript"
# lang = "go"

text_to_code=True

CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code_with_comments/"+lang #+"_new/"


pretrained_model="/home/rizwan/graphcodebert-base/"

DEVICES=[6,7]
CUDA_VISIBLE_DEVICES=','.join([str(i) for i in DEVICES])
mport = 2234


command = "CUDA_VISIBLE_DEVICES="+CUDA_VISIBLE_DEVICES \
     +" python -m torch.distributed.launch --nproc_per_node="+str(len(DEVICES)) \
      + ' --master_port=' + str(mport) \
      +" train_dense_encoder_with_comments.py \
     --max_grad_norm 2.0 \
     --encoder_model_type hf_roberta \
     --pretrained_model_cfg " + pretrained_model   + " \
     --eval_per_epoch 1 \
     --seed 12345 \
     --sequence_length 256 \
     --warmup_steps 1237 \
     --batch_size 8 \
     --train_file /home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/train.jsonl   \
     --dev_file /home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/" + lang +"/valid.jsonl \
     --output_dir "+CHECKPOINT_DIR_PATH+" \
     --learning_rate 2e-5 \
     --num_train_epochs 15 \
     --dev_batch_size 64 \
     --val_av_rank_start_epoch 0 \
     --fp16 \
    "
if text_to_code: command+= ' --text_to_code '
print (command, flush=True)
os.system(command)






