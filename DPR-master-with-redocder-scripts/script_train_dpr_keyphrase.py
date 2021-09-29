import os

CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/KP20k"
CHECKPOINT_DIR_PATH="/home/rizwan/DPR_models/biencoder_models_with_concode_tokens/KP20k/dpr_biencoder.9.15932"

pretrained_model="bert-base-uncased"
DEVICES=[0,2,6,7]
CUDA_VISIBLE_DEVICES=','.join([str(i) for i in DEVICES])

command = "CUDA_VISIBLE_DEVICES="+CUDA_VISIBLE_DEVICES+" python -m torch.distributed.launch --nproc_per_node="+str(len(DEVICES))+" train_dense_encoder.py \
 --max_grad_norm 2.0 \
 --encoder_model_type hf_bert \
 --pretrained_model_cfg " + pretrained_model   + " \
 --eval_per_epoch 1 \
 --seed 12345 \
 --sequence_length 256 \
 --warmup_steps 1237 \
 --batch_size 8 \
 --train_file /home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.train.jsonl   \
 --dev_file /home/wasiahmad/workspace/projects/NeuralKpGen/data/scikp/kp20k_separated/KP20k.valid.jsonl   \
 --output_dir "+CHECKPOINT_DIR_PATH+" \
 --learning_rate 2e-5 \
 --num_train_epochs 20 \
 --dev_batch_size 16 \
 --val_av_rank_start_epoch 1 \
 --fp16 --dataset KP20k \
"

print (command, flush=True)
os.system(command)






