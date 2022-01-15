# SCODE-R (Summary and CODE Retriever)

This is repository for the SCODE-R retriever in the [Retrieval Augmented Code Generation and Summarization](https://arxiv.org/abs/2108.11601) paper.

If you find this paper or this code useful, please cite this paper:
```
@inproceedings{parvez2021retrieval,
  title = {Retrieval Augmented Code Generation and Summarization},
  author = {Parvez, Md Rizwan and Ahmad, Wasi Uddin, and Chakraborty, Saikat and Ray, Baishakhi and Chang, Kai-Wei},
  booktitle = {EMNLP-Findings},
  year = {2021}
}
```



## Features
1. SCODE-R retriever model is based on bi-encoder architecture in [DPR](https://arxiv.org/abs/2004.04906).
2. Trains only using in batch negatives (the hard negatives are not considered). For details see [our paper](https://arxiv.org/abs/2108.11601).

## Installation

We tested with pytorch 1.7.1 Huggingface Transformers 3.0.2 (Provided in the root dir for convenience)

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone https://github.com/rizwan09/REDCODER.git
cd REDCODER/transformers-3.0.2
pip install -e .
cd ../
cd SCODE-R
pip install -e .
```
Install [apex](https://github.com/nvidia/apex#quick-start) for fp16 training.


DPR is tested on Python 3.6+, PyTorch 1.7.1+ and with Huggingface Transformers 3.0.2 (added in the [root dir](https://github.com/rizwan09/REDCODER/tree/main/transformers-3.0.2_alignment_proj))
DPR relies on third-party libraries for encoder code implementations.
It currently supports Huggingface RoBERTa encoder models.
Due to generality of the tokenization process, DPR uses Huggingface tokenizers as of now. So Huggingface is the only required dependency.




## SCODE-R input data format (if you want to know)
We support the datsets Concode, Conala, and the CodeXGLUE-CSNET (jsonl) formateted dataset. 
The corresponding data read and pre-processing codes are provided: (i) reading during training time  [here](https://github.com/rizwan09/REDCODER/blob/main/SCODE-R/dpr/utils/data_utils.py#L130).
(ii) reading during Encodding the candidates [here](https://github.com/rizwan09/REDCODER/blob/main/SCODE-R/generate_dense_embeddings.py#L119).
(iii) a: Reading at inference time: question file  [here](https://github.com/rizwan09/REDCODER/blob/main/SCODE-R/dense_retriever.py#L99),
b: inference time: candidate file reading [here](https://github.com/rizwan09/REDCODER/blob/main/SCODE-R/dense_retriever.py#L99). The reading should be similar to the reading of step as (ii).
Below for each datset, we provide the execution commands.

## Step-1: SCODE-R training
Retriever training quality depends on its effective batch size. The one reported in the paper used 2/3/4 x 12GB GPUs.
In order to start training on one machine with multigpus:
```bash
DEVICES={GPU IDs}
NUM_DEVICES={NUM of DEVICES}
PRETRAINED_MODEL_PATH={local file path of [hf_graphcoebert-base/hf_codebert-base] as discussed below e.g., ../redcoder_data/graphcodebert-base}
TRAIN_FILE_PATH={train filepath e.g., ../redcoder_data/codexglue_csnet/java or python/train.jsonl}
DEV_FILE_PATH={dev filepath e.g., ../redcoder_data/codexglue_csnet/java or python/dev.jsonl}
OUTPUT_DIR={dir to save checkpoints specially the best one}

CUDA_VISIBLE_DEVICES=${DEVICES} python -m torch.distributed.launch 
     --nproc_per_node=${NUM_DEVICES} train_dense_encoder.py 
     --max_grad_norm 2.0 
     --encoder_model_type hf_roberta 
     --pretrained_model_cfg ${PRETRAINED_MODEL_PATH}
     --eval_per_epoch 1 
     --seed 12345 
     --sequence_length 256 
     --warmup_steps 1237 
     --batch_size 8 
     --train_file ${TRAIN_FILE_PATH}
     --dev_file ${DEV_FILE_PATH}
     --output_dir ${OUTPUT_DIR}
     --learning_rate 2e-5 
     --num_train_epochs 15 
     --dev_batch_size 64 
     --val_av_rank_start_epoch 0 
     --fp16 
```

Notes:
- We need to create the local filepath of the pretrained model. The reason is they are not defined in this earlier version of Huggingface Transformers.
To create the local file for the pretrained model, please download: [[hf_graphcoderbert-base](https://huggingface.co/microsoft/graphcodebert-base)|[hf_codebert-base](https://huggingface.co/microsoft/codebert-base)]
To make it easy we also added it in our redcoder_data.

- The default setting is for code->text (code sum.) task.
- Use ```--text_to_code ``` for text->code (code gen) task.
- For (CodeXGLUE-CSNET) dataset:
     - The best results are found by using graphcodebert encoder. ```--pretrained_model_cfg {local file path of hf_graphcoebert}```. 
     - For Scode-R in Redcoder:
         - use: ```train_dense_encoder.py```  
     - For  Scode-R in  Redcoder-ext:
         - use: ```train_csnet_with_comments.py``` 
- For ```Concode``` dataset:
    - The best results are found by using codebert encoder. ```--pretrained_model_cfg {local file path of hf_codebert}```. 
    - use additional dataset flag: ```--dataset CONCODE```
    - By default, we did not change the original input data format which consists of a (NL + Code Env Variable -> Code) 
        - So we use an additional flag ```--concode_with_code``` 
        - For (NL -> Code) do not use this ```--concode_with_code``` flag.


## Step-2: SCODE-R candidate embedding

```bash
CHECKPOINT={retirver best checkpint from previous step e.g., ../redcoder_data/checkpint/codexglue_csnet_java or python_scoder_text_to_code.cp}
CANDIDATE_FILE={retrieval database file path; this file releasled with official CSNET; e.g., ../redcoder_data/retrieval_database/java or python_dedupe_definitions_v2.pkl}
DEVICES={GPU IDs}
NUM_DEVICES={NUM of DEVICES}
ENCODDING_CANDIDATE_PREFIX = {OUTPUT DIR/encoddings_${candidate_file}}
PRETRAINED_MODEL_PATH={local file path of [hf_graphcoebert-base/hf_codebert-base] as discussed above}
          

CUDA_VISIBLE_DEVICES=${DEVICES} python  -m torch.distributed.launch 
          --nproc_per_node=${NUM_DEVICES} generate_dense_embeddings.py 
          --model_file  ${CHECKPOINT}  
          --encoder_model_type hf_roberta
          --pretrained_model_cfg  ${PRETRAINED_MODEL_PATH}
          --batch_size 512 
          --ctx_file  ${CANDIDATE_FILE}
          --shard_id 0 
          --num_shards 1 
          --out_file  ENCODDING_CANDIDATE_PREFIX
```

So far we support the following dataset format:
- Concode:
    - same ```train/dev/test.json``` used in training the retriever model in the above step (that is the original Concode splits).
    - for each of ```train```, ```dev```, ```test``` need to encode seperately
    - When doing inference (discussed below) just need to provide (i) one raw_data_file_path's prefix like ```concode/``` and ```concode/*``` will be used (ii) one encoded_data_file_path  like ```concode/encoded_``` and ```concode/encoded_*``` will be used
    - Use additional flag ```--dataset CONCODE```
- CodeXGLUE-CSNET:
    - ```candidate_file``` can also be downloaded from ```https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python,java.zip```
    
- SCODE-R encoding is same for both Redcoder and Redcoder-ext. We only retrieve based on the candidate. It's paired data is only used in generation. So before the next step "inference", both are same.
- The default setting is for text->code (code gen) task.
- Use ```--code_to_text``` for code->text (code sum.) task.
- For code->text we use use the same candidate_file or in our paper we retrieved from a collection: CoeXGLUE-CSNET(trainsets)+[C_summarization dataset](https://openreview.net/pdf?id=zv-typ1gPxA)


## Step-3: SCODE-R inference (retrieve):

```bash
TOP_K=200
RETRIEVAL_RESULT_FILE={output file path like OUTPUT_DIR_PATH+split+_+${TOP_K}+_+{code_totext or text_to_code}+.json e.g., ../redcoder_data/retriever_output/csnet_text_to_code/with_comments/python_csnet_pos_only_retrieval_dedup_test_30.json}
CHECKPOINT={retirver best checkpint from first step e.g., ../redcoder_data/checkpint/codexglue_csnet_java or python_scoder_text_to_code.cp}
CANDIDATE_FILE={java/python_dedupe_definitions_v2.pkl file path this file reasled with official CSNET e.g., ../redcoder_data/retrieval_database/java or python_dedupe_definitions_v2.pkl}
ENCODDING_CANDIDATE_PREFIX = {OUTPUT DIR/encoddings_${candidate_file}}
PRETRAINED_MODEL_PATH={local file path of [hf_graphcoebert/hf_codebert] as discussed above}
FILE_FOR_WHICH_TO_RETIRVE={each of train/dev/test filepath e.g., CodeSearchNet/java or python/split.jsonl}

CUDA_VISIBLE_DEVICES=${SINGLE_GPU_DEVICE_IS_ENOUGH} python {dense_retriever.py | dense_retriever_with_comments.py}
            --model_file ${CHECKPOINT}
            --ctx_file  ${CANDIDATE_FILE}
            --qa_file ${FILE_FOR_WHICH_TO_RETIRVE}
            --encoded_ctx_file {encoded document files glob expression e.g., ENCODDING_CANDIDATE_PREFIX}
            --out_file  ${RETRIEVAL_RESULT_FILE}
            --n-docs  ${TOP_K}
            --sequence_length 256 
            --save_or_load_index 
```
Notes:
    - For code sum (code->text) use additional ```--code_to_text``` 

The tool writes retrieved results for subsequent reader model training into specified out_file.
It is a json with the following format:

```
[
    {
        "question": "This is input text (for text->code); code for (code->text)",  
        "answers": the target seq code (for text->code); text for (code->text), 
        "ctxs": [
            {
                "id": "...", # passage id of the retrived candidates from CANDIDATE_FILE
                "title": null, (for Concode, and for redcoder-ext this is: candidate's paired data)
                "text": candidate code (for text->code); candidate text for (code->text), (for CodeXGLUE-CSNET, and for redcoder-ext this is: candidate _NL_ paired data)
                "score": "...",  # retriever score
                "has_answer": {true|false please ignore this as we did not process it}
     },
]
```



## Reference

If you plan to use `SCODE-R` in your project, please consider citing [our paper](https://arxiv.org/abs/2108.11601):
```
@inproceedings{parvez2021retrieval,
  title = {Retrieval Augmented Code Generation and Summarization},
  author = {Parvez, Md Rizwan and Ahmad, Wasi Uddin, and Chakraborty, Saikat and Ray, Baishakhi and Chang, Kai-Wei},
  booktitle = {EMNLP-Findings},
  year = {2021}
}
```
## Misc
Our code is build upon modification of [DPR codebase](https://github.com/facebookresearch/DPR). 
We thank the authors of DPR, Huggingface, Pytorch, and so on. 

## License
Redcoder MIT license. 
