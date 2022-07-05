# SCODE-G (Summary and CODE Generator)

This is repository for the SCODE-G generator in the [Retrieval Augmented Code Generation and Summarization](https://arxiv.org/abs/2108.11601) paper.

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
1. SCODE-G generator model is based on [PLBART](https://arxiv.org/abs/2103.06333).
2. As input, SCODE-G takes the original input text/code and its top-k unimoal or bimoal retrived candidates. For details see [our paper](https://arxiv.org/abs/2108.11601).


## SCODE-G Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone https://github.com/rizwan09/REDCODER.git
cd REDCODER/SCODE-G
bash install_tools.sh
```
Install [apex](https://github.com/nvidia/apex#quick-start) for fp16 training.


## SCODE-G input data format
- Our input is the output from the SCODE-R. (e.g., ../redcoder_data/retriever_output/csnet_text_to_code/with_comments/java_csnet_pos_only_retrieval_dedup_test_30.json)

- It is a json with the following format:


```
[
    {
        "question": "This is input text (for text->code); code for (code->text)",
        "answers": the target seq code (for text->code); text for (code->text),
        "ctxs": [
            {
                "id": "...", # passage id of the retrived candidates from CANDIDATE_FILE
                "title": null, (for Concode with redcoder-ext this is: candidate's paired data)
                "text": candidate code (for text->code); candidate text for (code->text), (for CodeXGLUE-CSNET, and for redcoder-ext this is: candidate _NL_ paired data)
                "score": "...",  # retriever score
                "has_answer": {true|false please ignore this as we did not process it}
     },
]
```



## SCODE_G training

### Step1. Download PLBART checkpoint

```bash
bash download_pbart.sh
cd ..
```
You can move it to your desired directory.


### Step2: Data and Preprocessing

####  SCODE-G for text to code:
```bash
cd SCODE-G/text_to_code
```
##### Input data pre-processing:
- For CodeXGLUE-CSNET:
```bash
LANG={language java/python}
top_k={top-K candidates e.g., 4 for java 5 for python}
WITH_OR_WITHOUT_REF={with/no} # with mean keeping the target if retrieved as a candidate no means filtering target from retrieval
RETDIR={reteved output/input to SCODE-G directory  e.g., /redcoder_data/retriever_output/codexglue_csnet_text_to_code/}
SAVE_DIR={preprocssed output directory e.g., ../redcoder_data/codexglue_csnet_text_to_code_scode-g-preprocessed-input/}



bash {prepare_csnet_redcoder_ext.sh|prepare_csnet_redcoder.sh} ${LANG} ${top_k} ${WITH_OR_WITHOUT_REF} ${RETDIR} ${SAVE_DIR}
```
- Notes:
    - The retrieved directory ${RETDIR} should have files like  
      * ${RETDIR}/${LANG}\_csnet_pos_only_retrieval_dedup_${split}_30.json
      
    - For REDCODER: use ```prepare_csnet_redcoder.sh``` and for RECODER_ext  use ```prepare_csnet_redcoder_ext.sh```.

- For Concode:
```bash
LANG=java #must be java for Concode
top_k={top-K candidates e.g., 4 for java 5 for python}
WITH_OR_WITHOUT_REF={with/no} # with mean keeping the target if retrieved as a candidate no means filtering target from retrieval
RETDIR={reteved output/input to SCODE-G directory  e.g., /redcoder_data/retriever_output/cooncode/}
SAVE_DIR={preprocssed output directory e.g., ../redcoder_data/concode_scode-g-preprocessed-input/}



bash {prepare_concode_redcoder_ext.sh|prepare_concode_redcoder.sh} ${LANG} ${top_k} ${WITH_OR_WITHOUT_REF} ${RETDIR} ${SAVE_DIR}

```
- Notes:
    - The retrieved directory ${RETDIR} should contain files like  
      * ${RETDIR}/${split}_20.json
    - For REDCODER: use ```prepare_concode_redcoder.sh``` and for RECODER-ext:  use ```prepare_concode_redcoder_ext.sh```.



#### SCODE-G for code to text:

```bash
cd SCODE-G/code_to_text
```
##### Input data pre-processing:
- Currently we only support CodeXGLUE-CSNET:
```bash
LANG={language java/python}
top_k={top-K candidates e.g., 4 for java 5 for python}
WITH_OR_WITHOUT_REF={with/no} # with mean keeping the target if retrieved as a candidate no means filtering target from retrieval
RETDIR={reteved output/input to SCODE-G directory  e.g., /redcoder_data/retriever_output/codexglue_csnet_code_to_text/}
SAVE_DIR={preprocssed output directory e.g., ../redcoder_data/codexglue_csnet_code_to_text_scode-g-preprocessed-input/}



bash {prepare_csnet_redcoder_ext.sh|prepare_csnet_redcoder.sh} ${LANG} ${top_k} ${WITH_OR_WITHOUT_REF} ${RETDIR} ${SAVE_DIR}
```
- Notes:
    - The retrieved directory ```${RETDIR}``` should contain files like  
      *  ```${RETDIR}/${LANG}_csnet_code_text_retrieval_dedup_${SPLIT}_100.json```
    
    - For REDCODER: use ```prepare_csnet_redcoder.sh``` and for RECODER-ext:  use ```prepare_csnet_redcoder_ext.sh```.



### Step 3: Finetune and Evaluate




#### SCODE-G for text to code:

```bash
cd SCODE-G/text_to_code
```


- CodeXGLUE-CSNET:

```bash
GPU={GPU ID/s}
LANG={python or java}
path_2_data={directory where the pre-processed data located from previous step (i.e., step-2) . E.g., ../redcoder_data/codexglue_csnet_text_to_code_scode-g-preprocessed-input/}
PRETRAIN={Downloaded PLBART checkpoint path e.g., -./checkpoint_11_100000.pt}
SAVE_DIR={output directory. All predictsions will be saved here. The result.txt inside this dir will save the evaluation metrics. E.g., ../redcoder_data/codexglue_csnet_text_to_code_scode-g-output/}
UPDATE_FREQ={num of times before updates / accumualting step size} #optional
BATCH_SIZE={per gpu batch size} #optional
USE_PLBART={true|false} #optional only when to set false

bash run_csnet.sh ${GPU} ${LANG} ${path_2_data} ${PRETRAIN} ${SAVE_DIR}

```
###### Notes:
- According to [fairseq](https://github.com/pytorch/fairseq), effective batch size is equal
to:
```
PER_GPU_TRAIN_BATCH_SIZE * NUM_GPU * UPDATE_FREQ
```
- ${UPDATE\_FREQ}, ${BATCH\_SIZE}, and ${USE\_PLBART} are optional as their default values are 8, 4 and True respectively.
- The train/valid/test splits should be preprocessed in ```${path_2_data}```(as in step-2). For example, test split should be like: ```${path_2_data}/test.source``` and ```${path_2_data}/test.target```.
- ```run_csnet.sh``` file can handle both REDCODER and REDCODER-ext, just the ```${path_2_data}``` should be provided accordingly.








#### SCODE-G for code to text:

```bash
cd SCODE-G/code_to_text
```


- CodeXGLUE-CSNET:

```bash
GPU={GPU ID/s}
SOURCE={python or java}
PATH_2_DATA={directory where the pre-processed data located from previous step (i.e., step-2) . E.g., ../redcoder_data/codexglue_csnet_text_to_code_scode-g-preprocessed-input/}
PRETRAIN={Downloaded PLBART checkpoint path e.g., -./checkpoint_11_100000.pt}
SAVE_DIR={output directory. All predictsions will be saved here. The result.txt inside this dir will save the evaluation metrics. E.g., ../redcoder_data/codexglue_csnet_text_to_code_scode-g-output/}
UPDATE_FREQ={num of times before updates / accumualting step size }
BATCH_SIZE={per gpu batch size} #optional

bash run_csnet.sh ${GPU} ${SOURCE} ${PATH_2_DATA} ${PRETRAIN} ${SAVE_DIR}

```
###### Notes:
- According to [fairseq](https://github.com/pytorch/fairseq), effective batch size is equal
to:
```
PER_GPU_TRAIN_BATCH_SIZE * NUM_GPU * UPDATE_FREQ
```
- ${UPDATE_FREQ} and ${BATCH_SIZE}  are optional as their default values are 8, and 16  respectively.
- The train/valid/test splits should be preprocessed in ```${PATH_2_DATA}``` (as in step-2). For example, test split should be like: ```${PATH_2_DATA}/test.source``` and ```${PATH_2_DATA}/test.target```.
- ```run_csnet.sh``` file can handle both REDCODER and REDCODER-ext, just the ```${PATH_2_DATA}``` should be provided accordingly.




## Reference

If you plan to use `SCODE-G` in your project, please consider citing [our paper](https://arxiv.org/abs/2108.11601):
```
@inproceedings{parvez2021retrieval,
  title = {Retrieval Augmented Code Generation and Summarization},
  author = {Parvez, Md Rizwan and Ahmad, Wasi Uddin and Chakraborty, Saikat and Ray, Baishakhi and Chang, Kai-Wei},
  booktitle = {EMNLP-Findings},
  year = {2021}
}
```
## Misc
We thank the authors of PLABRT, Huggingface, Pytorch, and so on.

## License
Redcoder MIT license. 
