#!/usr/bin/env bash

DATASETS=(
    concode codexglue-csnet-java codexglue-csnet-python
)


DATASET_NAME=${1:-concode};
DATASET_NAME=${1:-codexglue-csnet-python};
#DATASET_NAME=${1:-codexglue-csnet-java};


if [[ ! " ${DATASETS[@]} " =~ " $DATASET_NAME " ]]; then
    echo "Dataset name must be from [$(IFS=\| ; echo "${DATASETS[*]}")].";
    echo "bash pipeline.sh <dataset>";
    exit;
fi

DATA_DIR="/home/wasiahmad/workspace/projects/CodeBART/data/codeXglue/text-to-code/concode";
OUT_DIR="/local/rizwan/workspace/projects/RaProLang/retrieval/bm25/code_to_text";
mkdir -p $OUT_DIR

CODE_BASE_DIR=`realpath ../..`;
export PYTHONPATH=${CODE_BASE_DIR}:$PYTHONPATH;

FILES=()
if [[ $DATASET_NAME == "concode" ]]; then
    FILES+=(${DATA_DIR}/train.json)
    FILES+=(${DATA_DIR}/valid.json)
    FILES+=(${DATA_DIR}/test.json)
    DB_PATH="${OUT_DIR}/${DATASET_NAME}.db";
fi


if [[ $DATASET_NAME == "codexglue-csnet-java" ]]; then
    DATA_DIR="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/java"
    FILES+=(/home/rizwan/DPR/deduplicated.summaries.txt)
    DB_PATH="${OUT_DIR}/${DATASET_NAME}.db";
fi

if [[ $DATASET_NAME == "codexglue-csnet-python" ]]; then
    DATA_DIR="/home/rizwan/CodeBERT/data/codesearch/data/code2nl/CodeSearchNet/python"
    FILES+=(/home/rizwan/DPR/deduplicated.summaries.txt)
    DB_PATH="${OUT_DIR}/${DATASET_NAME}.db";
fi

# Create db from preprocessed data.
if [[ ! -f $DB_PATH ]]; then
    python build_db.py --files "${FILES[@]}" --save_path $DB_PATH --code_to_text;
fi

# Index the preprocessed documents.
python build_es.py --db_path $DB_PATH --domain $DATASET_NAME --config_file_path config.json --port 9200 --code_to_text;

# Search documents based on BM25 scores.
OUTFILE=${OUT_DIR}/${DATASET_NAME}.train_code_text_bm25.json
if [[ ! -f $OUTFILE ]]; then
    python es_search.py \
        --index_name ${DATASET_NAME}_search_test \
        --input_data_file ${DATA_DIR}/train.jsonl \
        --output_fp $OUTFILE \
        --n_docs 10 \
        --port 9200 --code_to_text;
fi
echo "written to: "$OUTFILE

OUTFILE=${OUT_DIR}/${DATASET_NAME}.valid_code_text_bm25.json
if [[ ! -f $OUTFILE ]]; then
    python es_search.py \
        --index_name ${DATASET_NAME}_search_test \
        --input_data_file ${DATA_DIR}/valid.jsonl \
        --output_fp $OUTFILE \
        --n_docs 10 \
        --port 9200 --code_to_text;
fi
echo "written to: "$OUTFILE

OUTFILE=${OUT_DIR}/${DATASET_NAME}.test_code_text_bm25.json
if [[ ! -f $OUTFILE ]]; then
    python es_search.py \
        --index_name ${DATASET_NAME}_search_test \
        --input_data_file ${DATA_DIR}/test.jsonl \
        --output_fp $OUTFILE \
        --n_docs 10 \
        --port 9200 --code_to_text;
fi
echo "written to: "$OUTFILE

python eval_rank.py --input_file $OUTFILE;
