#!/usr/bin/env bash
# run: bash prepare.sh java 1 false false true &

LANG=${1:-"python"}
top_k=${2:-1}
NO_CONCODE_VARS=${3:-false}
MASK_RATE=${3:-0}
WITH_WITHOUT_SUFFIX=${4:-no} # with or no

SPMDIR=/local/wasiahmad/codebart
RETDIR=/local/rizwan/workspace/projects/RaProLang/retrieval/bm25/code_to_text/



SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/${LANG}_BM25_RET_CODE_1_NO_NL_"${WITH_WITHOUT_SUFFIX}
mkdir -p $SAVE_DIR

export PYTHONIOENCODING=utf-8;
# <path to data> which contains binarized data for each directions
CURRENT_DIR=`pwd`
CODE_DIR_HOME=`realpath ..`;

function prepare () {

if [[ "$NO_CONCODE_VARS" = true ]]; then
    EXTRA=" --NO_CONCODE_VARS "
else
    EXTRA=""
fi



for split in test; do
    python dpr_process.py \
        --retrieved_code_file ${RETDIR}/codexglue-csnet-${LANG}.${split}_code_text_bm25.json \
        --split $split \
        --top_k $top_k \
        --out_dir $SAVE_DIR \
        --mask_rate $MASK_RATE --WITH_WITHOUT_SUFFIX $WITH_WITHOUT_SUFFIX --lang $LANG \
        $EXTRA;
done


python evaluator.py ${SAVE_DIR}/test.target ${SAVE_DIR}/test.source
files2rouge  ${SAVE_DIR}/test.target ${SAVE_DIR}/test.source
}

for lang in $LANG; do
    prepare
done


