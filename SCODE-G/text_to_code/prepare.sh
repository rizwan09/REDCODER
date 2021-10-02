#!/usr/bin/env bash
# run: bash prepare.sh java 1 false false true &

LANG=${1:-java}
top_k=${2:-10}
NO_CONCODE_VARS=${3:-false}
MASK_RATE=${4:-0.00}
WITH_OR_WITHOUT_REF=${5:-no} #with or no

SPMDIR=/local/wasiahmad/codebart
RETDIR=/local/rizwan/DPR_models/biencoder_models_concode_without_code_tokens/java/
SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/concode/${WITH_OR_WITHOUT_REF}_${top_k}"
mkdir -p $SAVE_DIR

function prepare () {


if [[ "$NO_CONCODE_VARS" = true ]]; then
    EXTRA=" --NO_CONCODE_VARS "
else
    EXTRA=""
fi



for split in train valid test; do
    python process.py \
        --retrieved_code_file $RETDIR/${split}_20.json \
        --split $split \
        --top_k $top_k \
        --out_dir $SAVE_DIR \
        --mask_rate $MASK_RATE --WITH_OR_WITHOUT_REF $WITH_OR_WITHOUT_REF \
        $EXTRA;
done

}

function spm_preprocess () {

LANG=$1
for SPLIT in train valid test; do
    if [[ ! -f $SAVE_DIR/${SPLIT}.spm.$LANG ]]; then
        python encode.py \
            --model_file $SPMDIR/sentencepiece.bpe.model \
            --input_source $SAVE_DIR/$SPLIT.source \
            --input_target $SAVE_DIR/$SPLIT.target \
            --output_source $SAVE_DIR/${SPLIT}.spm.en_XX \
            --output_target $SAVE_DIR/${SPLIT}.spm.$LANG \
            --max_len 510 \
            --workers 60;
    fi
done

}

function binarize () {

LANG=$1
if [[ ! -d $SAVE_DIR/data-bin ]]; then
    fairseq-preprocess \
        --source-lang en_XX \
        --target-lang $LANG \
        --trainpref $SAVE_DIR/train.spm \
        --validpref $SAVE_DIR/valid.spm \
        --testpref $SAVE_DIR/test.spm \
        --destdir $SAVE_DIR/data-bin \
        --thresholdtgt 0 \
        --thresholdsrc 0 \
        --workers 60 \
        --srcdict ${SPMDIR}/dict.txt \
        --tgtdict ${SPMDIR}/dict.txt;
fi

}

for lang in $LANG; do
    prepare
    spm_preprocess $lang
    binarize $lang
done

