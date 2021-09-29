#!/usr/bin/env bash
# run: bash prepare.sh java 1 false false true &

LANG=${1:-java}
top_k=${2:-1}
NO_CONCODE_VARS=${3:-false}
MASK_RATE=${4:-0}
WITH_OR_WITHOUT_REF=${5:-with} #with or no


SPMDIR=/local/wasiahmad/codebart
RETDIR=/local/rizwan/DPR_models/csnet
#RETDIR=/home/rizwan/DPR_models/biencoder_graphcode_models_csnet_text_code
SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/${LANG}_retrievd_from_${WITH_OR_WITHOUT_REF}_ref_top_${top_k}_mask_rate_${MASK_RATE}"
#SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/${LANG}_retrievd_from_${WITH_OR_WITHOUT_REF}_ref_top_${top_k}"
#SAVE_DIR="/local/rizwan/workspace/projects/RaProLanG/data/plbart/csnet/${LANG}_retrievd_from_${WITH_OR_WITHOUT_REF}_ref_top_${top_k}_filtered"
mkdir -p $SAVE_DIR


function prepare () {


if [[ "$NO_CONCODE_VARS" = true ]]; then
    EXTRA=" --NO_CONCODE_VARS "
else
    EXTRA=""
fi



for split in train valid test; do
    python process.py \
        --retrieved_code_file $RETDIR/${LANG}_csnet_pos_only_retrieval_dedup_${split}_30.json \
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
