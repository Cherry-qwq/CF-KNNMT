:<<! 
[script description]: build a datastore for vanilla-knn-mt
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 EN-ZH
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt20.en-zh/en_zh_2.pt
DATA_PATH=$PROJECT_PATH/data-bin/en-zh
DATASTORE_SAVE_PATH=$PROJECT_PATH/datastore/vanilla/en-zh


CUDA_VISIBLE_DEVICES=5 python $PROJECT_PATH/knnbox-scripts/common/validate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--model-overrides "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}" \
--dataset-impl raw \
--valid-subset train \
--skip-invalid-size-inputs-valid-test \
--max-tokens 4096 \
--bpe fastbpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch vanilla_knn_mt@transformer_en_zh \
--knn-mode build_datastore \
--knn-datastore-path $DATASTORE_SAVE_PATH \
