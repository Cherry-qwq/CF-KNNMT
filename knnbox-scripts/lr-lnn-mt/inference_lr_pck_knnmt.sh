PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/it 
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/it

export OMP_WAIT_POLICY=PASSIVE

CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/knnbox-scripts/common/generate.py $DATA_PATH \
--task translation \
--path $BASE_MODEL \
--dataset-impl mmap \
--beam 4 \
--lenpen 0.6 \
--max-len-a 1.2 \
--max-len-b 10 \
--source-lang de \
--target-lang en \
--gen-subset test \
--max-tokens 2048 \
--scoring sacrebleu \
--tokenizer moses \
--remove-bpe \
--user-dir $PROJECT_PATH/knnbox/models \
--arch lr_pck_knn_mt@transformer_wmt19_de_en \
--whether_retrieve_selector_path $PROJECT_PATH/save-models/LRKNNMT/it/selector.pt \
--knn-mode inference \
--knn-datastore-path $PROJECT_PATH/datastore/pck/it_dim64 \
--knn-combiner-path $PROJECT_PATH/save-models/combiner/pck/it_dim64 \
--knn-max-k 4 \
--knn-temperature 10.0