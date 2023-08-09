PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATASET=law
DATA_PATH=$PROJECT_PATH/data-bin/$DATASET
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/$DATASET

export OMP_WAIT_POLICY=PASSIVE

CUDA_VISIBLE_DEVICES=1 python $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH \
--task translation \
--train-subset valid \
--valid-subset valid \
--best-checkpoint-metric f1 \
--finetune-from-model $BASE_MODEL \
--optimizer adam --adam-betas "(0.9, 0.98)" \
--adam-eps 1e-8 \
--lr 1e-4 --lr-scheduler reduce_lr_on_plateau --min-lr 3e-5 --lr-patience 5 --lr-shrink 0.5 --patience 30 \
--max-update 5000 --max-epoch 100 \
--criterion less_retrieve_criterion \
--save-interval-updates 100 --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
--tensorboard-logdir $PROJECT_PATH/save-models/LRKNNMT/$DATASET/log \
--batch-size 4 \
--update-freq 8 \
--user-dir $PROJECT_PATH/knnbox/models \
--arch less_retrieve_knn_mt@transformer_wmt19_de_en \
--whether_retrieve_selector_path $PROJECT_PATH/save-models/LRKNNMT/$DATASET/selector.pt \
--knn-mode train_less_retrieve \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-k 8 \
--knn-lambda 0.7 \
--knn-temperature 100.0