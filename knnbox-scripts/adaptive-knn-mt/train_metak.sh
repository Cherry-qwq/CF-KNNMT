:<<!
[script description]: train adptive-knn-mt's meta-k network
[dataset]: multi domain DE-EN dataset
[base model]: WMT19 DE-ENscript

note 1. You can adjust --batch-size and --update-freq based on your GPU memory.
original paper recommand that batch-size*update-freq equals 32.
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATA_PATH=$PROJECT_PATH/data-bin/medical
SAVE_DIR=$PROJECT_PATH/save-models/combiner/adaptive/medical
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/medical
MAX_K=8

# using paper's settings
CUDA_VISIBLE_DEVICES=6 python $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH \
--task translation \
--train-subset valid --valid-subset valid \
--best-checkpoint-metric "loss" \
--finetune-from-model $BASE_MODEL \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
--lr 3e-4 --lr-scheduler reduce_lr_on_plateau \
--min-lr 3e-05 --label-smoothing 0.001 \
--lr-patience 5 --lr-shrink 0.5 --patience 30 --max-epoch 500 --max-update 5000 \
--criterion label_smoothed_cross_entropy \
--save-interval-updates 100 \
--no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
--tensorboard-logdir $SAVE_DIR/log \
--save-dir $SAVE_DIR \
--batch-size 4 \
--update-freq 8 \
--user-dir $PROJECT_PATH/knnbox/models \
--arch "adaptive_knn_mt@transformer_wmt19_de_en" \
--knn-mode "train_metak" \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-max-k $MAX_K \
--knn-k-type trainable \
--knn-lambda-type trainable \
--knn-temperature-type fixed --knn-temperature 10.0 \
--knn-combiner-path $SAVE_DIR \

