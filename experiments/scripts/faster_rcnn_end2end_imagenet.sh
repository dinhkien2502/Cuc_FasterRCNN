#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
ITERS=100000
DATASET_TRAIN=imagenet_train
DATASET_TEST=imagenet_train

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/faster_rcnn_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"



time ./tools/train_net_imagenet.py --gpu ${GPU_ID} \
  --solver models/VGG16/faster_rcnn_end2end/solver.prototxt \
  
  --imdb ${DATASET_TRAIN} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
	--weights data/VGG16.v2.caffemodel \
  --set EXP_DIR seed_rng1701 RNG_SEED 1701
	
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net_imagenet.py --gpu ${GPU_ID} \
  --def models/${NET}/faster_rcnn_end2end/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${DATASET_TEST} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
