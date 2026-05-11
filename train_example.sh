#!/bin/bash
#Example training script that will train a model on the facebook dataset

echo "training model "

python train_three_feats.py \
  --env zfs \
  --network-type real \
  --network-fname data/facebook.pkl \
  --device cpu \
  --num-node-feats 3 \
  --train-steps 10000 \
  --episode-horizon 100 \
  --batch-size 64 \
  --gamma 0.99 \
  --lr 0.0001 \
  --n-steps 3 \
  --target-update 200 \
  --eps-schedule inverse \
  --eps-start 1.0 \
  --eps-end 0.01 \
  --eps-decay-steps 7500 \
  --seed 0 \
  --replay-capacity 200000\
  --latent-dim 128 \
  --hidden-dim 256 \
  --max-lv 3 \
  --outdir models/example \
  --run-id example_model \
  --log-every 2500 \
  --save-weights \
  --double-dqn

echo "model trained and saved to models/example"
