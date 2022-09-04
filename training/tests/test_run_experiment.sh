#!/bin/bash
set -uo pipefail
set +e

FAILURE=false

echo "running fast_dev_run test of real model class on real data"
python training/train.py --data_class=PICa --model_class=ViT2GPT2 \
  --batch_size 2 --lr 0.0001 \
  --fast_dev_run --num_sanity_val_steps 0 \
  --num_workers 1 || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Test for train.py failed"
  exit 1
fi
echo "Tests for train.py passed"
exit 0
