#!/usr/bin/env bash

conda init
conda activate prophet
echo "activated prophet"

cd /home/mira/PycharmProjects/prophet_plankton_model
wandb sweep --project prophet_sweep cfg/sweep/prophet_sweep_local.yaml 2> syn_temp.file
cat syn_temp.file
eval "$(awk 'NR==4 {print $6, $7, $8}' syn_temp.file)"
echo "ran wandb sweep"
