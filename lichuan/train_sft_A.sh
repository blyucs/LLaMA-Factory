#!/bin/bash

export WANDB_DISABLED="true"
export FORCE_TORCHRUN=1
llamafactory-cli train lichuan/olmo_1b_sft_s1.yaml
