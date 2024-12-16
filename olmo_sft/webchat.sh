#!/bin/bash

export GRADIO_SHARE="true"
export WANDB_DISABLED="true"
export API_PORT="8889"

llamafactory-cli webchat lichuan/olmo_1b_sft_predict.yaml
