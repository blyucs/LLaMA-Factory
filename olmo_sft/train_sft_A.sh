#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# 时间戳
TIME=$(date "+%Y_%m_%d_%H_%M_%S")

# 参数化开关
RUN_CONVERT=true
RUN_SFT=true
RUN_EVAL=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-convert)
      RUN_CONVERT=false
      shift
      ;;
    --no-sft)
      RUN_SFT=false
      shift
      ;;
    --no-eval)
      RUN_EVAL=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 定义CONVERT 相关变量
export PRETRAIN_PT_DIR="/root/a100_nas_lvbo/peixunban/002754_lvbo/save_check/olmo1b_2024_12_09_01_44_49/latest-unsharded"
export HF_PT_NAME="olmo1b_hf_pt_$TIME"
export HF_PT_DIR="/root/a100_nas_lvbo/peixunban/002754_lvbo/SFT/hf_pt/${HF_PT_NAME}"
export TOKENIZER_PATH="/root/a100_nas_lvbo/peixunban/002754_lvbo/OLMo/olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json"

# 定义 SFT 相关变量
export SFT_OUT_DIR="/root/a100_nas_lvbo/peixunban/002754_lvbo/SFT/sft_pt/olmo1b_sft_pt_$TIME"
export MODEL_NAME_DIR="${HF_PT_DIR}"
export SFT_ORIN_YAML="/root/a100_nas_lvbo/peixunban/002754_lvbo/SFT/llama-factory-lichuan/olmo_sft/olmo_1b_sft_s1.yaml"
export SFT_YQ_YAML="/root/a100_nas_lvbo/peixunban/002754_lvbo/SFT/llama-factory-lichuan/olmo_sft/olmo_1b_sft_s1_$TIME.yaml"
export SFT_DATASETS="tulu3-A,tulu3-B" # "physics-rewrite, tulu3-A,tulu3-B,physics,physics-arxiv,tulu-3-math,tulu-3-math-grade,biology"  # tulu3-A,tulu3-B, physics,physics-arxiv,tulu-3-math,tulu-3-math-grade
export LLAMA_FACTORY_DIR="/root/a100_nas_lvbo/peixunban/002754_lvbo/SFT/llama-factory-lichuan"

# 定义opencompass 相关变量
export MODEL_PATH=${SFT_OUT_DIR}
export OPENCOMPASS_DIR="/root/a100_nas_lvbo/peixunban/002754_lvbo/EVAL/opencompass"


if [ "$RUN_CONVERT" = true ]; then
  echo "========= CONVERT HF START =========="
  # 定义任务名称
  TASK_NAME="OLMo Conversion Task"
  echo "Task name: ${TASK_NAME}"
  conda deactivate
  conda activate olmo-lvbo
  # 转换为 Hugging Face 格式的预训练模型
  python /root/a100_nas_lvbo/peixunban/002754_lvbo/OLMo/scripts/convert_olmo_to_hf_new.py --input_dir "${PRETRAIN_PT_DIR}" \
      --tokenizer_json_path "${TOKENIZER_PATH}" \
      --output_dir "${HF_PT_DIR}"
  echo "===========CONVERT HF END============"
fi

if [ "$RUN_SFT" = true ]; then
  echo "===========SFT START============"
  wait
    # 切换到目标目录
  cd "${LLAMA_FACTORY_DIR}" || { echo "Failed to change directory to ${LLAMA_FACTORY_DIR}"; exit 1; }
  # 运行 SFT 相关任务
  echo "Running SFT tasks in ${LLAMA_FACTORY_DIR}..."
  conda deactivate
  conda activate llama-factory
  # 定义 SFT 输出目录
  # 复制原始 YAML 文件
  cp ${SFT_ORIN_YAML} ${SFT_YQ_YAML}

  # 修改 YAML 配置
  yq -i -y ".output_dir = \"${SFT_OUT_DIR}\"" ${SFT_YQ_YAML}
  yq -i -y ".model_name_or_path = \"${MODEL_NAME_DIR}\"" ${SFT_YQ_YAML}
  yq -i -y ".dataset = \"${SFT_DATASETS}\"" ${SFT_YQ_YAML}
  yq -i -y ".run_name = \"sft_olmo1b_${TIME}\"" ${SFT_YQ_YAML}

  # export WANDB_DISABLED="true"
  export FORCE_TORCHRUN=1
  llamafactory-cli train ${SFT_YQ_YAML}
  echo "============= SFT END============="
fi

if [ "$RUN_EVAL" = true ]; then
  # opencompass
  echo "=========== EVAL MMLU START============"
  wait
  cd ${OPENCOMPASS_DIR}
      # 切换到目标目录
  cd "${OPENCOMPASS_DIR}" || { echo "Failed to change directory to ${OPENCOMPASS_DIR}"; exit 1; }
  # 运行 EVAL 相关任务
  echo "Running EVAL tasks in ${OPENCOMPASS_DIR}..."
  conda deactivate
  conda activate opencompass-new
  python /root/a100_nas_lvbo/peixunban/002754_lvbo/EVAL/opencompass/run.py --models OLMo_1B_Chat \
      --datasets mmlu_gen_4d595a \
      --max-num-workers 16 \
      --hf-num-gpus 2  \
      --debug \
      -a vllm
  echo "====== EVAL MMLU END======="
fi
