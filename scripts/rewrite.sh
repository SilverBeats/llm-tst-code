#!/bin/bash

dataset=$1
llm_type=$2

# llama2-7b-chat-hf, llama2_13b_chat_hf, qwen_7b_chat, qwen_14b_chat
# gpt_3_5_turbo, gpt_4, mistral_7b_instruct, falcon_7b_instruct

api_models=("qwen-7b-chat" "qwen-14b-chat" "gpt-3.5-turbo" "gpt-4")
local_models=("llama2-7b-chat-hf" "llama2-13b-chat-hf" "mistral-7b-instruct" "falcon-7b-instruct")

is_api_model=-1
for t in "${api_models[@]}";do
  if [ "$t" == "$llm_type" ];then
    is_api_model=1
    break
  fi
done

for t in "${local_models[@]}";do
  if [ "$t" == "$llm_type" ];then
    is_api_model=0
    break
  fi
done

if [ "$is_api_model" == -1 ];then
  echo "unknown llm model: $llm_type"
  exit 8
fi

python main.py --dataset "$dataset" --llm_type "$llm_type" \
              --llm_model_dir "$3" --do_rewrite --template_type "common"