#!/bin/bash

dataset=$1
llm_type=$2
template_type=$3
template_idx=$4


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

if [ "$is_api_model" -eq -1 ];then
  echo "unknown llm model: $llm_type"
  exit 8
fi

if [ "$dataset" == "yelp" ]; then
    python main.py \
              --dataset "$dataset" \
              --llm_type "$llm_type" \
              --acc_model_dir_or_path "output/classifier/pos-neg/yelp/classifier.pk" \
              --d_ppl_model_dir "output/fluency/pos-neg/yelp/4" \
              --template_type "$template_type" \
              --template_idx "$template_idx" \
              --do_eval
elif [ "$dataset" == "amazon" ]; then
    python main.py \
              --dataset "$dataset" \
              --llm_type "$llm_type" \
              --acc_model_dir_or_path "output/classifier/pos-neg/amazon/classifier.pk" \
              --d_ppl_model_dir "output/fluency/pos-neg/amazon/4" \
              --template_type "$template_type" \
              --template_idx "$template_idx" \
              --do_eval
elif [ "$dataset" == "imagecaption" ]; then
    python main.py \
              --dataset "$dataset" \
              --llm_type "$llm_type" \
              --acc_model_dir_or_path "output/classifier/romantic-humorous/imagecaption/classifier.pk" \
              --d_ppl_model_dir "output/fluency/romantic-humorous/imagecaption/4/" \
              --template_type "$template_type" \
              --template_idx "$template_idx" \
              --do_eval
elif [ "$dataset" == "gender" ]; then
    python main.py \
              --dataset "$dataset" \
              --llm_type "$llm_type" \
              --acc_model_dir_or_path "output/classifier/male-female/gender/classifier.pk" \
              --d_ppl_model_dir "output/fluency/male-female/gender/4" \
              --template_type "$template_type" \
              --template_idx "$template_idx" \
              --do_eval
elif [ "$dataset" == "political" ]; then
    python main.py \
              --dataset "$dataset" \
              --llm_type "$llm_type" \
              --acc_model_dir_or_path "output/classifier/republican-democratic/political/classifier.pk" \
              --d_ppl_model_dir "output/fluency/republican-democratic/political/4" \
              --template_type "$template_type" \
              --template_idx "$template_idx" \
              --do_eval
else
  echo '未实现'
fi