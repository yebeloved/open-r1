#!/bin/bash
NUM_GPUS=3
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --dtype float16 \
    --output-dir $OUTPUT_DIR 