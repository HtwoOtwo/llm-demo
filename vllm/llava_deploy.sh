#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve "path to local path of llava-1.5-7b-hf" \
    --chat-template template_llava.jinja \
    --tensor_parallel_size=2