#!/bin/bash

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

PRETRAIN_CHKPT="/vol/bitbucket/cp524/dev/papers_with_code/mage/mage-vitl-1600.pth"
OUTPUT_DIR="output_large"

python gen_img_uncond.py --temp 6.0 --num_iter 20 \
--ckpt ${PRETRAIN_CHKPT} --batch_size 32 --num_images 32 \
--model mage_vit_large_patch16 --output_dir ${OUTPUT_DIR}
