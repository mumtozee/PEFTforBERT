# Intro

This repo showcases how PEFT can be applied to BERT. MLM was chosen as a sample task, however minimal code editing is needed to run this code for other tasks.

An example command to run PEFT for BERT:

<code>$ CUDA_VISIBLE_DEVICES=X,Y accelerate launch peft_train.py --config-path config.yaml</code>

# Notes
- passing a configuration is done using YAML, however parametrisation via command line arguments is also supported
- I've tried to support all of the available PEFT methods from HuggingFace up to `10th of December 2023`