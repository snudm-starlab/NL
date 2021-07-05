# Negotiation_Learning
This project is a PyTorch implementation of 'Negotiation Learning: Effectively Compressing Multiple Transformer Layers Through a Negotiator'. This paper proposes a novel distillation approach that transfers knowledge embedded in the teacher model's parameters to the student model.

## Overview
#### Brief Explanation of the paper and the code. 
The main idea of Negotiation Learning is to train multiple variants of the original teacher with the layers reordered. The code can run simple finetuning, simple KD, Patient KD, Dual Learning (DL), and Negotiation Learning (NL).

#### Baseline Codes
1. NL_BERT: This repository is based on the [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355). All source files are from the repository if not mentioned otherwise. The main scripts that actually run tasks are the following files, and they have been modified from the original files in the original repository:
- 'finetune.py', 'finetune_distilbert.py', 'finetune_tinybert', and 'finetune_NL.py' - based on 'NLI_KD_training.py' in the original repository.
- 'save_teacher_outputs.py - based on 'run_glue_benchmark.py' in the original repository.

2. NL_Transformer_Enc_Dec: This repository is based on the [GitHub repository](https://github.com/pytorch/fairseq) from Facebook. All source files are from the repository if not mentioned otherwise. The main scripts that actually run tasks are the following files, and they have been modified from the original files in the original repository:
- 'train.py', 'train_DL.py', and 'train_NL.py' - based on 'train.py' in the original repository.

The overall structure of the code package is as follows:
``` Unicode
Negotiation Learning
  │
  ├──  src        
  │     ├── NL_BERT
  │     │         ├── BERT
  │     │         │    └── pytorch_pretrained_bert: BERT sturcture files
  │     │         ├── data
  │     │         │    ├── data_raw
  │     │         │    │     ├── glue_data: task dataset
  │     │         │    │     └── download_glue_data.py
  │     │         │    ├── models
  │     │         │    │     └── bert_base_uncased: ckpt
  │     │         │    └── outputs
  │     │         │           └── save teacher model prediction & trained student model.
  │     │         ├── utils : The overall utils. 
  │     │         ├── envs.py: save directory paths for several usage.
  │     │         ├── save_teacher_outputs.py : save teacher prediction. Used for PTP, KD, PKD e.t.c. 
  │     │         ├── PTP.py : pretrain the student model with PTP. 
  │     │         └── finetune.py: comprehensive training file for teacher and student models.
  │     │
  │     │ 
  │     └── NL_Transformer_Enc_Dec
  │                   ├── fairseq
  │                   ├── fairseq_cli
  │                   ├── train.py
  │                   ├── train_DL.py
  │                   └── train_NL.py
  │
  ├── preprocess.sh: downloads GLUE datasets.
  ├── Makefile: Makefile used for demo.
  ├── Developers_Guide.docx
  ├── requirements.txt: run this file to download required environments.
  ├── LICENSE
  └── README.md
```

#### Data description
- GLUE datasets (for BERT)
- IWSLT'14 Translation datasets (for Transformer Enc-Dec)

* Note that: 
    * GLUE datasets consists of CoLA, diagnostic, MNLI, MRPC, QNLI, QQP, RTE, SNLI, SST-2, STS-B, WNLI
    * You can download GLUE datasets by running bash 'preprocess.sh'.
    * You can download IWSLT'14 following the script explained below.

## Install 

#### Environment 
* Ubuntu
* CUDA >= 10.0
* Pytorch >= 1.4 
* numpy
* torch
* Tensorly
* tqdm
* pandas
* apex

## Dependence Install
1. For BERT
```
git clone https://github.com/Anonym96/Negotiation_Learning
cd NL_BERT
pip install -r requirements.txt
```
2. For Transformer Encoder Decoder
```
git clone https://github.com/Anonym96/Negotiation_Learning
cd NL_Transformer_Enc_Dec/fairseq
pip install --editable ./
```

# Getting Started

## 1. BERT 

### Preprocess
Download GLUE datasets by running script:
```
bash preprocess.sh
```
You must download your own pretrained BERT model at 'src/NL_BERT/data/models/pretrained/bert-base-uncased'. 
Refer to 'src/NL_BERT/BERT/pytorch_pretrained_bert/modeling.py' line 43~51.

### Demo 
you can run the demo version.
```
make
```

### Run your own training  
* We provide an example how to run the codes. We use task: 'MRPC', teacher layer: 12, and student layer: 6 as an example.
* Before starting, we need to specify a few things.
    * task: one of the GLUE datasets
    * train_type: one of the followings - ft, kd, pkd 
    * model_type: one of the followings - Original, NL
    * student_hidden_layers: the number of student layers
    * train_seed: the train seed to use. If default -> random 
    * saving_criterion_acc: if the model's val accuracy is above this value, we save the model.
    * saving_criterion_loss: if the model's val loss is below this value, we save the model.
    * load_model_dir: specify a directory of the checkpoint if you want to load one.
    * output_dir: specify a directory where the outputs will be written and saved.
    
* First, We begin with finetuning the teacher model
    ```
    run script
    python src/NL_BERT/finetune.py \
    --task 'MRPC' \
    --train_batch_size 32 \
    --train_type 'ft' \
    --model_type 'Original' \
    --student_hidden_layers 12 \
    --saving_criterion_acc 0.0 \
    --saving_criterion_loss 1.0 \
    --layer_initialization '1,2,3,4,5,6' \
    --output_dir 'teacher_12layer'
    ```
   
    The trained model will be saved in 'src/NL_BERT/data/outputs/KD/{task}/teacher_12layer/'

* To use the teacher model's predictions for PKD, run script:
    ```
    python src/NL_BERT/save_teacher_outputs.py
    ```
    The teacher predictions will be saved in 'src/NL_BERT/data/outputs/KD/{task}/{task}_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl'
    (it depends on the code line x in 'src/NL_BERT/save_teacher_outputs.py'.

* To run Patient KD applied on BERT student, run script:
    ```
    run script:
    python src/NL_BERT/finetune.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'Original' \
    --student_hidden_layer 6 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.0 \
    --teacher_prediction '/home/ikhyuncho23/data/outputs/KD/MRPC/MRPC_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl' \
    --layer_initialization '1,2,3,4,5,6' \
    --output_dir 'pkd_run-1'
    ```
    
* To run Naive PD applied on BERT student, run script:
    ```
    run script:
    python src/NL_BERT/finetune.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'Original' \
    --student_hidden_layer 6 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.0 \
    --teacher_prediction '/home/ikhyuncho23/data/outputs/KD/MRPC/MRPC_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl' \
    --load_model_dir 'teacher_12layer/BERT.encoder_loss.pkl' \
    --layer_initialization '2,4,6,8,10,12' \
    --output_dir 'pkd_run-1'
    ```
    
* To run DL applied on BERT student, run following scripts in order:
    ```
    run script:
    python src/NL_BERT/finetune_NL.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'NL' \
    --NL_mode 2 \
    --student_hidden_layer 6 \
    --saving_criterion_acc 0.0 \
    --saving_criterion_loss 1.0 \
    --teacher_prediction '/home/ikhyuncho23/data/outputs/KD/MRPC/MRPC_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl' \
    --load_model_dir 'teacher_12layer/BERT.encoder_loss.pkl' \
    --layer_initialization '1,2,3,4,5,6,7,8,9,10,11,12,2,4,6,8,10,12' \
    --output_dir 'DL_run_1'
    ```
    
    Note that NL with 'NL_mode = 2' is equivalent to NL without the Negotiator which is equal to DL.
    After running the above script, now we initialize the student with the DL layers and finetune once more.
    ```
    run script:
    python src/NL_BERT/finetune.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'Original' \
    --student_hidden_layer 6 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.0 \
    --teacher_prediction '/home/ikhyuncho23/data/outputs/KD/MRPC/MRPC_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl' \
    --load_model_dir 'DL_run_1/BERT.encoder_loss_a;;.pkl' \
    --layer_initialization '2,4,6,8,10,12' \
    --output_dir 'DL_result_1'
    ```
    
* To run NL applied on BERT student, run following scripts in order (the only difference with the above DL is the 'NL_mode = 0' :    
    ```
    run script:
    python src/NL_BERT/finetune_NL.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'NL' \
    --NL_mode 0 \
    --student_hidden_layer 6 \
    --saving_criterion_acc 0.0 \
    --saving_criterion_loss 1.0 \
    --teacher_prediction '/home/ikhyuncho23/data/outputs/KD/MRPC/MRPC_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl' \
    --load_model_dir 'teacher_12layer/BERT.encoder_loss.pkl' \
    --layer_initialization '1,2,3,4,5,6,7,8,9,10,11,12,2,4,6,8,10,12' \
    --output_dir 'NL_run_1'
    ```
    then     
    ```
    run script:
    python src/NL_BERT/finetune.py \
    --task 'MRPC' \
    --train_type 'pkd' \
    --model_type 'Original' \
    --student_hidden_layer 6 \
    --saving_criterion_acc 1.0 \
    --saving_criterion_loss 0.0 \
    --teacher_prediction '/home/ikhyuncho23/data/outputs/KD/MRPC/MRPC_Originalbert_base_patient_kd_teacher_12layer_result_summary.pkl' \
    --load_model_dir 'NL_run_1/BERT.encoder_loss_a;;.pkl' \
    --layer_initialization '2,4,6,8,10,12' \
    --output_dir 'NL_result_1'
    ```

# 2. Transformer Enc-Dec 

We require a few additional Python dependencies for preprocessing:
```
pip install fastBPE sacremoses subword_nmt
```
Download and preprocess the data. The below example is for Iwslt'14 De->En translation.
```
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
Now we can train a Transformer Encoder Decoder model over this data. 
```
!python ~/NL_Transformer_Enc_Dec/train.py \
    ~/NL_Transformer_Enc_Dec/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_small_1 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_student' \
    --save-model-dir 'checkpoint_student' \
    --keep-best-checkpoints 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
* To apply Naive PD run the following scripts in order.
First we need to train the teacher model
```
!python ~/NL_Transformer_Enc_Dec/train.py \
    ~/NL_Transformer_Enc_Dec/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_small_2 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_teacher' \
    --save-model-dir 'checkpoint_teacher' \
    --keep-best-checkpoints 5 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
then
```
!python ~/train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_small_1 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_teacher' \
    --save-model-dir 'Checkpoint_Naive_PD_results' \
    --load-model-type naive \
    --restore-file checkpoint.best.pt \
    --layer-initialization 2,4,6,8 \
    --keep-best-checkpoints 5 \
    --validate-interval 2 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

* To apply DL run the following scripts in order.
```
!python ~/NL_Transformer_Enc_Dec/train_DL.py \
    ~/NL_Transformer_Enc_Dec/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_DL --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_DL --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_DL' \
    --save-model-dir 'checkpoint_DL' \
    --keep-best-checkpoints 5
```
then
```
!python ~/train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_small_1 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_DL' \
    --save-model-dir 'Checkpoint_DL_result' \
    --load-model-type naive \
    --restore-file checkpoint.best.pt \
    --layer-initialization 2,4,6,8 \
    --keep-best-checkpoints 5 \
    --validate-interval 2 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
* To apply NL run the following scripts in order.
```
!python ~/NL_Transformer_Enc_Dec/train_NL.py \
    ~/NL_Transformer_Enc_Dec/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_DL --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_DL --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_NL' \
    --save-model-dir 'checkpoint_NL' \
    --keep-best-checkpoints 5
```
then
```
!python ~/train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_small_1 --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_NL' \
    --save-model-dir 'Checkpoint_NL_result' \
    --load-model-type naive \
    --restore-file checkpoint.best.pt \
    --layer-initialization 2,4,6,8 \
    --keep-best-checkpoints 5 \
    --validate-interval 2 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

## Contact

- Ikhyun Cho (ikhyuncho@snu.ac.kr)
- U Kang (ukang@snu.ac.kr)
- Data Mining Lab. at Seoul National University.

*This software may be used only for research evaluation purposes.*  
*For other purposes (e.g., commercial), please contact the authors.*

