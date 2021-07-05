python ../train_NL.py \
    ~/Negotiation_Learning/src/NL_Transformer/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en_NL --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_NL --label-smoothing 0.1 \
    --max-tokens 4096 \
    --load-model-dir 'checkpoint_NL' \
    --save-model-dir 'checkpoint_NL' \
    --keep-best-checkpoints 5 \

python ../train.py \
    ~/Negotiation_Learning/src/NL_Transformer/data-bin/iwslt14.tokenized.de-en \
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
    