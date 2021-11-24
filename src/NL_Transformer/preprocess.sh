#codes for executing preprocessing

cd examples/translation/ \

bash prepare-iwslt14.sh \

cd ../.. \

TEXT=examples/translation/iwslt14.tokenized.de-en \

python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
