#!/bin/bash
for i in {1..19}
do
echo $i
python -m lib.main -loss=categorical_crossentropy --filter_size 150 -f=0 -e=18 --clipping 19 --window_sizes $i  -t=trainfile_clean.txt --test_file=test_clean.txt --posembeddingdim 50 --markup -dropoutrate 0.5
perl data/semeval/scorer/semeval2010_task8_scorer-v1.2.pl data/semeval/test_pred.txt data/semeval/test_key.txt > window$i.txt
done
