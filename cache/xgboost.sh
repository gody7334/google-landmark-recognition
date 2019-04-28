#/bin/bash

stage="7 5 4 2 1 5-7 4-7 2-7 1-7 4-6 3-6 1-6 4-5 2-5 1-5 2-4 1-4 1,3,5,7 1,5,7 2,6,7"
for s in $stage; do
    echo $s;
    python 000_baseline.py \
        -g 0 \
        -v 007_SUB_XGBOOST \
        -de EXP \
        -m blending_pred \
        -ho 0.15 \
        -cp /home/gody7334/gender-pronoun/input/result/002_CV_ALL/2019-04-09_09-52-02/check_point/ \
        -mp cv*stage2[$s]*.pth \
        -p /home/gody7334/gender-pronoun/input/dataset/test_stage_2.tsv \
        -t 500;
done;
