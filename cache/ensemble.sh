#/bin/bash

stage="7 6 5 4 3 2 1 6-7 5-7 4-7 3-7 2-7 1-7 5-6 4-6 3-6 2-6 1-6 4-5 3-5 2-5 1-5 3-4 2-4 1-4 1,3,5,7 2,4,6,7 1,5,7 2,6,7"
for s in $stage; do
    echo $s;
    python 000_baseline.py \
        -g 1 \
        -v 006_SUB_ENSEMBLE \
        -de EXP \
        -m ensemble_eval \
        -ho 0.15 \
        -cp /home/gody7334/gender-pronoun/input/result/002_CV_ALL/2019-04-09_09-52-02/check_point/ \
        -mp cv*stage2[$s]*.pth;
    python 000_baseline.py \
        -g 1 \
        -v 006_SUB_ENSEMBLE \
        -de EXP \
        -m ensemble_pred \
        -ho 0.15 \
        -cp /home/gody7334/gender-pronoun/input/result/002_CV_ALL/2019-04-09_09-52-02/check_point/ \
        -mp cv*stage2[$s]*.pth \
        -p /home/gody7334/gender-pronoun/input/dataset/test_stage_2.tsv;
done;
