#/bin/bash

mkdir trn-raw;
for i in {000..001}; do
    echo "process image $i";
    mkdir images_$i;
    tar -xf images_$i.tar -C images_$i;
    find images_$i -type f -print0 | xargs -0 mv -t trn-raw;
done
