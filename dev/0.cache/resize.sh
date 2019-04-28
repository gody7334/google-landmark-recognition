#/bin/bash
wd=256;
for i in `seq -f "%03g" $1 $2`; do
    echo "process image $i";
    echo "copy tar original image from HDD to SSD"
    cp ~/HDDNFS2/google-landmark/images_$i.tar ./
    mkdir images_$i;
    mkdir images_$i-$wd;

    echo "untar";
    tar -xf images_$i.tar -C images_$i;

    echo "move";
    find images_$i -type f -print0 | xargs -0 mv -t images_$i;

    echo "resize";
    python resize.py -wi $wd -hi $wd -s images_$i -d images_$i-$wd

    echo "tar";
    # this tar include root folder
    tar -cf images_$i-$wd.tar images_$i-$wd;

    echo "remove"
    rm -rf images_$i;
    rm -rf images_$i-$wd;

    echo "copy tar resize image back to HDD"
    cp images_$i-$wd.tar ~/HDDNFS2/google-landmark/

    echo "remove tar"
    rm images_$i.tar
    rm images_$i-$wd.tar
done;
