type_of_text_pair=student
gs=gs://cnclab/cast20.canard.construction/${type_of_text_pair}_text_pair_pred/canard.train.${type_of_text_pair}.top1000.pred

for file in tlogits flogits;do
    echo Downloading prediction ${file} ... in ${gs}
    for split in {00..63};do
        gsutil cp ${gs}${split}.${file} ./
    done

    echo Merge all the ${file} into one file
    cat canard.train.${type_of_text_pair}.top1000.pred??.${file} > ${type_of_text_pair}.${file}
    rm canard.train.${type_of_text_pair}.top1000.pred??.${file}
    echo Done
done
