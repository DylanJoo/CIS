# if [[ "$1" = "spr" ]];then
# if [ -s $FILE ];then

echo '-----------|--------|-------|--------|-------|---------|-----|'
echo '| Setting  | Source | R@100 | nDCG@3 | nDCG  | mAP@100 | mAP |'
# echo '-----------|--------|-------|--------|-------|---------|-----|'
QREL=data/cast20/2020qrels.txt

for RUN in runs/*top1000*trec;do
    echo '----------|--------|-------|------|---------|---------|--------|'
    echo ${RUN##*/}
    echo '----------|--------|-------|------|---------|---------|--------|'
    tools/trec_eval-9.0.7/trec_eval \
        -m ndcg_cut.3,5,500 -m map_cut.500 -m recall.500 \
        $QREL $RUN
done

