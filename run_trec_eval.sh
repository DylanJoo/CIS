# if [[ "$1" = "spr" ]];then
# if [ -s $FILE ];then

echo '-----------|--------|-------|--------|-------|---------|-----|'
echo '| Setting  | Source | R@100 | nDCG@3 | nDCG  | mAP@100 | mAP |'
echo '-----------|--------|-------|--------|-------|---------|-----|'
QREL=data/cast20/2020qrels.txt

for BASELINE in data/cast20/y2*;do
    echo ${baseline##*/}
    tools/trec_eval-9.0.7/trec_eval \
        -m ndcg_cut.3,500 -m map_cut.100,500 -m recall.100 \
        $QREL $BASELINE
done

SPR_A=runs/cast20.automatic.rewrite.spr.top1000.trec
SPR_M=runs/cast22.manual.rewrite.spr.top1000.trec
echo '----------|--------|-------|------|---------|---------|--------|'
for RUN in runs/*.trec;do
    echo ${run##*/}
    tools/trec_eval-9.0.7/trec_eval \
        -m ndcg_cut.3,500 -m map_cut.100,500 -m recall.100 \
        $QREL $RUN
done

