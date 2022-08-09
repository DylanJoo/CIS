# if [[ "$1" = "spr" ]];then
# if [ -s $FILE ];then

echo '-----------|--------|-------|--------|-------|---------|-----|'
echo '| Setting  | Source | R@100 | nDCG@3 | nDCG  | mAP@100 | mAP |'
echo '-----------|--------|-------|--------|-------|---------|-----|'
QREL=data/cast20/2020qrels.txt
BASELINE_A=data/cast20/y2_automatic_results_500.v1.0.run
BASELINE_M=data/cast20/y2_manual_results_500.v1.0.run
echo Automatic Baseline 
tools/trec_eval-9.0.7/trec_eval \
    -m ndcg_cut.3,500 -m map_cut.100,500 -m recall.100 \
    $QREL $BASELINE_A
echo '----------|--------|-------|------|---------|---------|--------|'
echo Manual Baseline 
tools/trec_eval-9.0.7/trec_eval \
    -m ndcg_cut.3,500 -m map_cut.100,500 -m recall.100 \
    $QREL $BASELINE_M
echo '----------|--------|-------|------|---------|---------|--------|'
echo Automatic Baseline monot5
RERANK=runs/cast20.automatic.eval.topics.baseline.top500.monot5.small.trec
tools/trec_eval-9.0.7/trec_eval \
    -m ndcg_cut.3,500 -m map_cut.100,500 -m recall.100 \
    $QREL $RERANK
echo '----------|--------|-------|------|---------|---------|--------|'
# echo Automatic Baseline conv monot5
# RERANK=runs/cast20.automatic.eval.topics.baseline.top500.conv_monomt5.small.trec
# tools/trec_eval-9.0.7/trec_eval \
#     -m ndcg_cut.3,500 -m map_cut.100,500 -m recall.100 \
#     $QREL $RERANK
# echo '----------|--------|-------|------|---------|---------|--------|'
