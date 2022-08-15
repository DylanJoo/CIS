# automatic (user context, 3q)
python3 tools/convert_run_to_monot5.py \
  -run runs/cast20.automatic.rewrite.spr.top1000.trec \
  -corpus data/cast20/collections/ \
  -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
  --output_text_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.3q.top1000.text_pairs.txt \
  --output_id_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.3q.top1000.id_pairs.txt \
  --use_context 3 

# gsutil cp data/cast20/conv-monot5-pairs/cast20.manual.rewrite.3q.top1000.text_pairs.txt \
#     gs://cnclab/cast20.convir/conv-monot5-pairs/

# # automatic (user context, allq)
# python3 tools/convert_run_to_monot5.py \
#   -run runs/cast20.automatic.rewrite.spr.top1000.trec \
#   -corpus data/cast20/collections/ \
#   -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --output_text_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.allq.top1000.text_pairs.txt \
#   --output_id_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.allq.top1000.id_pairs.txt \
#   --use_context 0 &
#
# automatic (user context, wndw=3)
# python3 tools/convert_run_to_monot5.py \
#   -run runs/cast20.automatic.rewrite.spr.top1000.trec \
#   -corpus data/cast20/collections/ \
#   -topic data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --output_text_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.wndw.3.top1000.text_pairs.txt \
#   --output_id_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.wndw.3.top1000.id_pairs.txt \
#   --use_context 3  \
#   --use_response  &
