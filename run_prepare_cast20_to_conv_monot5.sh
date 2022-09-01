for method in cqe t5 t5-cqe;do
    # automatic (utterance + user context, allq)
    python3 tools/convert_run_to_monot5.py \
      --run runs/cast_result/result.${method}.cast2020.eval.trec \
      --corpus data/cast20/collections/ \
      --topic_queries data/cast20/2020_evaluation_topics_v1.0.jsonl \
      --use_query_key utterance \
      --output_text_pair data/cast20/conv-monot5-pairs/cast20.sclin.${method}.top1000.conv.monot5.text_pairs.txt \
      --output_id_pair data/cast20/conv-monot5-pairs/cast20.sclin.${method}.top1000.conv.monot5.id_pairs.txt \
      --use_context 0 &

    # automatic (standard)
    python3 tools/convert_run_to_monot5.py \
      --run runs/cast_result/result.${method}.cast2020.eval.trec \
      --corpus data/cast20/collections/ \
      --topic_queries data/cast20/2020_evaluation_topics_v1.0.jsonl \
      --use_query_key automatic_rewritten \
      --output_text_pair data/cast20/monot5-pairs/cast20.sclin.${method}.top1000.monot5.text_pairs.txt \
      --output_id_pair data/cast20/monot5-pairs/cast20.sclin.${method}.top1000.monot5.id_pairs.txt
done

## archived
# # automatic (utterance + user context, allq)
# python3 tools/convert_run_to_monot5.py \
#   --run runs/cast20.automatic.rewrite.spr.top1000.trec \
#   --corpus data/cast20/collections/ \
#   --topic_queries data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --use_query_key utterance \
#   --output_text_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.top1000.uttr.conv.monot5.text_pairs.txt \
#   --output_id_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.top1000.uttr.conv.monot5.id_pairs.txt \
#   --use_context 0 & 
#
# # automatic (automatci + user context, allq) # failed
# python3 tools/convert_run_to_monot5.py \
#   --run runs/cast20.automatic.rewrite.spr.top1000.trec \
#   --corpus data/cast20/collections/ \
#   --topic_queries data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --use_query_key automatic_rewritten \
#   --output_text_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.top1000.auto.conv.monot5.text_pairs.txt \
#   --output_id_pair data/cast20/conv-monot5-pairs/cast20.automatic.rewrite.top1000.auto.conv.monot5.id_pairs.txt \
#   --use_context 0 &
#
# # automatic (standard)
# python3 tools/convert_run_to_monot5.py \
#   --run runs/cast20.automatic.rewrite.spr.top1000.trec \
#   --corpus data/cast20/collections/ \
#   --topic_queries data/cast20/2020_evaluation_topics_v1.0.jsonl \
#   --use_query_key automatic_rewritten \
#   --output_text_pair data/cast20/monot5-pairs/cast20.automatic.rewrite.top1000.monot5.text_pairs.txt \
#   --output_id_pair data/cast20/monot5-pairs/cast20.automatic.rewrite.top1000.monot5.id_pairs.txt
#
