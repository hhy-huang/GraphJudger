# gpt baseline
# python metrics/eval.py \
#     --pred_file ../datasets/GPT3.5_result_rebel/gpt_baseline/test_generated_graphs.txt\
#     --gold_file ../datasets/GPT3.5_result_rebel/test.source

# best
# python metrics/eval.py \
#     --pred_file ../datasets/GPT3.5_result_rebel/Graph_Iteration1/test_generated_graphs_final_2.txt\
#     --gold_file ../datasets/GPT3.5_result_rebel/test.source

python metrics/eval.py \
    --pred_file ../datasets/GPT3.5_result_GenWiki-Hard/pive_iter2/test_generated_graphs.txt\
    --gold_file ../datasets/GPT3.5_result_GenWiki-Hard/test.source