# text2KG

## Files Introduction
1. `GenWiki-HIQ` is the created dataset using verifier module, which contains 110K parallel graph-text pairs.
3. `datasets` contains the used REBEL dataset and `data_process.ipynb` to create the training data for the verifier module and test data for each iteration.
4. `graph_evaluation` contains the graph evaluation metrics.
5. `chat`contains the sctipts to prompt LLMs.
6. `single_verifier` contains the training sctipt for single verifier using T5-Large.


## Guidance 

The source data are in `./datasets/rebel`, which is processed by `./datasets/rebel/preprocess.ipynb` with raw data.

Then put them in `./datasets/GPT3.5_result_rebel`, where most of the time our data are.

For out methods, Run:

```shell
python ./chat/run_chatgpt_entity.py
```

Get the denoised text data and entities. Don't forget to change the iteration parameters in the codes. The results are under `./datasets/GPT3.5_result_rebel/Iteration{i}`. Then Run:

```shell
python ./chat/run_chatgpt_triple.py
```

Get the generated graph with denoised text data and entities under `.datasets/GPT3.5_result_rebel/Graph_Iteration{i}`.

Then you need to do the task of triple classification based on training data `.datasets/GPT3.5_result_rebel/train.source` and your generated graph `./datasets/GPT3.5_result_rebel/Graph_Iteration{i}/test_generated_graphs.txt`. Before the classification, you need to generate the training data for KG completion model. You can do that in `./datasets/GPT3.5_result_rebel/prepare_KGCom.ipynb`. You can finish training data generation, test data generation and triple filtering here.(3 cells)

So for triple classification(TC), you need to firstly run the first cell in `./datasets/prepare_KGCom.ipynb`, getting the training data `./datasets/GPT3.5_result_rebel/train_instructions_llama.json`. Then put that in `./graph_judger/data/rebel`, and then Run:

```shell
cd ./graph_judger
python lora_finetune_rebel.py
```

Then we can get the fine-tuned model. Then run the second cell of `./datasets/GPT3.5_result_rebel/prepare_KGCom.ipynb` to generate test data from generated triples. And move that under `./graph_judger/data/rebel`, which is `test_instructions_llama.csv`. Put that under `./graph_judger/data/rebel`. Then Run:

```shell
cd ./graph_judger
python lora_infer_rebel.py
```

Then we can get the classification result for every triple we generated, which is `pred_instructions_llama2_7b.csv`. Put that under `./datasets/GPT3.5_result_rebel/Graph_Iteration1`. Then run the third cell of `./datasets/GPT3.5_result_rebel/prepare_KGCom.ipynb` to remove the triples in `./datasets/GPT3.5_result_rebel/Graph_Iteration1/test_generated_graphs.txt` with the label we predicted in `pred_instructions_llama2_7b.csv`. Then we will get the final result of our method `./datasets/GPT3.5_result_rebel/Graph_Iteration1/test_generated_graphs_final.txt`(currently regard of the verifier module)

For evaluation, modify the path in `./graph_evaluation/eval.sh` to evaluate the result. Don't forget to put the Bert model under `./graph_evaluation/`.

```shell
cd ./graph_evaluation
bash eval.sh
```

For baseline, Run:

```shell
python ./chat/run_chatgpt.py
```

## Data

Now the evaluation data:

GPT3.5-turbo:   `./datasets/GPT3.5_result_rebel/gpt_baseline/test_generated_graphs.txt`

Iter_denoised_only:     `./datasets/GPT3.5_result_rebel/Graph_Iteration2/test_generated_graphs.txt`

Iter_denoised_&TC:    `./datasets/GPT3.5_result_rebel/Graph_Iteration2/test_generated_graphs.txt`