# <center><img src="img/icon.png" style="width: 3%">  Can LLMs be Good Graph Judger for Knowledge Graph Construction?</center>

This is the repo for the paper [Can LLMs be Good Graph Judger for Knowledge Graph Construction?](https://arxiv.org/abs/2411.17388). We will release the model checkpoints later.

![Illustration of multi-agent collaborative framework](./img/graphjudger.png)

## Project Structure
```
.
├── README.md
├── .gitignore
├── LICENSE
├── chat
│   ├── run_chatgpt_entity.py
│   ├── run_chatgpt_triple.py
│   └── run_chatgpt.py
├── datasets
│   ├── GPT3.5_result_GenWiki-Hard
│   ├── GPT3.5_result_rebel_sub
│   ├── GPT3.5_result_SCIERC
│   └── process_data.ipynb
├── graph_evaluation
│   ├── metrics
│   │   ├── eval.py
│   │   └── graph_matching.py
│   └── eval.sh
├── graphjudger
│   ├── data
│   │   ├── genwiki
│   │   ├── rebel_sub
│   │   └── scierc
│   ├── models
│   │   ├── genwiki
│   │   ├── rebel_sub
│   │   └── scierc
│   ├── lora_finetune_genwiki.py
│   ├── lora_finetune_rebel.py
│   ├── lora_finetune_scierc.py
│   ├── lora_infer_genwiki.py
│   ├── lora_infer_rebel.py
│   └── lora_infer_scierc.py
└── img
    ├── graphjudger.png
    ├── graphjudger2.png
    └── icon.png
```

## Guidance 

Coming soon.