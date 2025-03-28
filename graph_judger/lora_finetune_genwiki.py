import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.system("pip install datasets")
#os.system("pip install deepspeed")
#os.system("pip install accelerate")
#os.system("pip install transformers>=4.28.0")
import sys
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from datasets import load_dataset
import transformers
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128        # 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 4  # we don't always need 3 tbh
LEARNING_RATE = 3e-5  # the Karpathy constant
CUTOFF_LEN = 64  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
# DATA_PATH = "data/FB13/train_instructions_llama.json"
# DATA_PATH = "data/WN11/train_instructions_llama.json"
# DATA_PATH = "data/rebel/train_instructions_llama.json"
# DATA_PATH = "data/webnlg/train_instructions_llama.json"
DATA_PATH = "data/genwiki/train_instructions_llama.json"

OUTPUT_DIR = "models/llama2-7b-lora-genwiki"
# OUTPUT_DIR = "models/llama2-7b-lora-rebel"
# OUTPUT_DIR = "models/llama2-7b-lora-wn18rr"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
model = LlamaForCausalLM.from_pretrained(
    "models/llama-2-7b-hf",
    # load_in_8bit=True,
    # device_map=device_map
).half().cuda()
tokenizer = LlamaTokenizer.from_pretrained(
    "models/llama-2-7b-hf", add_eos_token=True
)
# model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config).cuda()
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files=DATA_PATH)
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""
def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }
def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
"""
        )
        if data_point["input"]
        else (
#             f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# {data_point["instruction"]}
# ### Response is
# """
#         )
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
            )["input_ids"]
        )
        - 1
    )  # no eos token
    result = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    full_tokens = result["input_ids"][:-1]
    attention_mask = result["attention_mask"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": attention_mask
#         "attention_mask": [1] * len(full_tokens)

    }
if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        optim="adamw_torch",
        report_to='tensorboard',
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))
# if torch.__version__ >= "2" and sys.platform != "win32":
#   model = torch.compile(model)
trainer.train()
# FIXME:
# model.save_pretrained(OUTPUT_DIR)                                   # .safetensor
model.save_pretrained(OUTPUT_DIR, only_save_weights=True)           # .bin
print("\n If there's a warning about missing keys above, please disregard :)")