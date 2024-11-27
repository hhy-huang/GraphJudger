import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.system("pip install datasets")
#os.system("pip install deepspeed")
#os.system("pip install accelerate")
#os.system("pip install transformers>=4.28.0")
import sys
import torch
import argparse
import pandas as pd
from peft import PeftModel
from tqdm import tqdm
import transformers
assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


tokenizer = LlamaTokenizer.from_pretrained("models/llama-2-7b-hf")
LOAD_8BIT = False
BASE_MODEL = "models/llama-2-7b-hf"
# LORA_WEIGHTS = "models/llama2-7b-lora-wn18rr/"
# LORA_WEIGHTS = "models/llama2-7b-lora-wn11/"
# LORA_WEIGHTS = "models/llama2-7b-lora-FB13/"
LORA_WEIGHTS = "models/llama2-7b-lora-gen/"
# LORA_WEIGHTS = "models/llama2-7b-lora-webnlg/"
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        # torch_dtype=torch.float16,
        # device_map="auto",
    ).half().cuda()
    pipeline = transformers.pipeline (
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device=torch.device("cuda:0")
                )
    pipeline.model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
    ).half().cuda()
    # model = PeftModel.from_pretrained(
    #     model,
    #     LORA_WEIGHTS,
    #     # torch_dtype=torch.float16,
    # ).half().cuda()
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""
# if not LOAD_8BIT:
#     model.half()  # seems to fix bugs for some users.
model.eval()
# if torch.__version__ >= "2" and sys.platform != "win32":
#     model = torch.compile(model)
def evaluate(
    instruction,
    input=None,
    temperature=0.05,       # 0.1
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,      # 256
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    print("Input:")
    print(prompt)
    sequences = pipeline (
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=64,
                )
    # with torch.no_grad():
    #     generation_output = model.generate(
    #         input_ids=input_ids,
    #         generation_config=generation_config,
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #         max_new_tokens=max_new_tokens,
    #     )
    # s = generation_output.sequences[0]
    # output = tokenizer.decode(s)
    print("Output:")
    output = ""
    for seq in sequences:
        print (f"{seq['generated_text']}") 
        output = seq['generated_text']
    # exit(0)
    return output.split("### Response:")[1].strip()
    # return output

if __name__ == "__main__":
    # testing code for readme
    parser = argparse.ArgumentParser()
    # parser.add_argument("--finput", type=str, default="data/WN11/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN11/pred_instructions_llama2_7b.csv")
    # parser.add_argument("--finput", type=str, default="data/FB13/test_instructions_llama.csv")
    # parser.add_argument("--foutput", type=str, default="data/FB13/pred_instructions_llama2_7b.csv")
    # parser.add_argument("--finput", type=str, default="data/rebel/test_instructions_llama2_7b_itr2.csv")
    # parser.add_argument("--foutput", type=str, default="data/rebel/pred_instructions_llama2_7b_itr2.csv")
    parser.add_argument("--finput", type=str, default="data/genwiki/test_instructions_llama2_7b_itr3.csv")
    parser.add_argument("--foutput", type=str, default="data/genwiki/pred_instructions_llama2_7b_itr3.csv")
    # parser.add_argument("--finput", type=str, default="data/WN18RR/test_instructions_llama_merge.csv")
    # parser.add_argument("--foutput", type=str, default="data/WN18RR/pred_instructions_llama2_7b_merge.csv")
    args = parser.parse_args()
    total_input = pd.read_csv(args.finput, header=0, sep=',')
    instruct, pred = [], []
    total_num = len(total_input)
    for index, data in tqdm(total_input.iterrows()):
        tqdm.write(f'{index}/{total_num}')
        cur_instruct = data['prompt']
        cur_response = evaluate(cur_instruct)
        cur_response = cur_response.replace('\n', ',')
        # tqdm.write(cur_response)
        pred.append(cur_response)
        instruct.append(cur_instruct)
    
    output = pd.DataFrame({'prompt': instruct, 'generated': pred})
    output.to_csv(args.foutput, header=True, index=False)

