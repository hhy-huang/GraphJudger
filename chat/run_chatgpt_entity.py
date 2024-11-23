"""
Target: Extract the entities from raw / denoised text, and 
        denoise the text with extracted entities.
"""

import os
import openai

openai.api_key = "***"
openai.api_base = "***"

import time
import json
import functools
from tqdm import tqdm

# read the text to be denoised
text = []
dataset = "GenWiki-Hard"     # rebel / webnlg / kelm
dataset_path = f'./datasets/GPT3.5_result_{dataset}/'
Iteration = 3

if Iteration == 1:                      # raw text
    with open(dataset_path + 'test.target', 'r') as f:
        for l in f.readlines():
            text.append(l.strip())
else:                                   # denoised text from last itr
    with open(dataset_path + f'Iteration{Iteration - 1}/test_denoised.target', 'r') as f:
        for l in f.readlines():
            text.append(l.strip())

# with open('./datasets/GPT3.5_result_KELM/test_500.target', 'r') as f:
#     for l in f.readlines():
#         text.append(l.strip())
#with open('GPT3.5_result_WebNLG/test.target', 'r') as f:
#    for l in f.readlines():
#        text.append(l.strip())
# with open('GPT3.5_result_GenWiki/test.target', 'r') as f:
#     for l in f.readlines():
#         text.append(l.strip())


@functools.lru_cache()
def get_chatgpt_completion(content):
    got_result = False
    while not got_result:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role": "user", "content": content}],
            temperature=0
            )
            got_result = True
        except Exception as e:
            print(e)
            time.sleep(1)
    return response["choices"][0][ "message"]["content"]

# extract entity
with open(dataset_path + f"Iteration{Iteration}/test_entity.txt", "w") as output_file:
    for i in tqdm(range(len(text))):
        # entity extraction Prompt
        prompt = f"Goal:\nTransform the text into a list of entities.\n\nHere are two examples:\nExample#1: \nText: \"Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\"\nList of entities: [\"Shotgate Thickets\", \"Nature reserve\", \"United Kingdom\", \"Essex Wildlife Trust\"]\nExample#2: \nText: \"Garczynski Nunatak () is a cone-shaped nunatak, the highest in a cluster of nunataks close west of Mount Brecher, lying at the north flank of Quonset Glacier in the Wisconsin Range of the Horlick Mountains of Antarctica. It was mapped by the United States Geological Survey from surveys and U.S. Navy air photos, 1959–60, and was named by the Advisory Committee on Antarctic Names for Carl J. Garczynski, a meteorologist in the Byrd Station winter party, 1961.\"\nList of entities: [\"Garczynski Nunatak\", \"nunatak\", \"Wisconsin Range\", \"Mount Brecher\", \"Quonset Glacier\", \"Horlick Mountains\"]\n\nRefer to the examples and here is the question:\nText: " + text[i] + "\nList of entities:"
        response = get_chatgpt_completion(prompt)
        output_file.write(response.strip().replace('\n','') + '\n')

# read last denoised entities
last_extracted_entities = []
with open(dataset_path + f'Iteration{Iteration}/test_entity.txt', 'r') as f:
    for l in f.readlines():
        last_extracted_entities.append(l.strip())

# denoise text
with open(dataset_path + f"Iteration{Iteration}/test_denoised.target", "w") as output_file:
    for i in tqdm(range(len(text))):
        # entity extraction Prompt
        prompt = f"Goal:\nDenoise the raw text with the given entities, which means remove the unrelated text and make it more formatted.\n\nHere are two examples:\nExample#1:\nRaw text: \"Zakria Rezai (born 29 July 1989) is an Afghan footballer who plays for Ordu Kabul F.C., which is a football club from Afghanistan. He is also an Afghanistan national football team player, and he has 9 caps in the history. He wears number 14 on his jersey and his position on field is centre back.\"\nEntities: [\"Zakria Rezai\",\"footballer\",\"Ordu Kabul F.C.\",\"Afghanistan\",\"29 July 1989\"]\nDenoised text: \"Zakria Rezai is a footballer. Zakria Rezai is a member of the sports team Ordu Kabul F.C. Zakria Rezai has the citizenship of Afghanistan. Zakria Rezai was born on July 29, 1989. Ordu Kabul F.C. is a football club. Ordu Kabul F.C. is based in Afghanistan.\"\nExample#2:\nRaw text: \"Elizabeth Smith, a renowned British artist, was born on 12 May 1978 in London. She is specialized in watercolor paintings and has exhibited her works in various galleries across the United Kingdom. Her most famous work, 'The Summer Breeze,' was sold at a prestigious auction for a record price. Smith is also a member of the Royal Society of Arts and has received several awards for her contributions to the art world.\"\nEntities: [\"Elizabeth Smith\", \"British artist\", \"12 May 1978\", \"London\", \"watercolor paintings\", \"United Kingdom\", \"The Summer Breeze\", \"Royal Society of Arts\"]\nDenoised text: \"Elizabeth Smith is a British artist. Elizabeth Smith was born on May 12, 1978. Elizabeth Smith was born in London. Elizabeth Smith specializes in watercolor paintings. Elizabeth Smith's artwork has been exhibited in the United Kingdom. 'The Summer Breeze' is a famous work by Elizabeth Smith. Elizabeth Smith is a member of the Royal Society of Arts.\"\n\nRefer to the examples and here is the question:\nRaw text: " + text[i] + "\nEntities: " + last_extracted_entities[i] + "\nDenoised text:"
        response = get_chatgpt_completion(prompt)
        output_file.write(response.strip().replace('\n','') + '\n')
