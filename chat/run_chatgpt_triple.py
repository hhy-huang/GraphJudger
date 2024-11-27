import os
import aiohttp
import asyncio
import json
import functools
from tqdm.asyncio import tqdm

# Set API key and base URL
api_key = "***"
api_base = "***"

# Read the text to be denoised
text = []
entity = []
dataset = "GenWiki-Hard"
dataset_path = f'./datasets/GPT3.5_result_{dataset}/'
Denoised_Iteration = 3
Graph_Iteration = 3

# Read denoised text
with open(dataset_path + f'Iteration{Denoised_Iteration}/test_denoised.target', 'r') as f:
    text = [l.strip() for l in f.readlines()]

# Read the corresponding entities
with open(dataset_path + f'Iteration{Denoised_Iteration}/test_entity.txt', 'r') as f:
    entity = [l.strip() for l in f.readlines()]

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

async def get_chatgpt_completion(session, content):
    url = f"{api_base}/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1
    }
    async with session.post(url, headers=headers, json=payload) as response:
        result = await response.json()
        return result["choices"][0]["message"]["content"]

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(len(text)):
            prompt = (
                f"Goal:\nTransform the text into a semantic graph(a list of triples) with the given text and entities. "
                f"In other words, You need to find relations between the given entities with the given text.\n"
                f"Attention:\n1.Generate triples as many as possible. "
                f"2.Make sure each item in the list is a triple with strictly three items.\n\n"
                f"Here are two examples:\n"
                f"Example#1: \nText: \"Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\"\n"
                f"Entity List: [\"Shotgate Thickets\", \"Nature reserve\", \"United Kingdom\", \"Essex Wildlife Trust\"]\n"
                f"Semantic Graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], "
                f"[\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\n"
                f"Example#2:\nText: \"The Eiffel Tower, located in Paris, France, is a famous landmark and a popular tourist attraction. "
                f"It was designed by the engineer Gustave Eiffel and completed in 1889.\"\n"
                f"Entity List: [\"Eiffel Tower\", \"Paris\", \"France\", \"landmark\", \"Gustave Eiffel\", \"1889\"]\n"
                f"Semantic Graph: [[\"Eiffel Tower\", \"located in\", \"Paris\"], [\"Eiffel Tower\", \"located in\", \"France\"], "
                f"[\"Eiffel Tower\", \"instance of\", \"landmark\"], [\"Eiffel Tower\", \"attraction type\", \"tourist attraction\"], "
                f"[\"Eiffel Tower\", \"designed by\", \"Gustave Eiffel\"], [\"Eiffel Tower\", \"completion year\", \"1889\"]]\n\n"
                f"Refer to the examples and here is the question:\nText: {text[i]}\nEntity List:{entity[i]}\nSemantic graph:"
            )
            tasks.append(get_chatgpt_completion(session, prompt))

        responses = await tqdm.gather(*tasks)

        # Write responses to the output file
        with open(dataset_path + f"Graph_Iteration{Graph_Iteration}/test_generated_graphs.txt", "w") as output_file:
            for response in responses:
                output_file.write(response.strip().replace('\n', '') + '\n')

# Run the async main function
asyncio.run(main())