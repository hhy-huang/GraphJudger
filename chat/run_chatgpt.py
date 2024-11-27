import os
import aiohttp
import asyncio
import json
import functools
from tqdm.asyncio import tqdm

# Set API key and base URL
api_key = "***"
api_base = "***"

# Read the text
text = []
entity = []
folder = "GPT3.5_result_GenWiki-Hard"
with open(f'./datasets/{folder}/test.target', 'r') as f:
    text = [l.strip() for l in f.readlines()]

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

async def get_chatgpt_completion(session, content):
    url = f"{api_base}/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7
    }
    while True:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            await asyncio.sleep(1)

async def process_texts():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(len(text)):
            prompt = (
                "Transform the text into a semantic graph. \nExample: \nText: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\n"
                "Semantic Graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\n"
                f"Text: {text[i]}\nSemantic graph:"
            )
            tasks.append(get_chatgpt_completion(session, prompt))

        responses = await tqdm.gather(*tasks)

        with open(f"./datasets/{folder}/gpt_baseline/test_generated_graphs.txt", "w") as output_file:
            for response in responses:
                output_file.write(response.strip().replace('\n', '') + '\n')

# Run the async process
asyncio.run(process_texts())