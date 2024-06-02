from together import Together
import ipdb
import pandas as pd
import pprint
from multiprocessing import Pool, cpu_count
import tqdm
import together 
import time
class TogetherModel():
    api_keys = [ #02 - 06
        '303e07102419a98aae9be912b8e612cf16c0a54babfde4927bf4e86e9a4b76d6',
        '96b81f69214836f2d22785fbcff82e8b48c78797c7683c34242a70739bdb378b',
        'cd073033f34fdfe978e4ef77f47b5ad7d0508fde15dd9b02802c4c7fca4eb126',
        '13649da461df6431e6505013783d8b8e724311ad1996c6b1a9aba890c3095cc4',
        'b8a9a5e93fcaf5d49a5579fde6e9708bee173848a5fa884af50129daff2543b8'
    ]

    def __init__(self, models):
        self.models = models
    def thread_call(self, args):
        max_input_tokens = 5000
        model, prompt, api_key, temp, top_k, top_p = args
        # Set the API key for the current process
        # Assuming the 'together' client or similar can be configured on a per-call basis
        client = Together(api_key = api_key)
        if len(prompt.split(" ")) > max_input_tokens:
            prompt = " ".join(prompt.split(" ")[:max_input_tokens])
        while True:
            try:
                output = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=512,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=1.0,
                )
                return {
                    'model':      model,
                    'prompt':   prompt,
                    'output':     output.choices[0].message.content,
                    'api_key':    api_key,  # For verification or logging, if needed
                    'raw_output': output,
                    'temp': temp,
                    'top_k': top_k,
                    'top_p': top_p
                }
            except together.error.RateLimitError:
                time.sleep(2)
            except together.error.InvalidRequestError:
                prompt = " ".join(prompt.split(" ")[:max_input_tokens/2])

    def get_responses(self, prompts, temp=0.7, top_k=1, top_p=0.9):
        rows = []
        tasks = [(model, question, self.api_keys[i % len(self.api_keys)], temp, top_k, top_p) for i, (model, question)
             in enumerate([(model, question) for model in self.models for question in prompts])]

        with Pool(processes=len(self.api_keys) * 3) as pool:
            # Use imap or imap_unordered for non-blocking pool map operation
            # Wrap the pool.imap or pool.imap_unordered call with tqdm for progress bar
            # Note: Adjust `total` parameter as per your tasks length if necessary
            print(tasks[0])
            for result in tqdm.tqdm(pool.imap(self.thread_call, tasks), total=len(tasks)):
                rows.append(result)
        return rows



if __name__ == '__main__':

    models = [
        'meta-llama/Llama-3-8b-chat-hf',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'Qwen/Qwen1.5-7B-Chat',
    ]
    tm = TogetherModel(models)
    long_prompt = "this is a sentence" * 8000
    res = tm.get_responses(["Given the following claim, output TRUE, FALSE, or UNSURE. Do not output any other words, just the verdict on the claim. Claim: There exists a producer and an actor called Simon Pegg."])
    print("RESULT")
    print(res)
    breakpoint()