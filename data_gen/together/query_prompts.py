import os, asyncio
import json
import argparse
from together import Together
from tqdm import tqdm

def chat_completion_short(messages):
    examples = []
    for message in tqdm(messages):
        examples.append({"prompt": message["question"], "response": f"{message["answer"]}."})
    return examples

def chat_completion(messages):
    client = Together(api_key="2713b3ba1312fb107ffa5db2e946eb43212a27967cd7503924f97be148a9e42a")
    together_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    examples = []
    for message in tqdm(messages):
        response = client.chat.completions.create(
            model=together_model,
            messages=[{"role": "user", "content": message["prompt"]}],
        )
        answer = f"{message["answer"]}. {response.choices[0].message.content}"
        examples.append({"prompt": message["question"], "response": answer})
    
    return examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model answers for MMLU questions")
    parser.add_argument("--length", type=str, default="medium", help="Length of the answer")
    parser.add_argument("--dev", action="store_true", help="Use the dev set")
    args = parser.parse_args()

    promptfile_len = "medium" if args.length == "short" else args.length

    json_path = f"prompts_{promptfile_len}.json" if not args.dev else f"prompts_{promptfile_len}_dev.json"
    with open(json_path, "r") as f:
        messages = json.load(f)

    if args.length == "short":
        examples = chat_completion_short(messages)
    else:
        examples = chat_completion(messages)
    output_path = f"responses_{args.length}.json" if not args.dev else f"responses_{args.length}_dev.json"
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=4)

