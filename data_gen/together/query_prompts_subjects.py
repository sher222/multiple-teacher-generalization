import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from together import Together

def chat_completion(subjects, questions):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    together_model_correct = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    together_model_wrong = "Qwen/Qwen1.5-1.8B-Chat"

    possible_ans = ["A", "B", "C", "D"]

    examples = {subject: [] for subject in subjects}
    for question in tqdm(questions):
        example = {"prompt": question["question"]}
        response = client.chat.completions.create(
            model=together_model_correct,
            messages=[{"role": "user", "content": question["prompt"]}],
        )   
        answer = f"{question["answer"]}. {response.choices[0].message.content}"
        example["preferred_response"] = answer
        response = client.chat.completions.create(
            model=together_model_wrong,
            messages=[{"role": "user", "content": question["wrong_prompt"]}],
        )
        answer = f"{response.choices[0].message.content}"
        example["unpreferred_response"] = answer
        examples[question["subject"]].append(example)
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get model answers for MMLU questions")
    parser.add_argument("--dev", action="store_true", help="Use the dev set")
    args = parser.parse_args()

    json_path = f"prompts_mmlu.json" if not args.dev else f"prompts_mmlu_dev.json"
    with open(json_path, "r") as f:
        questions = json.load(f)

    subjects = ["biology", "chemistry", "computer_science", "mathematics", "physics"]

    examples = chat_completion(subjects, questions)

    for subject in subjects:
        savepath = f"responses_{subject}.json" if not args.dev else f"responses_{subject}_dev.json"
        with open(savepath, "w") as f:
            json.dump(examples[subject], f, indent=4)

        
    