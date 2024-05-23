import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from together import Together
import sys
sys.path.insert(1, '../api')
from togetherclass import TogetherModel


'''
Attempt to use Sheryl's multithreading code to query models...
having rate limit issues and degenerate responses
'''
def chat_completion(subjects, questions):
    together_model = TogetherModel("mistralai/Mixtral-8x22B-Instruct-v0.1")

    prompts = [question["prompt"] for question in questions]
    responses = together_model.get_responses(prompts)

    # together_model = TogetherModel("Qwen/Qwen1.5-1.8B-Chat")
    # wrong_prompts = [question["wrong_prompt"] for question in questions]
    # wrong_responses = together_model.get_responses(wrong_prompts)
    possible_ans = ["A", "B", "C", "D"]

    examples = {subject: [] for subject in subjects}
    for i, question in enumerate(tqdm(questions)):
        example = {"prompt": question["question"]}
        answer = f"{question["answer"]}. {responses[i]}"
        example["response"] = answer
        # example["unpreferred_response"] = wrong_responses[i]
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
        savepath = f"responses_{subject}_mxt.json" if not args.dev else f"responses_{subject}_dev_mxt.json"
        with open(savepath, "w") as f:
            json.dump(examples[subject], f, indent=4)

        
    