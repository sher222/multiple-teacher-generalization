import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from together import Together
import sys
sys.path.insert(1, '../api')
from togetherclass import TogetherModel

def chat_completion(subjects, questions):
    together_model = TogetherModel("mistralai/Mixtral-8x22B-Instruct-v0.1")

    prompts = [question["correct_prompt"]["prompt"] for question in questions]
    responses = together_model.get_responses(prompts)

    wrong_prompts = [wrong_prompt["prompt"] for question in questions for wrong_prompt in question["wrong_prompts"]]
    wrong_responses = together_model.get_responses(wrong_prompts)
    
    possible_ans = ["A", "B", "C", "D"]

    examples = {subject: [] for subject in subjects}
    for i, question in enumerate(tqdm(questions)):
        example = {"prompt": question["question"], "wrong_response": []}
        answer = f"{question["correct_prompt"]["letter"]}. {responses[i]}"
        example["correct_response"] = answer
        for j, wrong_prompt in enumerate(question["wrong_prompts"]):
            answer = f"{wrong_prompt["letter"]}. {wrong_responses[i*3+j]}"
            example["wrong_response"].append(answer)
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
        savepath = f"responses_{subject}_mt.json" if not args.dev else f"responses_{subject}_dev_mt.json"
        with open(savepath, "w") as f:
            json.dump(examples[subject], f, indent=4)

        
    