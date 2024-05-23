import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from together import Together

def chat_completion(subjects, questions):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    together_model_correct = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    together_model_wrong = "mistralai/Mixtral-7B-Instruct-v0.3"

    possible_ans = ["A", "B", "C", "D"]

    examples = {subject: [] for subject in subjects}
    for question in tqdm(questions):
        example = {"prompt": question["question"], "wrong_response": []}
        response = client.chat.completions.create(
            model=together_model_correct,
            messages=[{"role": "user", "content": question["correct_prompt"]["prompt"]}],
        )   
        answer = f"{question["correct_prompt"]["letter"]}. {response.choices[0].message.content}"
        example["correct_response"] = answer
        # question["correct_response"] = f"{question['correct_prompt']['letter']}. {responses[i]}"
        for wrong_prompt in question["wrong_prompts"]:
            response = client.chat.completions.create(
                model=together_model_wrong,
                messages=[{"role": "user", "content": wrong_prompt["prompt"]}],
            )
            answer = f"{wrong_prompt["letter"]}. {response.choices[0].message.content}"
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
        savepath = f"responses_{subject}.json" if not args.dev else f"responses_{subject}_dev.json"
        with open(savepath, "w") as f:
            json.dump(examples[subject], f, indent=4)

        
    