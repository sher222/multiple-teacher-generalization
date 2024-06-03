import json
from together import Together
from tqdm import tqdm
import os 
import pandas as pd
import argparse

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def get_category(subject):
    for category, subjects in categories.items():
        if subject in subjects:
            return category
    return "other"

def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

easy_train = load_json("easy_train.json")
easy_test = load_json("easy_test.json")
hard_train = load_json("hard_train.json")
hard_test = load_json("hard_test.json")

def generate_data(args):
    data_path = args.data_path
    dirs = ["dev", "val", "train"] if not args.dev else ["dev"]
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    together_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    questions = []
    for dir_ in dirs:
        for subcategory, subjects in subcategories.items():
            for subject in subjects:
                path = os.path.join(data_path, dir_, f"{subcategory}_{dir_}.csv")
                print(f"Verifying path: {path}")
                if not os.path.exists(path):
                    print(f"Path does not exist: {path}")
                    continue
                df = pd.read_csv(path, header=None)
                if df.empty:
                    print(f"Empty dataframe: {path}")
                    continue
                print(f"Loaded dataframe with {len(df)} rows from {path}")
                for i, row in tqdm(df.iterrows(), desc=f"{subject} {dir_}", total=len(df)):
                    if len(row) != 6:
                        print(f"Skipping row with unexpected length: {row}")
                        continue
                    question = row[0]
                    answer = row[5]
                    question_str = f"{question}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}"
                    question = {
                        "prompt": question_str,
                        "correct_ans": answer,
                        "subject": subject,
                        "category": get_category(subject),
                        "difficulty": dir_.split('_')[0]  # Assuming the difficulty is part of the directory name
                    }
                    questions.append(question)
                    print(f"Added question: {question_str[:50]}...")

    savepath = "questions.json" if not args.dev else "questions_dev.json"
    with open(savepath, "w") as f:
        json.dump(questions, f, indent=4)

    return questions

def generate_responses(questions, model):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    responses = []
    for question in tqdm(questions, desc="generating responses"):
        preferred_prompt = f"You are an expert in {question['subject']}. The following multiple choice question has the correct answer {question['correct_ans']}. Provide an explanation of why {question['correct_ans']} is the correct answer. Do not mention the letter {question['correct_ans']} in your explanation.\n{question['prompt']}"
        dispreferred_prompt = f"Answer the following multiple choice question without being given the correct answer in advance.\n{question['prompt']}"

        response_preferred = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": preferred_prompt}]
        )
        response_dispreferred = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": dispreferred_prompt}]
        )
        
        responses.append({
            "prompt": question['prompt'],
            "correct_ans": question['correct_ans'],
            "subject": question['subject'],
            "category": question['category'],
            "difficulty": question['difficulty'],
            "preferred_response": response_preferred.choices[0].message.content.strip(),
            "dispreferred_response": response_dispreferred.choices[0].message.content.strip()
        })
    return responses

def generate_responses(questions, model):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    responses = []
    for question in tqdm(questions, desc="generating responses"):
        preferred_prompt = f"You are an expert in {question['subject']}. The following multiple choice question has the correct answer {question['correct_ans']}. Provide an explanation of why {question['correct_ans']} is the correct answer. Do not mention the letter {question['correct_ans']} in your explanation.\n{question['prompt']}"
        dispreferred_prompt = f"Answer the following multiple choice question without being given the correct answer in advance.\n{question['prompt']}"

        response_preferred = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": preferred_prompt}]
        )
        response_dispreferred = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": dispreferred_prompt}]
        )
        
        responses.append({
            "prompt": question['prompt'],
            "correct_ans": question['correct_ans'],
            "subject": question['subject'],
            "category": question['category'],
            "difficulty": question['difficulty'],
            "preferred_response": response_preferred.choices[0].message.content.strip(),
            "dispreferred_response": response_dispreferred.choices[0].message.content.strip()
        })
    return responses

def save_responses(responses, filepath):
    with open(filepath, "w") as f:
        json.dump(responses, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="generate preference data")
    parser.add_argument("--data_path", type=str, required=True, help="mmlu path")
    parser.add_argument("--dev", action="store_true", help="if using the dev set")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Data path {args.data_path} does not exist.")
        return

    questions = generate_data(args)
    if not questions:
        print("no questions generating, exiting.")
        return

    responses = generate_responses(questions, "mistralai/Mixtral-8x22B-Instruct-v0.1")
    save_responses(responses, "evaluated_responses.json")
    print("saved successfully.")

if __name__ == "__main__":
    main()

#python subject_preference_data.py --data_path /Users/kushalthaman/multiple-teacher-generalization/data_gen/together/data --dev
