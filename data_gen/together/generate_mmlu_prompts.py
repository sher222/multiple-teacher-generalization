import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from together import Together

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

def generate_data(args):
    data_path = args.data_path
    dirs = ["dev", "val", "test"] if not args.dev else ["dev"]
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    together_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    questions = []
    for dir_ in dirs:
        for subcategory, subjects in subcategories.items():
            for subject in subjects:
                path = os.path.join(data_path, dir_, f"{subcategory}_{dir_}.csv")
                print(f"verifying path: {path}")  
                if not os.path.exists(path):
                    print(f"the path does not exist: {path}")   
                    continue
                df = pd.read_csv(path, header=None)
                if df.empty:
                    print(f"empty df: {path}")   
                    continue
                print(f"loaded df with {len(df)} rows from {path}")   
                for i, row in tqdm(df.iterrows(), desc=f"{subject} {dir_}"):
                    assert len(row) == 6
                    question = row[0]
                    answer = row[5]
                    question_str = f"{question}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}"
                    question = {
                        "prompt": question_str,
                        "correct_ans": answer,
                        "subject": subject,
                        "category": get_category(subject)
                    }
                    questions.append(question)
                    print(f"added question: {question_str[:50]}...")   


    savepath = "questions.json" if not args.dev else "questions_dev.json"
    with open(savepath, "w") as f:
        json.dump(questions, f, indent=4)

    return questions

def split_data(questions, ratio=0.7):
    easy_questions = [q for q in questions if q['difficulty'] == 'easy']
    hard_questions = [q for q in questions if q['difficulty'] == 'hard']

    def split(questions, ratio):
        split_point = int(len(questions) * ratio)
        return questions[:split_point], questions[split_point:]

    easy_train, easy_test = split(easy_questions, ratio)
    hard_train, hard_test = split(hard_questions, ratio)

    return easy_train, easy_test, hard_train, hard_test

def assign_difficulty(questions, model):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    for question in tqdm(questions, desc="question difficulty"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Is the following question easy or hard? Answer in only a single word (easy or hard) \n{question['prompt']}"}]
        )
        difficulty = response.choices[0].message.content.strip().lower()
        if "easy" in difficulty:
            question['difficulty'] = 'easy'
        else:
            question['difficulty'] = 'hard'
    return questions

data_path = "/juice4/scr4/nlp/music/old_lakh_datasets/kathli/data"

def main():
    parser = argparse.ArgumentParser(description="generate MMLU splits")
    parser.add_argument("--data_path", type=str, required=True, help="path to MMLU")
    parser.add_argument("--dev", action="store_true", help="if using  dev set")
    parser.add_argument("--length", type=str, default="medium", help="length of the explanation")

    args = parser.parse_args()

    questions = generate_data(args)
    questions = assign_difficulty(questions, "mistralai/Mixtral-8x22B-Instruct-v0.1")
    easy_train, easy_test, hard_train, hard_test = split_data(questions)
    print("a")
    with open("easy_train.json", "w") as f:
        json.dump(easy_train, f, indent=4)
    with open("easy_test.json", "w") as f:
        json.dump(easy_test, f, indent=4)
    with open("hard_train.json", "w") as f:
        json.dump(hard_train, f, indent=4)
    with open("hard_test.json", "w") as f:
        json.dump(hard_test, f, indent=4)

if __name__ == "__main__":
    main()


#python generate_mmlu_prompts.py --data_path /path/to/mmlu/data --dev

