import json
from together import Together
import together
from tqdm import tqdm
import os 
import pandas as pd
import argparse
from multiprocessing import Pool
import time

class TogetherModel:
    api_keys = [ #02 - 06
        '303e07102419a98aae9be912b8e612cf16c0a54babfde4927bf4e86e9a4b76d6',
        '96b81f69214836f2d22785fbcff82e8b48c78797c7683c34242a70739bdb378b',
        'cd073033f34fdfe978e4ef77f47b5ad7d0508fde15dd9b02802c4c7fca4eb126',
        '13649da461df6431e6505013783d8b8e724311ad1996c6b1a9aba890c3095cc4',
        'b8a9a5e93fcaf5d49a5579fde6e9708bee173848a5fa884af50129daff2543b8'
    ]

    def __init__(self, preferred_model, dispreferred_model, api_keys):
        self.preferred_model = preferred_model
        self.dispreferred_model = dispreferred_model
        self.api_keys = api_keys

    def thread_call(self, args):
        model, prompt, api_key, temp, top_k, top_p = args
        client = Together(api_key=api_key)
        print("thread with api_key", api_key)
        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=1.0,
                )
                return {
                    'model': model,
                    'prompt': prompt,
                    'response': response.choices[0].message.content.strip(),
                    'api_key': api_key,
                    'raw_response': response,
                    'temp': temp,
                    'top_k': top_k,
                    'top_p': top_p
                }
            except together.error.RateLimitError:
                print("api", api_key, "rate limit exceeded, retrying after a short pause...")
                time.sleep(2)
            except together.error.InvalidRequestError:
                prompt = " ".join(prompt.split(" ")[:2500])
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None

    def get_responses(self, prompts, model, temp=0.7, top_k=1, top_p=0.9):
        tasks = [(model, prompt, self.api_keys[i % len(self.api_keys)], temp, top_k, top_p)
                 for i, prompt in enumerate(prompts)]

        rows = []
        num_processes = max(min(len(self.api_keys) * 2, len(tasks)), 1)  # Ensure num_processes is at least 1
        with Pool(processes=num_processes) as pool:
            for result in tqdm(pool.imap_unordered(self.thread_call, tasks), total=len(tasks)):
                if result:
                    rows.append(result)
        return rows


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


easy_train = load_json("easy_train_fullmmlu.json")
easy_test = load_json("easy_test_fullmmlu.json")
hard_train = load_json("hard_train_fullmmlu.json")
hard_test = load_json("hard_test_fullmmlu.json")

def generate_data(args):
    data_path = args.data_path
    dirs = ["dev"] if args.dev else ["dev", "val", "train"]
    splits = {
        "easy_train": {"humanities": [], "social_sciences": [], "STEM": []},
        "easy_test": {"humanities": [], "social_sciences": [], "STEM": []},
        "hard_train": {"humanities": [], "social_sciences": [], "STEM": []},
        "hard_test": {"humanities": [], "social_sciences": [], "STEM": []},
    }

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
                        "difficulty": "easy" if "easy" in dir_ else "hard"
                    }
                    category = get_category(subject)
                    if 'easy' in dir_ and 'train' in dir_:
                        splits["easy_train"][category].append(question)
                    elif 'easy' in dir_ and 'test' in dir_:
                        splits["easy_test"][category].append(question)
                    elif 'hard' in dir_ and 'train' in dir_:
                        splits["hard_train"][category].append(question)
                    elif 'hard' in dir_ and 'test' in dir_:
                        splits["hard_test"][category].append(question)
                    print(f"Added question: {question_str[:50]}...")

    return splits

def generate_responses(questions, together_model):
    preferred_prompts = []
    dispreferred_prompts = []
    for question in tqdm(questions, desc="Generating prompts"):
        preferred_prompt = f"You are an expert in {question['subject']}. The following multiple choice question has the correct answer {question['correct_ans']}. Provide a concise explanation of the correct answer {question['correct_ans']}. Start the response by first printing the correct option {question['correct_ans']}, followed by your explanation.\n{question['prompt']}"
        dispreferred_prompt = f"Answer the following multiple choice question by first printing the correct response, followed by a concise explanation. \n{question['prompt']}"
        preferred_prompts.append(preferred_prompt)
        dispreferred_prompts.append(dispreferred_prompt)

    print(f"Total preferred prompts generated: {len(preferred_prompts)}")
    preferred_responses = together_model.get_responses(preferred_prompts, together_model.preferred_model)
    print(f"Total preferred responses received: {len(preferred_responses)}")

    print(f"Total dispreferred prompts generated: {len(dispreferred_prompts)}")
    dispreferred_responses = together_model.get_responses(dispreferred_prompts, together_model.dispreferred_model)
    print(f"Total dispreferred responses received: {len(dispreferred_responses)}")

    formatted_responses = []
    for i in range(len(preferred_responses)):
        formatted_responses.append({
            "prompt": questions[i]['prompt'],
            "correct_ans": questions[i]['correct_ans'],
            "subject": questions[i]['subject'],
            "category": questions[i]['category'],
            "difficulty": questions[i]['difficulty'],
            "preferred_response": preferred_responses[i]['response'],
            "dispreferred_response": dispreferred_responses[i]['response']
        })
    return formatted_responses

def save_responses(splits_responses, base_filepath):
    for split_name, categories in splits_responses.items():
        for category, responses in categories.items():
            filepath = f"{base_filepath}_{split_name}_{category}.json"
            with open(filepath, "w") as f:
                json.dump(responses, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Generate MMLU preferred data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--dev", action="store_true", help="Use the dev set")
    args = parser.parse_args()

    api_keys = [
        '303e07102419a98aae9be912b8e612cf16c0a54babfde4927bf4e86e9a4b76d6',
        '96b81f69214836f2d22785fbcff82e8b48c78797c7683c34242a70739bdb378b',
        'cd073033f34fdfe978e4ef77f47b5ad7d0508fde15dd9b02802c4c7fca4eb126',
        '13649da461df6431e6505013783d8b8e724311ad1996c6b1a9aba890c3095cc4',
        'f1be5bacda4936866f3961cbdbd74bd70a799ea7d9c3ecbfec6a3cc831fe6f64'
    ]

    preferred_model = "meta-llama/Llama-3-70b-chat-hf"
    dispreferred_model = "Qwen/Qwen1.5-1.8B-Chat"

    json_dir = args.data_path  # Directly use the provided data path

    if not os.path.exists(json_dir):
        print(f"Data path {json_dir} does not exist.")
        return

    splits = {
        "easy_train": load_json(os.path.join(json_dir, "easy_train_fullmmlu.json")),
        "easy_test": load_json(os.path.join(json_dir, "easy_test_fullmmlu.json")),
        "hard_train": load_json(os.path.join(json_dir, "hard_train_fullmmlu.json")),
        "hard_test": load_json(os.path.join(json_dir, "hard_test_fullmmlu.json")),
    }

    if not any(any(v) for k, v in splits.items()):
        print("No questions loaded, exiting.")
        return

    together_model = TogetherModel(preferred_model, dispreferred_model, api_keys)
    splits_responses = {split: {category: generate_responses([q for q in questions if q['category'] == category], together_model)
                                for category in categories.keys()}
                        for split, questions in splits.items()}
    save_responses(splits_responses, "completed_evaluated_responses")
    print("Evaluated responses saved successfully.")

if __name__ == "__main__":
    main()



#python subject_prefs.py --data_path /Users/kushalthaman/multiple-teacher-generalization/data_gen/together/data

            # Use imap or imap_unordered for non-blocking pool map operation
            # Wrap the pool.imap or pool.imap_unordered call with tqdm for progress bar
            # Note: Adjust `total` parameter as per your tasks length if necessarys


#python subject_prefs.py --data_path /Users/kushalthaman/multiple-teacher-generalization/data_gen/together --dev
