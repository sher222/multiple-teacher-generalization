import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from together import Together

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for MMLU questions")
    parser.add_argument("--dev", action="store_true", help="Use the dev set")
    args = parser.parse_args()

    messages = []
    # data path is the path to the mmlu folder in this directory
    data_path = "/juice4/scr4/nlp/music/old_lakh_datasets/kathli/data"
    dirs = ["dev", "val", "test"] if not args.dev else ["dev"]
    difficulties = ["college", "high_school"]
    subjects = ["biology", "chemistry", "computer_science", "mathematics", "physics"]

    possible_ans = ["A", "B", "C", "D"]

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    together_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

    explanation_length = "an explanation"

    questions = []
    for subject in subjects:
        for dir_ in dirs:
            for difficulty in difficulties:
                path = os.path.join(data_path, dir_, f"{difficulty}_{subject}_{dir_}.csv")
                # read the csv file (don't use the first row as header)
                df = pd.read_csv(path, header=None)
                for i, row in tqdm(df.iterrows(), desc=f"{subject} {dir_} {difficulty}"):
                    assert len(row) == 6
                    question = row[0]
                    answer = row[5]
                    question_str = f"{question}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}"
                    message_str = f"The following multiple choice question has the answer {answer}. Give {explanation_length} of why {answer} is the correct answer. Do not mention the letter {answer} in your explanation.\n{question_str}"
                    wrong_message_str = f"Answer the following multiple choice question with first the letter of the answer followed by a period and then {explanation_length} of why that answer is the correct answer. \n{question_str}"
                    question = {"question": question_str, "prompt": message_str, "wrong_prompt": wrong_message_str, "answer": answer, "subject": subject, "difficulty": difficulty}
                    questions.append(question)

    savepath = f"prompts_mmlu.json" if not args.dev else f"prompts_mmlu_dev.json"
    with open(savepath, "w") as f:
        json.dump(questions, f, indent=4)

        
    