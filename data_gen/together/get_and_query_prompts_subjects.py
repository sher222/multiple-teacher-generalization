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
                    question = {"prompt": question_str, "wrong_ans": [], "subject": subject, "difficulty": difficulty}
                    for ans in possible_ans:
                        message_str = f"The following multiple choice question has the answer {ans}. Give an explanation of why {ans} is the correct answer. Do not mention the letter {ans} in your explanation.\n{question_str}"
                        response = client.chat.completions.create(
                            model=together_model,
                            messages=[{"role": "user", "content": message_str}],
                        )
                        answer = f"{ans}. {response.choices[0].message.content}"
                        if ans == answer:
                            question["correct_ans"] = answer
                        else:
                            question["wrong_ans"].append(answer)
                    questions.append(question)

    for subject in subjects:
        savepath = f"responses_{subject}.json" if not args.dev else f"responses_{subject}_dev.json"
        with open(savepath, "w") as f:
            json.dump(questions, f, indent=4)

        
    