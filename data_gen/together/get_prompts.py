import os
import json
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prompts for MMLU questions")
    parser.add_argument("--length", type=str, default="medium", help="Length of the answer")
    parser.add_argument("--dev", action="store_true", help="Use the dev set")
    args = parser.parse_args()

    messages = []
    data_path = "/juice4/scr4/nlp/music/old_lakh_datasets/kathli/data"
    dirs = ["dev", "val", "test"] if not args.dev else ["dev"]
    difficulty = "high_school"
    subjects = ["biology", "chemistry", "computer_science", "mathematics", "physics"]
    length = "one sentence" if args.length == "medium" else "paragraph (4 or more sentences)"

    for dir in dirs:
        for subject in subjects:
            path = os.path.join(data_path, dir, f"{difficulty}_{subject}_{dir}.csv")
            # read the csv file (don't use the first row as header)
            df = pd.read_csv(path, header=None)
            for i, row in df.iterrows():
                assert len(row) == 6
                question = row[0]
                answer = row[5]
                question_str = f"{question}\nA. {row[1]}\nB. {row[2]}\nC. {row[3]}\nD. {row[4]}"
                message_str = f"The following multiple choice question has the answer {answer}. Give a {length} explanation of why {answer} is the correct answer. Do not mention the letter {answer} in your explanation.\n{question_str}"
                messages.append({"prompt": message_str, "question": question_str, "answer": answer})

    savepath = f"prompts_{args.length}.json" if not args.dev else f"prompts_{args.length}_dev.json"
    with open(savepath, "w") as f:
        json.dump(messages, f, indent=4)
            
    

