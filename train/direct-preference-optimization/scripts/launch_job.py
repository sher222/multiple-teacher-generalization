import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=str, default="null")
parser.add_argument('--noFirstEval', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--dataset', type=str, default="mmlu_short")
parser.add_argument('--expNamePrefix', type=str, default="")
parser.add_argument('--lr', type=str, default="5e-7")
parser.add_argument('--beta', type=str, default="0.1")
parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument('--loss', type=str, default="sft")
parser.add_argument('--epochs', type=str, default="1")
parser.add_argument('--evalCount', type=str, default="64")
parser.add_argument('--evalBatchSize', type=str, default="16")
parser.add_argument('--evalEvery', type=int, default=1168)
parser.add_argument('--gradientAccumulation', type=int, default=2)

args = parser.parse_args()

setup_command = "export PATH=/iris/u/sherylh/anaconda3/bin:$PATH; . /iris/u/sherylh/unpaired-dpo/venv/bin/activate; export WANDB_API_KEY=76c13fc00466f6d0a7f023b8b806b6b77b4ecd0f; cd /iris/u/sherylh/multiple-teacher-generalization/train/direct-preference-optimization/"


exp_name = f"{args.expNamePrefix}{args.dataset}_{args.loss}_beta{args.beta}_lr{args.lr}"
run_command = f"python -u train.py model=qwen1.8b datasets=[{args.dataset}] n_epochs={args.epochs} loss={args.loss} lr={args.lr} exp_name={exp_name} gradient_accumulation_steps={args.gradientAccumulation} batch_size={args.batchSize} eval_batch_size={args.evalBatchSize} trainer=BasicTrainer sample_during_eval=true eval_every={args.evalEvery} n_eval_examples={args.evalCount} n_eval_model_samples={args.evalCount}  do_first_eval={not args.noFirstEval} model.archive={args.archive} debug={args.debug} " + (f"loss.beta={args.beta}" if args.loss != "sft" else "")

print(run_command)


os.system(f"{setup_command}; {run_command}")

"""
srun --account=nlp --gres=gpu:1 --partition=jag-hi --constraint=48G python3 /iris/u/sherylh/multiple-teacher-generalization/train/direct-preference-optimization/scripts/launch_job.py --epochs=3 --dataset=mmlu_medium

srun --account=nlp --gres=gpu:1 --partition=jag-hi --constraint=48G python3 /iris/u/sherylh/multiple-teacher-generalization/train/direct-preference-optimization/scripts/launch_job.py --epochs=3 --dataset=mmlu_short --loss=dpo --archive=/iris/u/sherylh/.cache/sherylh/sftbase_mmlu_short_sft_beta0.1_lr5e-7_2024-05-23_02-28-45_203321/step-3504/policy.pt --debug=True
"""