"""

srun --account=nlp --gres=gpu:1 --partition=jag-hi --constraint=48G python3 /iris/u/sherylh/multiple-teacher-generalization/train/direct-preference-optimization/scripts/launch_job.py --epochs=3 --dataset=mmlu_medium

srun --account=nlp --gres=gpu:1 --partition=jag-hi --constraint=48G python3 /iris/u/sherylh/multiple-teacher-generalization/train/direct-preference-optimization/scripts/launch_job.py --epochs=1 --expNamePrefix=base_ --dataset=mmlu_long --evalBatchSize=8 --batchSize=4

base paths:

dpo:
batchSize=2
/iris/u/sherylh/.cache/sherylh/base__mmlu_long_sft_beta0.1_lr5e-7_2024-05-23_03-00-29_294802/LATEST/policy.pt
/iris/u/sherylh/.cache/sherylh/base_mmlu_short_sft_beta0.1_lr5e-7_2024-05-23_02-54-43_585820/LATEST/policy.pt
/iris/u/sherylh/.cache/sherylh/base_mmlu_medium_sft_beta0.1_lr5e-7_2024-05-23_02-50-41_675272/LATEST/policy.pt
"""
import os

loss = ["dpo", "ipo"]
datasets = ["mmlu_short", "mmlu_medium", "mmlu_long"]
archive_paths = ["/iris/u/sherylh/.cache/sherylh/base_mmlu_short_sft_beta0.1_lr5e-7_2024-05-23_02-54-43_585820/LATEST/policy.pt", "/iris/u/sherylh/.cache/sherylh/base_mmlu_medium_sft_beta0.1_lr5e-7_2024-05-23_02-50-41_675272/LATEST/policy.pt", "/iris/u/sherylh/.cache/sherylh/base__mmlu_long_sft_beta0.1_lr5e-7_2024-05-23_03-00-29_294802/LATEST/policy.pt"]

for l in loss:
    for i in range(0, 3):
        screen_name = datasets[i].split("_")[1] + "_" + l
        slurm_command = f"srun --account=nlp --gres=gpu:1 --partition=jag-hi --constraint=48G --mem=64GB python3 /iris/u/sherylh/multiple-teacher-generalization/train/direct-preference-optimization/scripts/launch_job.py --epochs=2 --dataset={datasets[i]} --loss={l} --archive={archive_paths[i]} --batchSize=" + ("1" if i == 2 else "2")+ (" --evalBatchSize=8 --gradientAccumulation=1" if i == 2 else "")
        command = f"screen -dmS {screen_name} bash -c '{slurm_command}; exec bash'"
        print(command)
        os.system(command)
