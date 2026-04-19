import os
import re
import json
import subprocess
from glob import glob
import argparse

def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {command}")
        exit(1)

def main(data_path, overwrite=False, action_steps=8, int_steps=1):
    checkpoint_dir = os.path.join(data_path, "checkpoints")
    output_base = os.path.join(data_path, "eval")
    result_file = os.path.join(output_base, f"aggregated_scores_action={action_steps}_int={int_steps}.json")
    # os.makedirs(output_base, exist_ok=True)

    scores = {}
    std_scores = {}

    total_score = 0.0
    total_std = 0.0
    num_scores = 0

    max_score = float("-inf")
    max_epoch = ""

    checkpoint_files = glob(os.path.join(checkpoint_dir, "*.ckpt"))

    for ckpt_path in checkpoint_files:
        match = re.search(r"epoch=([0-9]+)-test_mean_score=.*\.ckpt", os.path.basename(ckpt_path))
        if not match:
            continue
        epoch = match.group(1)
        epoch_key = f"epoch={epoch}"
        output_dir = os.path.join(output_base, epoch_key)
        # os.makedirs(output_dir, exist_ok=True)

        eval_log_path = os.path.join(output_dir, f"eval_log_action_steps={action_steps}_int_steps={int_steps}.json")

        if overwrite or not os.path.exists(eval_log_path):
            print(f"Running eval.py for {epoch_key}")
            run_command(f"python eval.py --checkpoint \"{ckpt_path}\" --output_dir \"{output_dir}\" --device cuda:0") #  --set_action_steps={action_steps} --set_int_steps={int_steps} --k_p_scale=3")
        else:
            print(f"Using cached eval_log.json for {epoch_key}")

        if not os.path.exists(eval_log_path):
            print(f"Warning: eval_log.json not found for {epoch_key}")
            continue

        with open(eval_log_path, "r") as f:
            log = json.load(f)

        score = log.get("test/mean_score")
        std = log.get("test/std_score")

        if score is not None:
            scores[epoch_key] = score
            total_score += score
            if score > max_score:
                max_score = score
                max_epoch = epoch_key

        if std is not None:
            std_scores[epoch_key] = std
            total_std += std

        if score is not None:
            num_scores += 1

        checkpoint_json = {epoch_key: {"score": score, "std": std}}
        with open(os.path.join(output_dir, f"eval_log_agg_action={action_steps}_int={int_steps}.json"), "w") as f:
            json.dump(checkpoint_json, f, indent=2)

    avg_score = total_score / num_scores if num_scores > 0 else 0.0
    avg_std = total_std / num_scores if num_scores > 0 else 0.0

    final_json = {
        "average": round(avg_score, 6),
        "average_std": round(avg_std, 6),
        "max": {"epoch": max_epoch, "score": max_score},
        "scores": scores,
        "std_scores": std_scores,
    }

    with open(result_file, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"Aggregated scores saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate evaluation scores")
    parser.add_argument("data_path", help="Path to data directory containing checkpoints")
    parser.add_argument("--overwrite", action="store_true", help="Force re-evaluation even if eval_log.json exists")
    parser.add_argument("--action_steps", type=int, default=8, help="Number of action steps")
    parser.add_argument("--int_steps", type=int, default=1, help="Number of integration steps")
    args = parser.parse_args()

    main(args.data_path, overwrite=args.overwrite, action_steps=args.action_steps, int_steps=args.int_steps)