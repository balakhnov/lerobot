import subprocess
import sys
from pathlib import Path

def run_distillation(num_steps, input_policy_path):
    command = [
        sys.executable,
        "original_script.py",
        "--num_steps", str(num_steps),
        "--input_policy_path", str(input_policy_path)
    ]
    subprocess.run(command, check=True)

def main():
    steps = [8, 4, 2, 1]
    output_base_dir = Path("outputs/distil")

    for i, num_steps in enumerate(steps):
        print(f"Running distillation with {num_steps} steps")
        
        if i == 0:
            input_policy_path = "outputs/train/example_pusht_diffusion"
        else:
            input_policy_path = output_base_dir / f"example_pusht_diffusion_{steps[i-1]}"

        output_dir = output_base_dir / f"example_pusht_diffusion_{num_steps}"
        output_dir.mkdir(parents=True, exist_ok=True)

        run_distillation(num_steps, input_policy_path)

        print(f"Completed distillation with {num_steps} steps")
        print(f"Output saved to: {output_dir}")
        print("---")

if __name__ == "__main__":
    main()