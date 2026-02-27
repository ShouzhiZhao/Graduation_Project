import argparse
import subprocess
import sys
import os
from pathlib import Path
import paths

def run_simulation(target_movie_id, config_file):
    print(f"Running simulation for target_movie_id={target_movie_id}...")
    env = os.environ.copy()
    env["TARGET_MOVIE_ID"] = str(target_movie_id)
    
    cmd = [
        sys.executable,
        "data_generator.py",
        "--config_file", config_file,
        "--output_file", "dummy_output.json"
    ]
    
    subprocess.run(cmd, cwd=paths.SIMULATION_AGENT_DIR, env=env, check=True)

def run_deepim_train(epochs):
    print(f"Running DeepIM training for {epochs} epochs...")
    cmd = [
        sys.executable,
        "train_genim.py",
        "--epochs", str(epochs)
    ]
    subprocess.run(cmd, cwd=paths.DEEPIM_DIR, check=True)

def run_deepim_optimize(seed_num, iterations):
    print(f"Running DeepIM optimization (seed_num={seed_num}, iterations={iterations})...")
    cmd = [
        sys.executable,
        "optimize_genim.py",
        "--seed_num", str(seed_num),
        "--iterations", str(iterations)
    ]
    subprocess.run(cmd, cwd=paths.DEEPIM_DIR, check=True)

def run_analysis():
    print("Running Analysis...")
    # Assuming analyze.py or similar is the entry point
    # Checking real_data/analyze.py content first might be good, but assuming standard run
    cmd = [
        sys.executable,
        "analyze.py" 
    ]
    # Check if analyze.py exists, otherwise maybe train_predict_global.py
    if (paths.DATA_ANALYZE_DIR / "analyze.py").exists():
         subprocess.run(cmd, cwd=paths.DATA_ANALYZE_DIR, check=True)
    elif (paths.DATA_ANALYZE_DIR / "train_predict_global.py").exists():
         print("analyze.py not found, running train_predict_global.py instead")
         cmd = [sys.executable, "train_predict_global.py"]
         subprocess.run(cmd, cwd=paths.DATA_ANALYZE_DIR, check=True)
    else:
        print("No analysis script found.")

def main():
    parser = argparse.ArgumentParser(description="Graduation Project Unified Runner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run Simulation Agent")
    sim_parser.add_argument("--target_movie_id", type=int, default=0, help="Target movie ID")
    sim_parser.add_argument("--config_file", type=str, default="config/config.yaml", help="Config file path relative to simulation_agent")

    # DeepIM Train command
    train_parser = subparsers.add_parser("train", help="Train DeepIM Model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

    # DeepIM Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run DeepIM Optimization")
    opt_parser.add_argument("--seed_num", type=int, default=10, help="Number of seed nodes")
    opt_parser.add_argument("--iterations", type=int, default=500, help="Number of iterations")

    # Analysis command
    subparsers.add_parser("analyze", help="Run Data Analysis")

    args = parser.parse_args()

    if args.command == "simulate":
        run_simulation(args.target_movie_id, args.config_file)
    elif args.command == "train":
        run_deepim_train(args.epochs)
    elif args.command == "optimize":
        run_deepim_optimize(args.seed_num, args.iterations)
    elif args.command == "analyze":
        run_analysis()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
