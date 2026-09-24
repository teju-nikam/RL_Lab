"""
Reinforcement Learning Lab (Course Code: 01AMP403)
DKTE Society's Textile and Engineering Institute, Ichalkaranji
Department of CSE-AIML | Final Year B.Tech. (AI) Semester VII

Master Test Runner / Orchestrator Script
Executes all 10 prescribed lab experiments and generates a summary status report.
"""

import os
import sys
import time
import subprocess

EXPERIMENTS = [
    ("exp1", "bandit.py", "Experiment 1: Multi-Armed Bandit Problem"),
    ("exp2", "mdp_model.py", "Experiment 2: Markov Decision Process (MDP) Modeling"),
    ("exp3", "policy_evaluation.py", "Experiment 3: Dynamic Programming – Policy Evaluation"),
    ("exp4", "policy_iteration.py", "Experiment 4: Policy Iteration"),
    ("exp5", "value_iteration.py", "Experiment 5: Value Iteration"),
    ("exp6", "monte_carlo.py", "Experiment 6: Monte Carlo Prediction and Control"),
    ("exp7", "td_learning.py", "Experiment 7: Temporal Difference (TD) Learning"),
    ("exp8", "sarsa.py", "Experiment 8: SARSA Algorithm"),
    ("exp9", "q_learning.py", "Experiment 9: Q-Learning Algorithm"),
    ("exp10", "dqn_cartpole.py", "Experiment 10: Deep Q-Network (DQN)")
]


def run_experiment(folder, script, title):
    print("\n" + "=" * 80)
    print(f"RUNNING: {title}")
    print(f"Directory: {folder}/ | Script: {script}")
    print("=" * 80)

    script_path = os.path.join(folder, script)
    if not os.path.exists(script_path):
        print(f"[-] Error: Script not found: {script_path}")
        return False, 0.0

    start_time = time.time()
    try:
        # Run python script from repository root
        result = subprocess.run([sys.executable, script],
                                cwd=os.path.abspath(folder),
                                capture_output=False,
                                text=True)
        elapsed = time.time() - start_time
        success = (result.returncode == 0)
        status_str = "SUCCESS" if success else f"FAILED (Exit Code: {result.returncode})"
        print(f"\n[>] Status: {status_str} (Elapsed: {elapsed:.2f}s)")
        return success, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[-] Execution Exception: {e}")
        return False, elapsed


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    print("*" * 80)
    print("D.K.T.E. Society's Textile and Engineering Institute, Ichalkaranji")
    print("Department of CSE-AIML | B.Tech. (AI) Semester VII")
    print("Course: Reinforcement Learning Lab (01AMP403)")
    print("Master Experiment Test Runner")
    print("*" * 80)

    # Allow running specific experiment: python run_all.py 1 or python run_all.py all
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        exp_num = int(sys.argv[1])
        if 1 <= exp_num <= len(EXPERIMENTS):
            folder, script, title = EXPERIMENTS[exp_num - 1]
            run_experiment(folder, script, title)
            return
        else:
            print(f"Invalid experiment number {exp_num}. Choose between 1 and 10.")
            return

    results = []
    total_start = time.time()

    for folder, script, title in EXPERIMENTS:
        success, elapsed = run_experiment(folder, script, title)
        results.append((title, folder, success, elapsed))

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("FINAL LAB EXPERIMENT EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<50} | {'Status':<10} | {'Runtime':<10}")
    print("-" * 80)
    all_passed = True
    for title, folder, success, elapsed in results:
        status_text = "PASSED [OK]" if success else "FAILED [X]"
        if not success:
            all_passed = False
        print(f"{title:<50} | {status_text:<10} | {elapsed:6.2f}s")
    print("-" * 80)
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Overall Result: {'ALL 10 EXPERIMENTS PASSED SUCCESSFULLY!' if all_passed else 'SOME EXPERIMENTS FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
