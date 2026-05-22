import subprocess
import re

commands = [
    ("Baseline", "python train.py --model lightgcn --dataset ml-100k --epochs 20"),
    ("IPS", "python train.py --model lightgcn --dataset ml-100k --epochs 20 --causal ips"),
    ("CausE", "python train.py --model lightgcn --dataset ml-100k --epochs 20 --causal cause"),
    ("Causal PG", "python train.py --model lightgcn --dataset ml-100k --epochs 20 --causal pg --pg-estimator dr"),
]

results = []

for name, cmd in commands:
    print(f"Running {name}...")
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        recall_match = re.search(r"Best Recall@20:\s+([0-9.]+)", output)
        ndcg_match = re.search(r"Best NDCG@20:\s+([0-9.]+)", output)
        
        recall = float(recall_match.group(1)) if recall_match else 0.0
        ndcg = float(ndcg_match.group(1)) if ndcg_match else 0.0
        
        results.append((name, recall, ndcg))
    except Exception as e:
        print(f"Error running {name}: {e}")
        results.append((name, 0.0, 0.0))

print("\n--- EXPERIMENT RESULTS ---")
print(f"{'Model':<15} | {'Recall@20':<10} | {'NDCG@20':<10}")
print("-" * 41)
for name, recall, ndcg in results:
    print(f"{name:<15} | {recall:<10.4f} | {ndcg:<10.4f}")
