import os
import sys
import numpy as np
from pathlib import Path

# Thêm src vào sys.path để import được modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.pipeline import run_full_pipeline

def run_experiment():
    # Khởi tạo danh sách các seed muốn chạy (ở đây là 0 và 1)
    seeds = [0, 1]
    
    results = {}
    
    for seed in seeds:
        print(f"\n\n{'#'*80}")
        print(f"### RUNNING SEED {seed} ###")
        print(f"{'#'*80}\n")
        
        # Chạy toàn bộ pipeline (train + evaluate) với seed tương ứng
        history, acc, f1, pre, rec, cm = run_full_pipeline(seed)
        
        results[seed] = {
            'accuracy': acc,
            'f1': f1,
            'precision': pre,
            'recall': rec
        }
        
        print(f"\nResults for seed {seed}:")
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%")
        print(f"Precision: {pre*100:.2f}%")
        print(f"Recall:    {rec*100:.2f}%")
    
    print("\n\n" + "="*80)
    print("SUMMARY OF NEW SEEDS")
    print("="*80)
    for seed in seeds:
        res = results[seed]
        print(f"Seed {seed:2d} -> Acc: {res['accuracy']*100:.2f}% | F1: {res['f1']*100:.2f}% | Pre: {res['precision']*100:.2f}% | Rec: {res['recall']*100:.2f}%")
        
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    avg_f1 = np.mean([r['f1'] for r in results.values()])
    avg_pre = np.mean([r['precision'] for r in results.values()])
    avg_rec = np.mean([r['recall'] for r in results.values()])
    
    print("-" * 80)
    print(f"AVERAGE OF SEEDS {seeds}:")
    print(f"Accuracy:  {avg_acc*100:.2f}%")
    print(f"F1 Score:  {avg_f1*100:.2f}%")
    print(f"Precision: {avg_pre*100:.2f}%")
    print(f"Recall:    {avg_rec*100:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
