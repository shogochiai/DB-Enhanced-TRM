"""
Evaluation with Real TRM ONNX Model
====================================

Run experiments using the actual trm.onnx from LazyCore.
"""

import json
import time
from pathlib import Path
from typing import List, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from trm_onnx_wrapper import TRMOnnxModel
from db_filters import VocabularyDatabase, DBEnhancedTRM


def load_dataset(jsonl_path: str) -> List[Dict]:
    """Load dataset from JSONL file."""
    pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def evaluate_model(model, pairs: List[Dict], threshold: float = 0.5) -> Dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    predictions = []
    ground_truths = []
    latencies = []
    
    for pair in pairs:
        spec = pair['spec_text']
        test = pair['test_dsl']
        gt_score = pair['parity_score']
        
        # Measure latency
        start_time = time.time()
        pred_score = model.predict(spec, test)
        latency_ms = (time.time() - start_time) * 1000
        
        predictions.append(pred_score)
        ground_truths.append(gt_score)
        latencies.append(latency_ms)
    
    # Convert to binary predictions
    pred_binary = [1 if p >= threshold else 0 for p in predictions]
    gt_binary = [1 if g >= threshold else 0 for g in ground_truths]
    
    # Calculate confusion matrix
    tp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 1 and g == 0)
    tn = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 0 and g == 0)
    fn = sum(1 for p, g in zip(pred_binary, gt_binary) if p == 0 and g == 1)
    
    total = len(pairs)
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False positive rate (critical for adversarial evaluation)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Robustness score
    robustness = 1.0 - fpr
    
    # Mean absolute error
    mae = sum(abs(p - g) for p, g in zip(predictions, ground_truths)) / total
    
    # Average latency
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positive_rate': fpr,
        'robustness_score': robustness,
        'mae': mae,
        'avg_latency_ms': avg_latency,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'n_samples': total,
        'predictions': predictions,
        'ground_truths': ground_truths
    }


def run_full_evaluation(data_dir: str, output_dir: str, model_path: str):
    """Run full evaluation on all configurations and datasets."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Real TRM ONNX Model Evaluation")
    print("=" * 80)
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_clean = load_dataset(data_path / 'train_clean_v2.jsonl')
    val_clean = load_dataset(data_path / 'val_clean_v2.jsonl')
    val_adversarial = load_dataset(data_path / 'val_adversarial_v2.jsonl')
    val_mixed = load_dataset(data_path / 'val_mixed_v2.jsonl')
    
    print(f"  Train Clean: {len(train_clean)} pairs")
    print(f"  Val Clean: {len(val_clean)} pairs")
    print(f"  Val Adversarial: {len(val_adversarial)} pairs")
    print(f"  Val Mixed: {len(val_mixed)} pairs")
    print()
    
    # Build vocabulary database from training data
    print("Building vocabulary database from training data...")
    vocab_db = VocabularyDatabase()
    vocab_db.build_from_pairs(train_clean)
    
    stats = vocab_db.get_stats()
    print(f"  Spec keywords: {stats['spec_keywords']}")
    print(f"  Test markers: {stats['test_markers']}")
    print(f"  Domain terms: {stats['domain_terms']}")
    print(f"  Valid signatures: {stats['valid_signatures']}")
    print()
    
    # Load real TRM model
    print(f"Loading real TRM ONNX model from {model_path}...")
    trm = TRMOnnxModel(model_path)
    print("Model loaded successfully!")
    print()
    
    # Create all four configurations
    print("Creating model configurations...")
    configs = {
        'Vanilla TRM': trm,
        'TRM + DB prefilter': DBEnhancedTRM(
            trm, vocab_db,
            enable_prefilter=True,
            enable_postfilter=False,
            prefilter_threshold=0.1
        ),
        'TRM + DB postfilter': DBEnhancedTRM(
            trm, vocab_db,
            enable_prefilter=False,
            enable_postfilter=True
        ),
        'TRM + DB pre + post': DBEnhancedTRM(
            trm, vocab_db,
            enable_prefilter=True,
            enable_postfilter=True,
            prefilter_threshold=0.1
        )
    }
    print(f"  {len(configs)} configurations ready")
    print()
    
    # Evaluation sets
    eval_sets = {
        'val_clean': val_clean,
        'val_adversarial': val_adversarial,
        'val_mixed': val_mixed
    }
    
    # Run evaluation
    print("=" * 80)
    print("Running Evaluation")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for config_name, model in configs.items():
        print(f"Configuration: {config_name}")
        print("-" * 80)
        
        config_results = {}
        
        for eval_name, eval_pairs in eval_sets.items():
            print(f"  Evaluating on {eval_name}...", end=' ', flush=True)
            
            metrics = evaluate_model(model, eval_pairs, threshold=0.5)
            config_results[eval_name] = metrics
            
            print(f"Acc={metrics['accuracy']:.3f}, FP={metrics['false_positive_rate']:.3f}, "
                  f"Robust={metrics['robustness_score']:.3f}, Latency={metrics['avg_latency_ms']:.1f}ms")
        
        all_results[config_name] = config_results
        print()
    
    # Save results
    results_file = output_path / 'real_trm_evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print()
    
    # Generate summary table
    print("=" * 80)
    print("Summary Table: Main Results (Adversarial Set)")
    print("=" * 80)
    print()
    
    print("| Model                  | Clean Acc | Adv FP  | Robustness | ms/pair |")
    print("|------------------------|-----------|---------|------------|---------|")
    
    for config_name in configs.keys():
        clean_metrics = all_results[config_name]['val_clean']
        adv_metrics = all_results[config_name]['val_adversarial']
        
        clean_acc = clean_metrics['accuracy']
        adv_fp = adv_metrics['false_positive_rate']
        robustness = adv_metrics['robustness_score']
        latency = adv_metrics['avg_latency_ms']
        
        # Format config name to fit table
        config_short = config_name.replace('TRM + DB ', '').replace('TRM', 'Vanilla')
        if len(config_short) > 22:
            config_short = config_short[:19] + '...'
        
        print(f"| {config_short:22s} | {clean_acc:9.3f} | {adv_fp:7.3f} | {robustness:10.3f} | {latency:7.1f} |")
    
    print()
    print("=" * 80)
    
    # Additional analysis
    print("\nDetailed Analysis:")
    print("-" * 80)
    
    vanilla_fp = all_results['Vanilla TRM']['val_adversarial']['false_positive_rate']
    prefilter_fp = all_results['TRM + DB prefilter']['val_adversarial']['false_positive_rate']
    
    if vanilla_fp > 0:
        fp_reduction = (vanilla_fp - prefilter_fp) / vanilla_fp * 100
    else:
        fp_reduction = 0
    
    print(f"  Adversarial FP Reduction (Vanilla → Prefilter): {fp_reduction:.1f}%")
    print(f"  Vanilla FP: {vanilla_fp:.3f} → Prefilter FP: {prefilter_fp:.3f}")
    print()
    
    # Clean accuracy trade-off
    vanilla_clean_acc = all_results['Vanilla TRM']['val_clean']['accuracy']
    prefilter_clean_acc = all_results['TRM + DB prefilter']['val_clean']['accuracy']
    
    acc_drop = (vanilla_clean_acc - prefilter_clean_acc) * 100
    
    print(f"  Clean Accuracy Trade-off: {acc_drop:+.1f}%")
    print(f"  Vanilla: {vanilla_clean_acc:.3f} → Prefilter: {prefilter_clean_acc:.3f}")
    print()
    
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results'
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'trm.onnx'
    
    results = run_full_evaluation(data_dir, output_dir, model_path)
