# Reproducibility Guide

This document provides detailed instructions for reproducing all experimental results presented in the paper.

## System Requirements

- **Operating System**: Linux (Ubuntu 22.04 recommended)
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum
- **Disk Space**: 1GB for code, data, and models

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/shogochiai/DB-Enhanced-TRM.git
cd DB-Enhanced-TRM
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `onnxruntime>=1.16.0` - For running the TRM ONNX model
- `numpy>=1.25.0` - For numerical computations
- `scipy>=1.11.0` - For statistical tests
- `matplotlib>=3.7.0` - For visualizations

### Step 3: Verify Installation

```bash
python -c "import onnxruntime; import numpy; import scipy; import matplotlib; print('All dependencies installed successfully')"
```

## Dataset Generation

The adversarial datasets are already included in the `data/` directory. To regenerate them from scratch:

```bash
python code/generate_improved_adversarial.py
```

This will create:
- `data/train_clean_v2.jsonl` (50 pairs)
- `data/val_clean_v2.jsonl` (15 pairs)
- `data/val_adversarial_v2.jsonl` (60 pairs)
- `data/val_mixed_v2.jsonl` (75 pairs)

### Dataset Format

Each line in the JSONL files contains a JSON object with the following structure:

```json
{
  "spec_text": "System SHALL validate FHIR R4 Patient resource schema",
  "test_dsl": "test validates Patient resource schema",
  "parity_score": 0.0,
  "quality": "adversarial",
  "domain": "medical",
  "adversarial_type": "requirement_deletion"
}
```

## Running Experiments

### Full Evaluation Pipeline

To reproduce all experimental results:

```bash
python code/run_real_trm_evaluation.py data results models/trm.onnx
```

This script:
1. Loads the four datasets
2. Builds the vocabulary database from training data
3. Loads the TRM ONNX model
4. Evaluates four configurations:
   - Vanilla TRM
   - TRM + DB Prefilter
   - TRM + DB Postfilter
   - TRM + DB Pre + Post
5. Saves results to `results/real_trm_evaluation_results.json`

**Expected Runtime**: ~5-10 minutes on a modern CPU

### Expected Output

The evaluation script will print a summary table:

```
================================================================================
Summary Table: Main Results (Adversarial Set)
================================================================================
| Model                  | Clean Acc | Adv FP  | Robustness | ms/pair |
|------------------------|-----------|---------|------------|---------|
| Vanilla TRM            |     0.733 |   0.683 |      0.317 |     7.3 |
| TRM + DB Prefilter     |     0.733 |   0.417 |      0.583 |     5.1 |
| TRM + DB Postfilter    |     0.667 |   0.283 |      0.717 |     7.5 |
| TRM + DB Pre + Post    |     0.667 |   0.283 |      0.717 |     5.4 |
================================================================================
```

## Generating Visualizations

To reproduce all figures in the paper:

```bash
python code/visualize_improved_results.py
```

This will generate:
- `figures/figure1_improved_adversarial_distribution.png` - Score distribution histograms
- `figures/figure2_improved_comparison_bars.png` - Performance comparison bar charts
- `figures/figure3_category_wise_fp.png` - Category-wise false positive rates

## Statistical Analysis

To reproduce all statistical tests:

```bash
python code/statistical_analysis_improved.py
```

This will perform:
- **McNemar's tests** comparing each configuration
- **Bootstrap confidence intervals** (10,000 iterations) for robustness scores
- **Effect size analysis** (Cohen's h)

Expected output:

```
McNemar's Test: Vanilla TRM vs TRM + DB Pre + Post
  Vanilla wrong, Pre+Post correct: 24
  Vanilla correct, Pre+Post wrong: 0
  McNemar's statistic: 22.0417
  P-value: 0.000003
  Result: HIGHLY SIGNIFICANT (p < 0.001) ***

Bootstrap 95% Confidence Intervals for Robustness Score
  Vanilla TRM         : 0.317  [0.200, 0.433]
  TRM + DB Pre + Post : 0.717  [0.600, 0.833]
  Interpretation: Confidence intervals DO NOT OVERLAP.
  The improvement is statistically robust.

Effect Size Analysis
  Cohen's h: 0.824 (LARGE)
```

## Understanding the TRM Model

The `trm.onnx` model is a production Tiny Recursive Model with the following specifications:

- **Input 1**: `input_ids` - Token IDs (batch_size, 512), int64
- **Input 2**: `attention_mask` - Attention mask (batch_size, 512), float32
- **Output**: `score` - Similarity score (batch_size,), float32

### Tokenization

The model uses **byte-level tokenization**:
- Each character is converted to its UTF-8 byte value (0-255)
- Special tokens: `[CLS]=1`, `[SEP]=2`, `[PAD]=0`
- Input format: `[CLS] + text1 + [SEP] + text2 + [SEP]`
- Maximum length: 512 tokens (padded or truncated)

### Example

```python
from code.trm_onnx_wrapper import TRMOnnxWrapper

trm = TRMOnnxWrapper("models/trm.onnx")
score = trm.predict(
    "System SHALL validate FHIR R4 Patient resource",
    "test validates FHIR R4 Patient resource"
)
print(f"Similarity score: {score:.4f}")
```

## Troubleshooting

### Issue: ONNX Runtime not found

```bash
pip install onnxruntime --upgrade
```

### Issue: Out of memory

Reduce batch size in `run_real_trm_evaluation.py`:
```python
# Line 45
BATCH_SIZE = 1  # Reduce from default
```

### Issue: Different results

Ensure you're using the same random seed:
```python
random.seed(42)
np.random.seed(42)
```

## Hardware Specifications Used

The experiments in the paper were conducted on:
- **CPU**: Intel Xeon or equivalent
- **RAM**: 16GB
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11.0

## Verification Checklist

To verify your reproduction:

- [ ] All datasets generated successfully
- [ ] TRM model loads without errors
- [ ] Evaluation completes in ~5-10 minutes
- [ ] Adversarial FP rate for Vanilla TRM is ~0.68
- [ ] Adversarial FP rate for Pre+Post is ~0.28
- [ ] All p-values are < 0.001
- [ ] Confidence intervals do not overlap
- [ ] All visualizations are generated

## Contact

If you encounter any issues during reproduction, please:
1. Check this guide carefully
2. Verify your environment matches the requirements
3. Open an issue on GitHub with:
   - Your system specifications
   - Error messages
   - Steps to reproduce the issue

We are committed to ensuring full reproducibility of our results.
