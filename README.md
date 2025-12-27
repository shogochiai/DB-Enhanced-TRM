# DB-Enhanced Tiny Recursive Models: Robust Lightweight Retrieval under Adversarial and Out-of-Distribution Inputs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the complete implementation, datasets, and experimental results for the paper **"DB-Enhanced Tiny Recursive Models: Robust Lightweight Retrieval under Adversarial and Out-of-Distribution Inputs"**.

## Abstract

Tiny Recursive Models (TRMs) offer remarkable parameter efficiency but remain vulnerable to adversarial and out-of-distribution inputs. This paper introduces a **DB-enhanced TRM architecture** that integrates deterministic lexical database filters (prefilter and postfilter) around a neural TRM core. Through comprehensive adversarial evaluation, we demonstrate that this neuro-symbolic approach reduces adversarial false positives by **58.6%** while maintaining computational efficiency.

## Repository Structure

```
DB-Enhanced-TRM/
├── paper/
│   ├── DB_Enhanced_TRM.md          # Paper in Markdown format
│   └── DB_Enhanced_TRM.pdf         # Paper in PDF format
├── code/
│   ├── trm_onnx_wrapper.py         # TRM ONNX model wrapper
│   ├── db_filters.py               # Prefilter and postfilter implementation
│   ├── generate_improved_adversarial.py  # Dataset generation
│   ├── run_real_trm_evaluation.py  # Evaluation script
│   ├── visualize_improved_results.py     # Visualization script
│   └── statistical_analysis_improved.py  # Statistical analysis
├── data/
│   ├── train_clean_v2.jsonl        # Training data (clean only)
│   ├── val_clean_v2.jsonl          # Validation data (clean)
│   ├── val_adversarial_v2.jsonl    # Validation data (adversarial)
│   └── val_mixed_v2.jsonl          # Validation data (mixed)
├── figures/
│   ├── figure1_improved_adversarial_distribution.png
│   ├── figure2_improved_comparison_bars.png
│   └── figure3_category_wise_fp.png
├── models/
│   └── trm.onnx                    # Production TRM model
├── results/
│   ├── real_trm_evaluation_results.json
│   └── improved_statistical_analysis.txt
├── README.md                       # This file
├── REPRODUCIBILITY.md              # Detailed reproduction instructions
├── requirements.txt                # Python dependencies
└── LICENSE                         # License information
```

## Key Results

| Model                 | Clean Acc | Adv FP | Robustness | ms/pair |
|-----------------------|-----------|--------|------------|---------|
| Vanilla TRM           | 0.733     | 0.683  | 0.317      | 7.3     |
| TRM + DB Prefilter    | 0.733     | 0.417  | 0.583      | 5.1     |
| TRM + DB Postfilter   | 0.667     | 0.283  | 0.717      | 7.5     |
| TRM + DB Pre + Post   | 0.667     | 0.283  | 0.717      | 5.4     |

**Key Findings**:
- **Prefilter** reduces adversarial FP by 39% (effective against low-overlap attacks)
- **Postfilter** reduces adversarial FP by 58.6% (effective against high-overlap semantic attacks)
- **Combined approach** achieves highest robustness (0.717) with improved efficiency
- All improvements are **statistically significant (p < 0.001)** with **large effect sizes (Cohen's h > 0.5)**

## Quick Start

### Prerequisites

```bash
Python 3.11+
pip install -r requirements.txt
```

### Run Experiments

```bash
# Generate adversarial datasets
python code/generate_improved_adversarial.py

# Run evaluation with real TRM ONNX model
python code/run_real_trm_evaluation.py data results models/trm.onnx

# Generate visualizations
python code/visualize_improved_results.py

# Perform statistical analysis
python code/statistical_analysis_improved.py
```

## Reproducibility

For detailed instructions on reproducing all experimental results, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation

If you use this work in your research, please cite:

```bibtex
@article{dbenhancedtrm2024,
  title={DB-Enhanced Tiny Recursive Models: Robust Lightweight Retrieval under Adversarial and Out-of-Distribution Inputs},
  author={Manus AI},
  year={2024},
  journal={arXiv preprint},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The TRM model used in this research was developed as part of an internal software quality assurance project. We thank the development team for providing access to the trained model for research purposes.

## Contact

For questions or issues, please open an issue on this repository or contact the authors.

---

**Note**: The DOI will be updated once the Zenodo integration is complete.
