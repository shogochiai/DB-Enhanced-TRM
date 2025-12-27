# Pre-trained Model

The `trm.onnx` model is a production Tiny Recursive Model trained for semantic similarity matching in software engineering contexts. It was trained on a proprietary dataset of specification-test pairs to identify semantic alignment between requirements and test cases.

## Model Specifications

- **Format**: ONNX (Open Neural Network Exchange)
- **Size**: 11MB
- **Tokenization**: Byte-level (UTF-8)
- **Max Sequence Length**: 512 tokens
- **Input Format**: `[CLS] + spec_text + [SEP] + test_text + [SEP]`
- **Output**: Similarity score (float, 0-1 range)

## Architecture Details

- **Tokenization**: Byte-level tokenization with recursive attention mechanism
- **Parameters**: Approximately 11MB (ONNX format)
- **Training Method**: Contrastive learning on positive (semantically aligned) and negative (misaligned) pairs
- **Training Data**: Proprietary dataset of specification-test pairs across multiple domains (fintech, healthcare, gaming)

## Usage

```python
from code.trm_onnx_wrapper import TRMOnnxWrapper

trm = TRMOnnxWrapper("models/trm.onnx")
score = trm.predict(
    "System SHALL validate FHIR R4 Patient resource",
    "test validates FHIR R4 Patient resource"
)
print(f"Similarity score: {score:.4f}")
```

## Limitations

The model training process is proprietary, but the trained weights are provided to ensure full reproducibility of all experimental results presented in this paper. This does not affect the validity of the main contributions, which focus on the DB-enhanced architecture and adversarial evaluation framework.
