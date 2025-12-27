"""
TRM ONNX Wrapper with Byte-Level Tokenization
==============================================

Implements the exact tokenization logic from LazyCore STPM.
"""

import onnxruntime as ort
import numpy as np
from typing import List, Tuple


class TRMOnnxModel:
    """
    Wrapper for the real trm.onnx model with LazyCore-compatible tokenization.
    """
    
    # Special token IDs (from LazyCore STPM)
    PAD_TOKEN_ID = 0  # NUL byte
    CLS_TOKEN_ID = 1  # SOH (start of heading)
    SEP_TOKEN_ID = 2  # STX (start of text)
    MAX_SEQ_LEN = 512
    
    def __init__(self, model_path: str):
        """
        Initialize TRM ONNX model.
        
        Args:
            model_path: Path to trm.onnx file
        """
        self.session = ort.InferenceSession(model_path)
        
        # Verify model inputs/outputs
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        assert len(inputs) == 2, f"Expected 2 inputs, got {len(inputs)}"
        assert inputs[0].name == "input_ids", f"Expected input_ids, got {inputs[0].name}"
        assert inputs[1].name == "attention_mask", f"Expected attention_mask, got {inputs[1].name}"
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        assert outputs[0].name == "score", f"Expected score, got {outputs[0].name}"
    
    def _char_to_token(self, char: str) -> int:
        """
        Convert a character to a token ID (byte value).
        
        Args:
            char: Single character
            
        Returns:
            token_id: Byte value (0-255)
        """
        return ord(char) % 256
    
    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize text to byte sequence.
        
        Args:
            text: Input text
            
        Returns:
            tokens: List of token IDs
        """
        return [self._char_to_token(c) for c in text]
    
    def _prepare_input(self, text1: str, text2: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input for TRM model following LazyCore format:
        [CLS] + text1 + [SEP] + text2 + [SEP] + padding
        
        Args:
            text1: First text (e.g., spec)
            text2: Second text (e.g., test)
            
        Returns:
            input_ids: Token IDs array (1, 512)
            attention_mask: Attention mask array (1, 512)
        """
        # Tokenize
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        # Combine: [CLS] + text1 + [SEP] + text2 + [SEP]
        combined = [self.CLS_TOKEN_ID] + tokens1 + [self.SEP_TOKEN_ID] + tokens2 + [self.SEP_TOKEN_ID]
        
        # Truncate if too long
        if len(combined) > self.MAX_SEQ_LEN:
            combined = combined[:self.MAX_SEQ_LEN]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1.0] * len(combined)
        
        # Pad to MAX_SEQ_LEN
        padding_length = self.MAX_SEQ_LEN - len(combined)
        combined += [self.PAD_TOKEN_ID] * padding_length
        attention_mask += [0.0] * padding_length
        
        # Convert to numpy arrays
        input_ids = np.array([combined], dtype=np.int64)
        attention_mask = np.array([attention_mask], dtype=np.float32)
        
        return input_ids, attention_mask
    
    def predict(self, text1: str, text2: str) -> float:
        """
        Predict parity score for a text pair.
        
        Args:
            text1: First text (e.g., spec)
            text2: Second text (e.g., test)
            
        Returns:
            score: Parity score (0.0 to 1.0)
        """
        # Prepare input
        input_ids, attention_mask = self._prepare_input(text1, text2)
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )
        
        # Extract score
        score = float(outputs[0][0])
        
        # Clip to [0, 1] (in case model outputs outside this range)
        score = max(0.0, min(1.0, score))
        
        return score
    
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict scores for a batch of pairs.
        
        Args:
            pairs: List of (text1, text2) tuples
            
        Returns:
            scores: List of parity scores
        """
        return [self.predict(text1, text2) for text1, text2 in pairs]


if __name__ == "__main__":
    # Test the TRM ONNX wrapper
    print("=" * 80)
    print("Testing TRM ONNX Wrapper")
    print("=" * 80)
    print()
    
    # Load model
    model_path = "/home/ubuntu/paper_experiment/trm.onnx"
    trm = TRMOnnxModel(model_path)
    print("Model loaded successfully!")
    print()
    
    # Test cases
    test_cases = [
        ("System SHALL validate FHIR R4 Patient resource schema",
         "test validates patient resource against FHIR R4 schema",
         "High quality pair"),
        
        ("System SHALL validate FHIR R4 Patient resource schema",
         "test checks temperature sensor readings",
         "Different domain"),
        
        ("Order router SHALL enforce pre-trade risk limits on notional exposure <= $1M",
         "test validates order router enforces risk limits on exposure",
         "Adversarial (missing constraint)"),
        
        ("Physics engine SHALL compute rigid body collisions using swept AABB",
         "test engine physics collisions AABB swept body rigid using compute",
         "Adversarial (lexical corruption)"),
    ]
    
    print("Test Results:")
    print("-" * 80)
    
    for spec, test, description in test_cases:
        score = trm.predict(spec, test)
        print(f"\n{description}:")
        print(f"  Spec: {spec[:60]}...")
        print(f"  Test: {test[:60]}...")
        print(f"  Score: {score:.4f}")
    
    print()
    print("=" * 80)
