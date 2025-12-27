"""
DB-Enhanced Filters for TRM
============================

Implements prefilter and postfilter based on LazyCore STPM architecture:
- Prefilter: N-gram Jaccard similarity threshold
- Postfilter: Constraint-based score adjustment
"""

import re
from typing import Set, List, Dict
import json


class VocabularyDatabase:
    """
    Vocabulary database built from training data.
    Stores:
    - Spec keywords (SHALL, MUST, etc.)
    - Test markers (test, validates, etc.)
    - Domain-specific terms
    - Valid pair signatures
    """
    
    def __init__(self):
        self.spec_keywords = set()
        self.test_markers = set()
        self.domain_terms = set()
        self.valid_signatures = set()  # (spec_key_terms, test_key_terms) tuples
        
    def build_from_pairs(self, pairs: List[Dict]):
        """
        Build vocabulary database from training pairs.
        
        Args:
            pairs: List of dicts with 'spec_text', 'test_dsl', 'parity_score'
        """
        for pair in pairs:
            spec = pair['spec_text']
            test = pair['test_dsl']
            score = pair.get('parity_score', 0.0)
            
            # Only learn from high-quality pairs
            if score >= 0.7:
                self._extract_and_add_vocab(spec, test)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_and_add_vocab(self, spec: str, test: str):
        """Extract vocabulary from a high-quality pair."""
        spec_tokens = self._tokenize(spec)
        test_tokens = self._tokenize(test)
        
        # Extract spec keywords
        spec_keywords = {'shall', 'must', 'should', 'will', 'may', 'required', 'mandatory'}
        self.spec_keywords.update(set(spec_tokens) & spec_keywords)
        
        # Extract test markers
        test_markers_set = {'test', 'validates', 'verifies', 'checks', 'ensures', 'confirms', 'assert', 'expect'}
        self.test_markers.update(set(test_tokens) & test_markers_set)
        
        # Extract domain terms (capitalized or technical terms)
        domain_candidates = [t for t in spec_tokens + test_tokens if len(t) > 4]
        self.domain_terms.update(domain_candidates)
        
        # Store signature (key terms from both sides)
        spec_key = tuple(sorted(set(spec_tokens) & (self.spec_keywords | self.domain_terms)))[:5]
        test_key = tuple(sorted(set(test_tokens) & (self.test_markers | self.domain_terms)))[:5]
        
        if spec_key and test_key:
            self.valid_signatures.add((spec_key, test_key))
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'spec_keywords': len(self.spec_keywords),
            'test_markers': len(self.test_markers),
            'domain_terms': len(self.domain_terms),
            'valid_signatures': len(self.valid_signatures)
        }


class DBPrefilter:
    """
    N-gram Jaccard prefilter.
    Rejects pairs with low token overlap before TRM scoring.
    """
    
    def __init__(self, vocab_db: VocabularyDatabase, threshold: float = 0.1):
        self.vocab_db = vocab_db
        self.threshold = threshold
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _extract_ngrams(self, tokens: List[str], n: int = 3) -> Set[str]:
        """Extract n-grams."""
        if len(tokens) < n:
            return set([''.join(tokens)])
        
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = ''.join(tokens[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def check(self, spec_text: str, test_text: str) -> bool:
        """
        Check if pair passes prefilter.
        
        Returns:
            True if pair should be scored by TRM, False if rejected
        """
        spec_tokens = self._tokenize(spec_text)
        test_tokens = self._tokenize(test_text)
        
        # N-gram Jaccard similarity
        spec_ngrams = self._extract_ngrams(spec_tokens, n=3)
        test_ngrams = self._extract_ngrams(test_tokens, n=3)
        jaccard = self._jaccard_similarity(spec_ngrams, test_ngrams)
        
        # Additional check: Must have at least some token overlap
        token_overlap = len(set(spec_tokens) & set(test_tokens))
        min_tokens = min(len(spec_tokens), len(test_tokens))
        token_overlap_ratio = token_overlap / min_tokens if min_tokens > 0 else 0.0
        
        # Pass if either n-gram Jaccard or token overlap exceeds threshold
        return jaccard >= self.threshold or token_overlap_ratio >= self.threshold


class DBPostfilter:
    """
    Constraint-based postfilter.
    Adjusts TRM scores based on structural validity checks.
    """
    
    def __init__(self, vocab_db: VocabularyDatabase):
        self.vocab_db = vocab_db
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into set of words."""
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    def _extract_requirements(self, spec_text: str) -> Set[str]:
        """Extract SHALL/MUST requirements."""
        requirements = set()
        pattern = re.compile(r'(SHALL|MUST)\s+([a-z]+(?:\s+[a-z]+){0,3})', re.IGNORECASE)
        matches = pattern.findall(spec_text)
        
        for modal, phrase in matches:
            phrase_tokens = re.findall(r'\b\w+\b', phrase.lower())
            requirements.update(phrase_tokens)
        
        return requirements
    
    def _extract_constraints(self, text: str) -> Set[str]:
        """Extract numeric and boundary constraints."""
        constraints = set()
        
        # Numeric patterns
        numeric_pattern = re.compile(r'\b\d+(?:\.\d+)?(?:[KMB]|million|billion)?\b', re.IGNORECASE)
        constraints.update(numeric_pattern.findall(text))
        
        # Boundary patterns
        boundary_pattern = re.compile(r'\b(<=|>=|<|>|=|between|within|at least|at most|pre-trade|post-trade)\b', re.IGNORECASE)
        constraints.update(boundary_pattern.findall(text))
        
        return constraints
    
    def adjust_score(self, score: float, spec_text: str, test_text: str) -> float:
        """
        Adjust TRM score based on structural validity.
        
        Args:
            score: Original TRM score
            spec_text: Specification text
            test_text: Test DSL text
            
        Returns:
            adjusted_score: Score after postfilter adjustment
        """
        penalty = 0.0
        
        spec_tokens = self._tokenize(spec_text)
        test_tokens = self._tokenize(test_text)
        
        # Penalty 1: Missing SHALL/MUST requirements in test
        spec_reqs = self._extract_requirements(spec_text)
        if spec_reqs:
            req_coverage = len(spec_reqs & test_tokens) / len(spec_reqs)
            if req_coverage < 0.5:
                penalty += 0.3  # Heavy penalty for missing requirements
        
        # Penalty 2: Missing constraints in test
        spec_constraints = self._extract_constraints(spec_text)
        test_constraints = self._extract_constraints(test_text)
        if spec_constraints:
            constraint_coverage = len(spec_constraints & test_constraints) / len(spec_constraints)
            if constraint_coverage < 0.5:
                penalty += 0.2  # Moderate penalty for missing constraints
        
        # Penalty 3: Lexical overlap too low
        token_overlap = len(spec_tokens & test_tokens)
        min_tokens = min(len(spec_tokens), len(test_tokens))
        overlap_ratio = token_overlap / min_tokens if min_tokens > 0 else 0.0
        if overlap_ratio < 0.2:
            penalty += 0.1  # Light penalty for low overlap
        
        # Apply penalty
        adjusted_score = max(0.0, score - penalty)
        
        return adjusted_score


class DBEnhancedTRM:
    """
    DB-Enhanced TRM with prefilter and postfilter.
    """
    
    def __init__(
        self,
        trm_model,
        vocab_db: VocabularyDatabase,
        enable_prefilter: bool = False,
        enable_postfilter: bool = False,
        prefilter_threshold: float = 0.1
    ):
        self.trm = trm_model
        self.vocab_db = vocab_db
        self.enable_prefilter = enable_prefilter
        self.enable_postfilter = enable_postfilter
        
        if enable_prefilter:
            self.prefilter = DBPrefilter(vocab_db, threshold=prefilter_threshold)
        
        if enable_postfilter:
            self.postfilter = DBPostfilter(vocab_db)
    
    def predict(self, spec_text: str, test_text: str) -> float:
        """
        Predict parity score with DB enhancements.
        
        Returns:
            score: Final score after prefilter and postfilter
        """
        # Step 1: Prefilter
        if self.enable_prefilter:
            if not self.prefilter.check(spec_text, test_text):
                return 0.0  # Rejected by prefilter
        
        # Step 2: TRM scoring
        score = self.trm.predict(spec_text, test_text)
        
        # Step 3: Postfilter
        if self.enable_postfilter:
            score = self.postfilter.adjust_score(score, spec_text, test_text)
        
        return score


if __name__ == "__main__":
    # Test the DB filters
    print("Testing DB Filters")
    print("=" * 80)
    
    # Create sample training data
    training_pairs = [
        {
            'spec_text': 'System SHALL validate FHIR R4 Patient resource schema',
            'test_dsl': 'test validates patient resource against FHIR R4 schema',
            'parity_score': 0.95
        },
        {
            'spec_text': 'Order router SHALL enforce pre-trade risk limits on notional exposure',
            'test_dsl': 'test validates order router enforces risk limits on notional exposure before trade',
            'parity_score': 0.90
        }
    ]
    
    # Build vocabulary database
    vocab_db = VocabularyDatabase()
    vocab_db.build_from_pairs(training_pairs)
    
    print("\nVocabulary Database Stats:")
    print(json.dumps(vocab_db.get_stats(), indent=2))
    
    # Test prefilter
    print("\n" + "=" * 80)
    print("Testing Prefilter")
    print("=" * 80)
    
    prefilter = DBPrefilter(vocab_db, threshold=0.1)
    
    test_cases = [
        ("System SHALL validate FHIR R4 Patient resource schema",
         "test validates patient resource against FHIR R4 schema",
         "High quality pair"),
        ("System SHALL validate FHIR R4 Patient resource schema",
         "test checks temperature sensor readings",
         "Different domain"),
        ("Physics engine SHALL compute rigid body collisions",
         "test engine physics collisions AABB swept body rigid using compute",
         "Lexical corruption")
    ]
    
    for spec, test, desc in test_cases:
        result = prefilter.check(spec, test)
        print(f"\n{desc}:")
        print(f"  Pass: {result}")
    
    print("\n" + "=" * 80)
