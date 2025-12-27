"""
Improved Adversarial Dataset Generator
======================================

Generate comprehensive adversarial examples that test both prefilter and postfilter.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

random.seed(42)

# Domain-specific vocabulary
DOMAINS = {
    'fintech': {
        'spec_keywords': ['order', 'trade', 'risk', 'limit', 'notional', 'exposure', 'pre-trade', 'post-trade', 'settlement', 'clearing'],
        'test_markers': ['validates', 'checks', 'enforces', 'verifies', 'ensures'],
        'constraints': ['<= $1M', '>= 100 shares', 'within 5 seconds', 'before market close']
    },
    'medical': {
        'spec_keywords': ['FHIR', 'R4', 'Patient', 'resource', 'schema', 'diagnosis', 'treatment', 'medication', 'allergy', 'vital signs'],
        'test_markers': ['validates', 'checks', 'verifies', 'ensures', 'confirms'],
        'constraints': ['against FHIR R4', 'within 24 hours', 'with 99.9% accuracy', 'per HIPAA']
    },
    'gamedev': {
        'spec_keywords': ['physics', 'engine', 'collision', 'rigid body', 'swept AABB', 'raycast', 'particle', 'shader', 'texture'],
        'test_markers': ['computes', 'renders', 'detects', 'applies', 'simulates'],
        'constraints': ['at 60 FPS', 'within 16ms', 'using swept AABB', 'with spatial hashing']
    },
    'iot': {
        'spec_keywords': ['sensor', 'temperature', 'humidity', 'gateway', 'MQTT', 'telemetry', 'device', 'edge'],
        'test_markers': ['reads', 'monitors', 'transmits', 'processes', 'aggregates'],
        'constraints': ['every 5 seconds', 'with ±0.5°C accuracy', 'over TLS 1.3', 'to cloud endpoint']
    }
}

def generate_clean_pair(domain: str, quality: str) -> Dict:
    """Generate a clean spec-test pair."""
    vocab = DOMAINS[domain]
    
    # Select keywords
    keywords = random.sample(vocab['spec_keywords'], k=min(3, len(vocab['spec_keywords'])))
    marker = random.choice(vocab['test_markers'])
    constraint = random.choice(vocab['constraints']) if quality == 'high' else ''
    
    # Build spec
    spec = f"System SHALL {keywords[0]} {keywords[1]}"
    if len(keywords) > 2:
        spec += f" {keywords[2]}"
    if constraint:
        spec += f" {constraint}"
    
    # Build test (high quality matches well)
    if quality == 'high':
        test = f"test {marker} {keywords[0]} {keywords[1]}"
        if len(keywords) > 2:
            test += f" {keywords[2]}"
        if constraint:
            test += f" {constraint}"
    else:
        # Medium quality: partial match
        test = f"test {marker} {keywords[0]}"
        if random.random() > 0.5 and len(keywords) > 1:
            test += f" {keywords[1]}"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 1.0 if quality == 'high' else 0.7,
        'quality': quality,
        'domain': domain,
        'adversarial_type': 'clean'
    }

def adversarial_domain_mismatch(spec_domain: str, test_domain: str) -> Dict:
    """Category A: Domain mismatch (low lexical overlap)."""
    spec_vocab = DOMAINS[spec_domain]
    test_vocab = DOMAINS[test_domain]
    
    spec_keywords = random.sample(spec_vocab['spec_keywords'], k=3)
    test_keywords = random.sample(test_vocab['spec_keywords'], k=3)
    test_marker = random.choice(test_vocab['test_markers'])
    
    spec = f"System SHALL {spec_keywords[0]} {spec_keywords[1]} {spec_keywords[2]}"
    test = f"test {test_marker} {test_keywords[0]} {test_keywords[1]} {test_keywords[2]}"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': f"{spec_domain}_vs_{test_domain}",
        'adversarial_type': 'domain_mismatch'
    }

def adversarial_complete_replacement(domain: str) -> Dict:
    """Category A: Complete replacement (low overlap)."""
    vocab = DOMAINS[domain]
    
    # Original spec
    keywords = random.sample(vocab['spec_keywords'], k=3)
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]}"
    
    # Completely unrelated test
    unrelated_phrases = [
        "test checks temperature sensor readings in IoT gateway",
        "test validates user authentication with OAuth 2.0",
        "test ensures database transaction rollback on error",
        "test verifies network packet routing through firewall",
        "test confirms file system permissions are correct"
    ]
    test = random.choice(unrelated_phrases)
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'complete_replacement'
    }

def adversarial_random_text(domain: str) -> Dict:
    """Category A: Random natural language (low overlap)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=3)
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]}"
    
    random_texts = [
        "the quick brown fox jumps over the lazy dog",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "all work and no play makes jack a dull boy",
        "to be or not to be that is the question",
        "it was the best of times it was the worst of times"
    ]
    test = random.choice(random_texts)
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'random_text'
    }

def adversarial_requirement_deletion(domain: str) -> Dict:
    """Category B: Requirement deletion (high overlap, semantic invalid)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=4)
    marker = random.choice(vocab['test_markers'])
    constraint = random.choice(vocab['constraints'])
    
    # Spec with all details
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]} {keywords[3]} {constraint}"
    
    # Test missing critical keyword and constraint
    test = f"test {marker} {keywords[0]} {keywords[1]}"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'requirement_deletion'
    }

def adversarial_constraint_weakening(domain: str) -> Dict:
    """Category B: Constraint weakening (high overlap, semantic invalid)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=3)
    marker = random.choice(vocab['test_markers'])
    constraint = random.choice(vocab['constraints'])
    
    # Spec with specific constraint
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]} {constraint}"
    
    # Test without constraint
    test = f"test {marker} {keywords[0]} {keywords[1]} {keywords[2]}"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'constraint_weakening'
    }

def adversarial_negation_injection(domain: str) -> Dict:
    """Category B: Negation injection (high overlap, semantic flip)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=3)
    marker = random.choice(vocab['test_markers'])
    
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]}"
    
    # Inject negation
    test = f"test {marker} system does not {keywords[0]} {keywords[1]} {keywords[2]}"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'negation_injection'
    }

def adversarial_partial_match(domain: str) -> Dict:
    """Category B: Partial match (high overlap, missing details)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=5)
    marker = random.choice(vocab['test_markers'])
    
    # Spec with many details
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]} using {keywords[3]} and {keywords[4]}"
    
    # Test with only high-level match
    test = f"test {marker} {keywords[0]} {keywords[1]}"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'partial_match'
    }

def adversarial_token_shuffling(domain: str) -> Dict:
    """Category C: Token shuffling (high overlap, wrong order)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=4)
    
    spec = f"System SHALL {keywords[0]} {keywords[1]} {keywords[2]} {keywords[3]}"
    
    # Shuffle tokens
    tokens = ['test'] + keywords[::-1] + ['SHALL', 'System']
    test = ' '.join(tokens)
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'token_shuffling'
    }

def adversarial_synonym_substitution(domain: str) -> Dict:
    """Category C: Synonym substitution (boundary case)."""
    vocab = DOMAINS[domain]
    
    keywords = random.sample(vocab['spec_keywords'], k=3)
    
    spec = f"System SHALL enforce {keywords[0]} {keywords[1]} constraints"
    
    # Use synonyms that are similar but not exact
    test = f"test checks system applies {keywords[0]} {keywords[1]} restrictions"
    
    return {
        'spec_text': spec,
        'test_dsl': test,
        'parity_score': 0.0,
        'quality': 'adversarial',
        'domain': domain,
        'adversarial_type': 'synonym_substitution'
    }

def generate_improved_datasets(output_dir: str):
    """Generate improved adversarial datasets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Improved Adversarial Datasets")
    print("=" * 80)
    print()
    
    # Training set: Clean only (50 pairs)
    print("Generating training set (clean only)...")
    train_clean = []
    domains = list(DOMAINS.keys())
    
    for i in range(50):
        domain = domains[i % len(domains)]
        quality = 'high' if i < 25 else 'medium'
        pair = generate_clean_pair(domain, quality)
        train_clean.append(pair)
    
    print(f"  Generated {len(train_clean)} clean training pairs")
    
    # Validation clean (15 pairs)
    print("Generating validation clean set...")
    val_clean = []
    for i in range(15):
        domain = domains[i % len(domains)]
        pair = generate_clean_pair(domain, 'high')
        val_clean.append(pair)
    
    print(f"  Generated {len(val_clean)} clean validation pairs")
    
    # Validation adversarial (60 pairs)
    print("Generating validation adversarial set...")
    val_adversarial = []
    
    # Category A: Low Overlap (20 pairs)
    print("  Category A: Low Lexical Overlap...")
    
    # Domain mismatch (10 pairs)
    for i in range(10):
        spec_domain = domains[i % len(domains)]
        test_domain = domains[(i + 1) % len(domains)]
        pair = adversarial_domain_mismatch(spec_domain, test_domain)
        val_adversarial.append(pair)
    
    # Complete replacement (5 pairs)
    for i in range(5):
        domain = domains[i % len(domains)]
        pair = adversarial_complete_replacement(domain)
        val_adversarial.append(pair)
    
    # Random text (5 pairs)
    for i in range(5):
        domain = domains[i % len(domains)]
        pair = adversarial_random_text(domain)
        val_adversarial.append(pair)
    
    print(f"    Generated 20 low-overlap adversarial pairs")
    
    # Category B: High Overlap, Semantic Invalid (30 pairs)
    print("  Category B: High Overlap, Semantic Invalid...")
    
    # Requirement deletion (10 pairs)
    for i in range(10):
        domain = domains[i % len(domains)]
        pair = adversarial_requirement_deletion(domain)
        val_adversarial.append(pair)
    
    # Constraint weakening (10 pairs)
    for i in range(10):
        domain = domains[i % len(domains)]
        pair = adversarial_constraint_weakening(domain)
        val_adversarial.append(pair)
    
    # Negation injection (5 pairs)
    for i in range(5):
        domain = domains[i % len(domains)]
        pair = adversarial_negation_injection(domain)
        val_adversarial.append(pair)
    
    # Partial match (5 pairs)
    for i in range(5):
        domain = domains[i % len(domains)]
        pair = adversarial_partial_match(domain)
        val_adversarial.append(pair)
    
    print(f"    Generated 30 high-overlap adversarial pairs")
    
    # Category C: Boundary (10 pairs)
    print("  Category C: Boundary Cases...")
    
    # Token shuffling (5 pairs)
    for i in range(5):
        domain = domains[i % len(domains)]
        pair = adversarial_token_shuffling(domain)
        val_adversarial.append(pair)
    
    # Synonym substitution (5 pairs)
    for i in range(5):
        domain = domains[i % len(domains)]
        pair = adversarial_synonym_substitution(domain)
        val_adversarial.append(pair)
    
    print(f"    Generated 10 boundary adversarial pairs")
    print(f"  Total adversarial: {len(val_adversarial)} pairs")
    
    # Validation mixed
    val_mixed = val_clean + val_adversarial
    random.shuffle(val_mixed)
    
    # Save datasets
    print()
    print("Saving datasets...")
    
    with open(output_path / 'train_clean_v2.jsonl', 'w') as f:
        for pair in train_clean:
            f.write(json.dumps(pair) + '\n')
    
    with open(output_path / 'val_clean_v2.jsonl', 'w') as f:
        for pair in val_clean:
            f.write(json.dumps(pair) + '\n')
    
    with open(output_path / 'val_adversarial_v2.jsonl', 'w') as f:
        for pair in val_adversarial:
            f.write(json.dumps(pair) + '\n')
    
    with open(output_path / 'val_mixed_v2.jsonl', 'w') as f:
        for pair in val_mixed:
            f.write(json.dumps(pair) + '\n')
    
    # Save statistics
    stats = {
        'train_clean': len(train_clean),
        'val_clean': len(val_clean),
        'val_adversarial': len(val_adversarial),
        'val_mixed': len(val_mixed),
        'adversarial_breakdown': {
            'category_a_low_overlap': 20,
            'category_b_high_overlap_invalid': 30,
            'category_c_boundary': 10
        }
    }
    
    with open(output_path / 'dataset_stats_v2.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Saved to {output_path}")
    print()
    print("=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    
    # Print sample from each category
    print()
    print("Sample Adversarial Examples:")
    print("-" * 80)
    
    categories = {
        'domain_mismatch': 'Category A: Domain Mismatch',
        'requirement_deletion': 'Category B: Requirement Deletion',
        'token_shuffling': 'Category C: Token Shuffling'
    }
    
    for adv_type, label in categories.items():
        sample = next((p for p in val_adversarial if p['adversarial_type'] == adv_type), None)
        if sample:
            print(f"\n{label}:")
            print(f"  Spec: {sample['spec_text']}")
            print(f"  Test: {sample['test_dsl']}")


if __name__ == "__main__":
    generate_improved_datasets("data_v2")
