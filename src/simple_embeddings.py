"""
Simple embedding simulation for demo purposes.
Uses keyword matching instead of transformer models.
"""

import numpy as np
from typing import List


def simple_embed(text: str) -> np.ndarray:
    """
    Create simple embedding based on keyword presence.
    
    This is a simplified approach for demo purposes that doesn't
    require heavy ML models but demonstrates the concept.
    """
    # Normalize text
    text_lower = text.lower()
    
    # Feature keywords
    keywords = {
        'redis': 0, 'cache': 1, 'caching': 1,
        'mongodb': 2, 'postgres': 3, 'postgresql': 3, 'database': 4,
        'decision': 5, 'decided': 5, 'using': 6,
        'upload': 7, 'file': 7, 'size': 8, 'limit': 8,
        'mb': 9, 'constraint': 10, 'deployed': 11,
        'maybe': 12, 'could': 12, 'looks': 13,
        'interesting': 13, 'good': 14, 'well': 14
    }
    
    # Create 50-dim vector
    vector = np.zeros(50)
    
    # Set keyword features
    for word, idx in keywords.items():
        if word in text_lower:
            vector[idx] = 1.0
    
    # Add some variance based on text length
    vector[-1] = len(text) / 100.0
    
    # Add TF-IDF-like weighting
    words = text_lower.split()
    for i, word in enumerate(words[:20]):  # First 20 words
        vector[20 + (hash(word) % 20)] += 0.5
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector
