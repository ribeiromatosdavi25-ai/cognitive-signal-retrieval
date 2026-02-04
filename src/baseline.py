"""
Baseline semantic-only retrieval for comparison.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from simple_embeddings import simple_embed
from typing import Dict, List, Tuple


class BaselineScorer:
    """
    Standard semantic similarity-only retrieval.
    
    This is the baseline we're comparing against - pure embedding similarity
    without considering cognitive signal strength.
    """
    
    def __init__(self):
        """Initialize baseline scorer."""
        pass
    
    def rank_fragments(
        self, 
        fragments: List[Dict], 
        query: str
    ) -> List[Tuple[Dict, float]]:
        """
        Rank fragments by semantic similarity only.
        
        Args:
            fragments: List of fragment dicts
            query: Query string
            
        Returns:
            List of (fragment, score) tuples, sorted by similarity
        """
        # Encode all texts
        texts = [f['text'] for f in fragments]
        embeddings = [simple_embed(text) for text in texts]
        query_embedding = simple_embed(query)
        
        # Compute similarities
        similarities = [
            cosine_similarity([query_embedding], [emb])[0][0]
            for emb in embeddings
        ]
        
        # Pair and sort
        scored = list(zip(fragments, similarities))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
