"""
Cognitive Signal Scoring for Context Retrieval

This module implements a multi-dimensional scoring system that weighs
cognitive signal strength over pure semantic similarity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple


class CognitiveSignalScorer:
    """
    Scores context fragments based on cognitive signal strength.
    
    Combines semantic similarity with signal type, stability, and operability
    to rank context fragments by their true relevance to a query.
    """
    
    # Signal type weights (what kind of information)
    SIGNAL_TYPES = {
        # High signal (0.8-1.0)
        'DECISION': 1.0,      # "We're using X"
        'CONSTRAINT': 0.95,   # "Must not exceed Y"
        'STATE': 0.9,         # "System is in mode Z"
        'ACTION': 0.85,       # "Deployed to prod"
        
        # Medium signal (0.4-0.7)
        'PROPOSAL': 0.6,      # "We could try X"
        'ANALYSIS': 0.5,      # "X has pros/cons"
        'QUESTION': 0.4,      # "Should we use X?"
        
        # Low signal (0.1-0.3)
        'EXPLORATION': 0.3,   # "X looks interesting"
        'COMMENT': 0.2,       # "Cool, makes sense"
        'TANGENT': 0.1        # Off-topic mention
    }
    
    # Stability levels (how confirmed is this info)
    STABILITY_LEVELS = {
        'CONFIRMED': 1.0,     # Explicitly confirmed
        'ACTIVE': 0.8,        # Currently operative
        'PROPOSED': 0.5,      # Suggested but not decided
        'MENTIONED': 0.3,     # Just referenced
        'SUPERSEDED': 0.1     # Overridden by later info
    }
    
    # Operability (how actionable)
    OPERABILITY = {
        'EXECUTABLE': 1.0,    # Can act on this now
        'CONCRETE': 0.7,      # Specific but not actionable
        'ABSTRACT': 0.4,      # Conceptual
        'VAGUE': 0.2          # Unclear/ambiguous
    }
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.semantic_weight = 0.4
        self.signal_weight = 0.6
        self.semantic_threshold = 0.3
    
    def compute_signal_strength(
        self, 
        signal_type: str, 
        stability: str, 
        operability: str
    ) -> float:
        """
        Compute overall signal strength from three dimensions.
        
        Args:
            signal_type: Type of information (DECISION, EXPLORATION, etc.)
            stability: How confirmed (CONFIRMED, PROPOSED, etc.)
            operability: How actionable (EXECUTABLE, ABSTRACT, etc.)
            
        Returns:
            Signal strength score (0.0-1.0)
        """
        type_score = self.SIGNAL_TYPES.get(signal_type, 0.1)
        stability_score = self.STABILITY_LEVELS.get(stability, 0.3)
        operability_score = self.OPERABILITY.get(operability, 0.2)
        
        # Weighted combination
        signal = (
            type_score * 0.5 +
            stability_score * 0.3 +
            operability_score * 0.2
        )
        
        return signal
    
    def score_fragment(
        self, 
        fragment: Dict, 
        query: str,
        query_embedding: np.ndarray = None
    ) -> float:
        """
        Score a single fragment against a query.
        
        Args:
            fragment: Dict with 'text', 'type', 'stability', 'operability'
            query: Query string
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            Combined score (0.0-1.0+)
        """
        # Semantic similarity
        fragment_embedding = self.model.encode([fragment['text']])[0]
        
        if query_embedding is None:
            query_embedding = self.model.encode([query])[0]
        
        semantic = cosine_similarity(
            [fragment_embedding], 
            [query_embedding]
        )[0][0]
        
        # If semantically irrelevant, return 0
        if semantic < self.semantic_threshold:
            return 0.0
        
        # Cognitive signal strength
        signal = self.compute_signal_strength(
            fragment['type'],
            fragment['stability'],
            fragment['operability']
        )
        
        # Combined score
        score = (
            semantic * self.semantic_weight + 
            signal * self.signal_weight
        )
        
        return score
    
    def rank_fragments(
        self, 
        fragments: List[Dict], 
        query: str
    ) -> List[Tuple[Dict, float]]:
        """
        Rank all fragments by relevance to query.
        
        Args:
            fragments: List of fragment dicts
            query: Query string
            
        Returns:
            List of (fragment, score) tuples, sorted by score descending
        """
        # Pre-compute query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Score all fragments
        scored = [
            (frag, self.score_fragment(frag, query, query_embedding))
            for frag in fragments
        ]
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
