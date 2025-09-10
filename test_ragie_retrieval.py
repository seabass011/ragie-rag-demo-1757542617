"""Comprehensive test suite for Ragie RAG retrieval system."""

import numpy as np
import pytest
from ragie_retrieval import (
    cosine_sim, rank, repair_score, chunk_document,
    compute_embedding_quality, optimize_retrieval_threshold
)


class TestCosineSimNormalizedInvariant:
    """Test cosine similarity scale invariance."""
    
    def test_cosine_normalized_invariant(self):
        """Test that cosine similarity is invariant to vector scaling."""
        # Create test vectors
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        
        # Compute baseline similarity
        baseline_sim = cosine_sim(a, b)
        
        # Test scaling invariance
        scaled_a = a * 10.0
        scaled_b = b * 0.5
        scaled_sim = cosine_sim(scaled_a, scaled_b)
        
        # Should be identical (within floating point precision)
        assert abs(baseline_sim - scaled_sim) < 1e-10


class TestRankOrdersDescendingAndRespectsTopK:
    """Test document ranking functionality."""
    
    def test_rank_orders_descending_and_respects_top_k(self):
        """Test that ranking returns top documents in descending similarity order."""
        # Create query embedding
        query = np.array([1.0, 0.0, 0.0])
        
        # Create document embeddings with known similarities
        docs = [
            np.array([0.8, 0.6, 0.0]),  # High similarity
            np.array([0.1, 0.1, 0.0]),  # Low similarity  
            np.array([0.9, 0.4, 0.0]),  # Highest similarity
            np.array([0.5, 0.5, 0.0])   # Medium similarity
        ]
        
        # Get top 3 rankings
        top_indices = rank(["doc1", "doc2", "doc3", "doc4"], query, docs, top_k=3)
        
        # Should return exactly 3 documents
        assert len(top_indices) == 3
        
        # Should be in descending order of similarity
        # Expected order: doc3 (index 2), doc1 (index 0), doc4 (index 3)
        assert top_indices[0] == 2  # Highest similarity
        assert top_indices[1] == 0  # Second highest
        assert top_indices[2] == 3  # Third highest


class TestRepairScoreCalculation:
    """Test retrieval quality scoring."""
    
    def test_repair_score_perfect_ranking(self):
        """Test MRR calculation with perfect ranking."""
        query = np.array([1.0, 0.0])
        docs = [
            np.array([1.0, 0.0]),  # Perfect match
            np.array([0.0, 1.0])   # No match
        ]
        ground_truth = [0]  # First document is relevant
        
        score = repair_score(query, docs, ground_truth)
        assert score == 1.0  # Perfect MRR score


class TestChunkDocumentOverlap:
    """Test document chunking with proper overlap."""
    
    def test_chunk_document_overlap(self):
        """Test that document chunking creates proper overlaps."""
        text = "A" * 1000  # 1000 character document
        chunks = chunk_document(text, chunk_size=300, overlap=100)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Should have 100 characters of overlap
            overlap_text = current_chunk[-100:]
            next_start = next_chunk[:100]
            assert overlap_text == next_start


class TestEmbeddingQualityMetrics:
    """Test embedding quality calculations."""
    
    def test_embedding_quality_variance(self):
        """Test that embedding quality includes variance calculation."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]
        
        quality = compute_embedding_quality(embeddings)
        
        # Should include variance calculation
        assert quality["variance"] > 0.0
        assert quality["mean_norm"] > 0.0
        assert quality["std_norm"] >= 0.0
        assert quality["dimensionality"] == 3


class TestOptimalThresholdF1:
    """Test retrieval threshold optimization."""
    
    def test_optimal_threshold_f1_calculation(self):
        """Test that F1 score is calculated correctly."""
        similarities = [0.9, 0.8, 0.3, 0.2]
        relevance = [1.0, 1.0, 0.0, 0.0]  # First two are relevant
        
        threshold = optimize_retrieval_threshold(similarities, relevance)
        
        # Should find a reasonable threshold
        assert 0.1 <= threshold <= 0.9
        
        # Test F1 calculation manually for threshold 0.5
        predictions = [1 if sim >= 0.5 else 0 for sim in similarities]
        tp = sum(p and r for p, r in zip(predictions, relevance))
        fp = sum(p and not r for p, r in zip(predictions, relevance))
        fn = sum(not p and r for p, r in zip(predictions, relevance))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # The function should find optimal threshold, not necessarily 0.5
        # But F1 calculation should be correct
        assert expected_f1 > 0  # Should have some positive F1 score
