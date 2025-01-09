import torch
from torch.nn.functional import cosine_similarity
from scipy.spatial.distance import euclidean
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class SpeechRetriever:
    """Handles similarity computation and retrieval"""
    
    @staticmethod
    def compute_similarities(query_embedding: torch.Tensor,
                           chunk_embeddings: List[torch.Tensor],
                           metric: str = 'cosine') -> List[float]:
        """Compute similarities between query and chunks"""
        similarities = []
        
        # Get mean embeddings if necessary
        if len(query_embedding.shape) > 2:
            query_embedding = torch.mean(query_embedding, dim=1)
        
        for chunk_emb in chunk_embeddings:
            if chunk_emb is None:
                similarities.append(-float('inf'))
                continue
            
            if len(chunk_emb.shape) > 2:
                chunk_emb = torch.mean(chunk_emb, dim=1)
            
            if metric == 'cosine':
                sim = cosine_similarity(
                    query_embedding.flatten().unsqueeze(0),
                    chunk_emb.flatten().unsqueeze(0)
                ).item()
            elif metric == 'euclidean':
                sim = -euclidean(
                    query_embedding.flatten().numpy(),
                    chunk_emb.flatten().numpy()
                )
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            similarities.append(sim)
        
        return similarities