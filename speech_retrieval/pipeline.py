import os
import logging
from typing import List, Tuple
import torch
import torchaudio
from slt.speech_retrieval.speech_retrieval.models.hubert_encoder import HubertEncoder
from slt.speech_retrieval.speech_retrieval.models.embedding_normalizer import EmbeddingNormalizer, NormalizerConfig
from slt.speech_retrieval.speech_retrieval.models.chunker import AudioChunker, AudioChunk
from slt.speech_retrieval.speech_retrieval.models.retriever import SpeechRetriever

logger = logging.getLogger(__name__)

class SpeechRetrievalPipeline:
    """End-to-end pipeline for speech-based retrieval"""
    
    def __init__(self,
                 data_dir: str,
                 results_path: str,
                 chunk_duration: int = 60,
                 hubert_model: str = "facebook/hubert-base-ls960"):
        """Initialize pipeline components"""
        self.chunker = AudioChunker(chunk_duration)
        self.hubert_encoder = HubertEncoder(hubert_model)
        
        normalizer_config = NormalizerConfig(
            data_dir=data_dir,
            results_path=results_path,
            gen_subset="retrieval"
        )
        self.normalizer = EmbeddingNormalizer(normalizer_config)
    
    def process_chunks(self, chunks: List[AudioChunk]) -> List[AudioChunk]:
        """Process chunks: extract embeddings and normalize"""
        for chunk in chunks:
            # Extract embeddings
            embedding = self.hubert_encoder.extract_embeddings(
                chunk.waveform,
                chunk.sample_rate
            )
            
            # Normalize embeddings
            normalized_embedding = self.normalizer.normalize_embeddings(embedding)
            
            # Store normalized embedding
            chunk.embedding = normalized_embedding
        
        return chunks
    
    def retrieve(self,
                context_path: str,
                query_path: str,
                metric: str = 'cosine',
                top_k: int = 1) -> List[Tuple[AudioChunk, float]]:
        """Perform end-to-end retrieval"""
        # Split context into chunks
        logger.info("Splitting context audio into chunks...")
        context_chunks = self.chunker.split_audio(context_path)
        
        # Process context chunks
        logger.info("Processing context chunks...")
        context_chunks = self.process_chunks(context_chunks)
        
        # Process query
        logger.info("Processing query...")
        query_chunks = self.chunker.split_audio(query_path)
        query_chunks = self.process_chunks(query_chunks)
        
        if not query_chunks or query_chunks[0].embedding is None:
            raise ValueError("Failed to process query")
        
        # Compute similarities
        logger.info(f"Computing similarities using {metric} distance...")
        similarities = SpeechRetriever.compute_similarities(
            query_chunks[0].embedding,
            [chunk.embedding for chunk in context_chunks],
            metric
        )
        
        # Get top-k chunks
        chunk_similarities = list(zip(context_chunks, similarities))
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_similarities[:top_k]

def save_retrieved_chunks(chunks: List[Tuple[AudioChunk, float]], 
                        output_dir: str):
    """Save retrieved chunks and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (chunk, similarity) in enumerate(chunks):
        # Save audio
        audio_path = os.path.join(output_dir, f"chunk_{i}.wav")
        torchaudio.save(audio_path, chunk.waveform, chunk.sample_rate)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"chunk_{i}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Start time: {chunk.start_time:.2f}s\n")
            f.write(f"End time: {chunk.end_time:.2f}s\n")
            f.write(f"Similarity score: {similarity:.4f}\n")