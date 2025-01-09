from dataclasses import dataclass
import torch
import torchaudio
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Class to hold information about an audio chunk"""
    start_time: float
    end_time: float
    waveform: torch.Tensor
    sample_rate: int
    embedding: Optional[torch.Tensor] = None

class AudioChunker:
    """Handles splitting of audio into fixed-length chunks"""
    
    def __init__(self, chunk_duration: int = 60):
        self.chunk_duration = chunk_duration
    
    def split_audio(self, audio_path: str) -> List[AudioChunk]:
        """Split audio file into chunks"""
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Calculate chunk size
        samples_per_chunk = self.chunk_duration * sample_rate
        total_samples = waveform.shape[1]
        
        chunks = []
        for i in range(0, total_samples, samples_per_chunk):
            start_sample = i
            end_sample = min(i + samples_per_chunk, total_samples)
            
            chunk = AudioChunk(
                start_time=start_sample / sample_rate,
                end_time=end_sample / sample_rate,
                waveform=waveform[:, start_sample:end_sample],
                sample_rate=sample_rate
            )
            chunks.append(chunk)
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
