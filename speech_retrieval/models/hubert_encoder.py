import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class HubertEncoder:
    """Handles HuBERT embedding extraction"""
    
    def __init__(self, model_name: str = "facebook/hubert-base-ls960"):
        self.model = HubertModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model.eval()
    
    def preprocess_audio(self, waveform: torch.Tensor, 
                        sample_rate: int) -> torch.Tensor:
        """Preprocess audio for HuBERT model"""
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        return waveform
    
    def extract_embeddings(self, waveform: torch.Tensor, 
                          sample_rate: int,
                          layer: int = -1) -> torch.Tensor:
        """Extract HuBERT embeddings from audio"""
        # Preprocess audio
        waveform = self.preprocess_audio(waveform, sample_rate)
        
        # Prepare inputs
        inputs = self.feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer]
        
        return hidden_states