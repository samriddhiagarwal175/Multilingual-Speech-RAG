import torch
import subprocess
from dataclasses import dataclass
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class NormalizerConfig:
    """Configuration for embedding normalization"""
    data_dir: str
    results_path: str
    gen_subset: str
    batch_size: int = 1
    normalize: bool = True

class EmbeddingNormalizer:
    """Handles normalization of embeddings"""
    
    def __init__(self, config: NormalizerConfig):
        self.config = config
        self._setup_directories()
    
    def _setup_directories(self):
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.results_path, exist_ok=True)
    
    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings using the speech normalizer pipeline"""
        # Save embeddings temporarily
        temp_path = os.path.join(self.config.data_dir, "temp_embeddings.pt")
        torch.save(embeddings, temp_path)
        
        try:
            # Run normalization
            self._run_normalization_pipeline()
            
            # Load normalized embeddings
            normalized_path = os.path.join(
                self.config.results_path, 
                "normalized_embeddings.pt"
            )
            normalized_embeddings = torch.load(normalized_path)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return normalized_embeddings
    
    def _run_normalization_pipeline(self):
        """Run the normalization pipeline"""
        cmd = [
            "python", "examples/speech_recognition/new/infer.py",
            "--config-dir", "examples/hubert/config/decode/",
            "--config-name", "infer_viterbi",
            f"task.data={self.config.data_dir}",
            f"task.normalize={str(self.config.normalize).lower()}",
            f"common_eval.results_path={os.path.join(self.config.results_path, 'log')}",
            f"common_eval.path={os.path.join(self.config.data_dir, 'checkpoint_best.pt')}",
            f"dataset.gen_subset={self.config.gen_subset}",
            "+task.labels=[\"unit\"]",
            f"+decoding.results_path={self.config.results_path}",
            "common_eval.post_process=none",
            f"+dataset.batch_size={self.config.batch_size}",
            "common_eval.quiet=True"
        ]
        subprocess.run(cmd, check=True)