import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def extract_hubert_embeddings(audio_path, layer=-1):
    """
    Extract HuBERT embeddings from an audio file.
    
    Parameters:
    audio_path (str): Path to the audio file
    layer (int): Which transformer layer to extract embeddings from (-1 for last layer)
    
    Returns:
    torch.Tensor: HuBERT embeddings
    """
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Load HuBERT model and feature extractor
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    # Resample if necessary (HuBERT expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Prepare inputs
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # Get hidden states from specified layer
    # -1 means last layer, -2 second to last, etc.
    hidden_states = outputs.hidden_states[layer]
    
    return hidden_states

def get_mean_embeddings(hidden_states):
    """
    Calculate mean embeddings across time dimension.
    
    Parameters:
    hidden_states (torch.Tensor): HuBERT hidden states
    
    Returns:
    torch.Tensor: Mean embeddings
    """
    # Average across time dimension (dim=1)
    mean_embeddings = torch.mean(hidden_states, dim=1)
    return mean_embeddings

# Example usage
if __name__ == "__main__":
    audio_path = "/home/snp2453/slt/speech_retrieval/context.mp3"
    
    # Extract embeddings
    hidden_states = extract_hubert_embeddings(audio_path)
    print(f"Hidden states shape: {hidden_states.shape}")  # [batch_size, sequence_length, hidden_size]
    
    # Get mean embeddings
    mean_embeddings = get_mean_embeddings(hidden_states)
    print(f"Mean embeddings shape: {mean_embeddings.shape}")  # [batch_size, hidden_size]
    
    # Save embeddings if needed
    torch.save(mean_embeddings, "context_hubert_embeddings.pt")