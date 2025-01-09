import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import HubertModel, Wav2Vec2Processor
from pathlib import Path

# Load HuBERT model and processor
model_name = "facebook/hubert-base-ls960"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def process_audio(file_path):
    # Load audio file
    audio, sample_rate = sf.read(file_path)
    
    # Resample if necessary (HuBERT expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(torch.tensor(audio)).numpy()
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Normalize audio
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-5)
    
    # Prepare input for the model
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(input_values)
    
    # Extract the last hidden states (embeddings)
    embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
    
    return embeddings

# Process all audio files
input_dir = Path("gg32849/Desktop/Research/SLT_Project/SpeechRAG-main/Audio_chunking/output_chunks")
output_dir = Path("gg32849/Desktop/Research/SLT_Project/SpeechRAG-main/Audio_chunking/embeddings")
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(201):  # Process chunk_0.wav to chunk_200.wav
    input_file = input_dir / f"chunk_{i}.wav"
    output_file = output_dir / f"chunk_{i}_embedding.npy"
    
    if input_file.exists():
        print(f"Processing {input_file}")
        embeddings = process_audio(str(input_file))
        np.save(output_file, embeddings)
        print(f"Saved embeddings to {output_file}")
    else:
        print(f"File not found: {input_file}")

print("Processing complete!")
