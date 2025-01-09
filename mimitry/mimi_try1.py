import torch
import torchaudio
from transformers import AutoFeatureExtractor, MimiModel
import os
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pydub import AudioSegment
import json
import torch.nn.functional as F
import IPython.display as ipd
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
import os
from pydub import AudioSegment
import json
import torch
from transformers import pipeline
import os
from transformers import pipeline
import torch
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import librosa
import openai
from datasets import Audio, load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from openai import OpenAI
from llama_index.core import VectorStoreIndex, Document
import os
from typing import List
from llama_index.core import SimpleDirectoryReader, StorageContext, ServiceContext, VectorStoreIndex
import openai
import textwrap
from transformers import pipeline
import torch
import librosa
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import transformers
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and feature extractor
model = None
feature_extractor = None

def initialize_mimi_model():
    """Initialize MIMI model and feature extractor"""
    global model, feature_extractor
    if model is None or feature_extractor is None:
        model = MimiModel.from_pretrained("kyutai/mimi")
        feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    return model, feature_extractor

@dataclass
class AudioChunkInfo:
    """Class to hold information about an audio chunk"""
    chunk_number: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    duration: float    # in seconds
    file_path: str

@dataclass
class SimilarityResult:
    """Class to hold similarity computation results"""
    chunk_id: str
    similarity_score: float
    chunk_start_time: float
    chunk_end_time: float
    embedding_path: str

def extract_mimi_embeddings(audio_path):
    """Extract MIMI embeddings from an audio file."""
    global model, feature_extractor
    
    # Initialize model if not already done
    if model is None or feature_extractor is None:
        model, feature_extractor = initialize_mimi_model()
    
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, feature_extractor.sampling_rate)
            waveform = resampler(waveform)
        
        # Prepare inputs
        inputs = feature_extractor(
            raw_audio=waveform.squeeze().numpy(),
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt"
        )
        
        # Extract features
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = model.encode(inputs["input_values"])
            embeddings = encoder_outputs.audio_codes.float()
            
            # Convert to fixed-size embedding by taking mean across time dimension
            if len(embeddings.shape) == 3:
                embeddings = torch.mean(embeddings, dim=1)
            elif len(embeddings.shape) == 2:
                embeddings = torch.mean(embeddings, dim=0, keepdim=True)
                
            # Normalize the embeddings
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings
        
    except Exception as e:
        logger.error(f"Error in extract_mimi_embeddings: {str(e)}")
        raise
    
class MP3Splitter:
    """Class to handle MP3 file splitting"""
    
    def __init__(self, chunk_duration: int = 60, output_dir: Optional[str] = None,
                 min_chunk_duration: int = 30):
        self.chunk_duration = chunk_duration * 1000  # Convert to milliseconds
        self.min_chunk_duration = min_chunk_duration * 1000
        self.output_dir = output_dir
    
    def _create_output_dir(self, input_file: str) -> str:
        if self.output_dir is None:
            base_path = Path(input_file).parent
            file_name = Path(input_file).stem
            output_dir = base_path / f"{file_name}_chunks"
        else:
            output_dir = Path(self.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    
    def _get_output_path(self, output_dir: str, chunk_number: int) -> str:
        return str(Path(output_dir) / f"chunk_{chunk_number:03d}.mp3")
    
    def split_audio(self, input_file: str) -> List[AudioChunkInfo]:
        """Split MP3 file into chunks"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        output_dir = self._create_output_dir(input_file)
        logger.info(f"Chunks will be saved in: {output_dir}")
        
        try:
            logger.info("Loading audio file...")
            audio = AudioSegment.from_mp3(input_file)
            
            total_duration = len(audio)
            num_chunks = math.ceil(total_duration / self.chunk_duration)
            
            chunk_infos = []
            
            logger.info("Splitting audio into chunks...")
            for i in tqdm(range(num_chunks), desc="Processing chunks"):
                start_time = i * self.chunk_duration
                end_time = min((i + 1) * self.chunk_duration, total_duration)
                
                if i == num_chunks - 1 and (end_time - start_time) < self.min_chunk_duration:
                    if chunk_infos:
                        prev_chunk = chunk_infos[-1]
                        os.remove(prev_chunk.file_path)
                        
                        extended_chunk = audio[prev_chunk.start_time * 1000:end_time]
                        extended_chunk.export(prev_chunk.file_path, format="mp3")
                        
                        chunk_infos[-1] = AudioChunkInfo(
                            chunk_number=prev_chunk.chunk_number,
                            start_time=prev_chunk.start_time,
                            end_time=end_time / 1000,
                            duration=(end_time - prev_chunk.start_time * 1000) / 1000,
                            file_path=prev_chunk.file_path
                        )
                    break
                
                chunk = audio[start_time:end_time]
                output_path = self._get_output_path(output_dir, i)
                chunk.export(output_path, format="mp3")
                
                chunk_info = AudioChunkInfo(
                    chunk_number=i,
                    start_time=start_time / 1000,
                    end_time=end_time / 1000,
                    duration=(end_time - start_time) / 1000,
                    file_path=output_path
                )
                chunk_infos.append(chunk_info)
            
            logger.info(f"Successfully created {len(chunk_infos)} chunks")
            return chunk_infos
            
        except Exception as e:
            logger.error(f"Error splitting audio: {str(e)}")
            raise

class AudioMetadata:
    """Class to handle audio metadata operations"""
    
    @staticmethod
    def save_chunk_metadata(chunk_infos: List[AudioChunkInfo], output_dir: str):
        """Save metadata for all chunks"""
        metadata_path = os.path.join(output_dir, "chunks_metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write("Chunk Information:\n")
            f.write("-" * 50 + "\n")
            
            for chunk in chunk_infos:
                f.write(f"Chunk {chunk.chunk_number:03d}:\n")
                f.write(f"  Start Time: {chunk.start_time:.2f} seconds\n")
                f.write(f"  End Time: {chunk.end_time:.2f} seconds\n")
                f.write(f"  Duration: {chunk.duration:.2f} seconds\n")
                f.write(f"  File: {os.path.basename(chunk.file_path)}\n")
                f.write("-" * 50 + "\n")
        
        # Also save as JSON for easier processing
        metadata_json = {}
        for chunk in chunk_infos:
            metadata_json[f"chunk_{chunk.chunk_number:03d}"] = {
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "duration": chunk.duration,
                "file_path": chunk.file_path
            }
        
        json_path = os.path.join(output_dir, "chunk_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)

class EmbeddingSimilarityCalculator:
    """Handles similarity computations between embeddings"""
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_embedding(self, embedding_path: Union[str, Path]) -> torch.Tensor:
        """Load embedding from file and process it"""
        try:
            embedding = torch.load(embedding_path, map_location=self.device)
            
            # Convert to float if needed
            if not embedding.is_floating_point():
                embedding = embedding.float()
            
            # Get to 2D tensor shape (1, features)
            if len(embedding.shape) == 3:  # (batch, sequence, features)
                embedding = embedding.mean(dim=1)  # Average over sequence dimension
            if len(embedding.shape) == 2:  # (sequence, features)
                embedding = embedding.mean(dim=0, keepdim=True)  # Average to single vector
            if len(embedding.shape) == 1:  # (features,)
                embedding = embedding.unsqueeze(0)  # Add batch dimension
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading embedding from {embedding_path}: {str(e)}")
            raise

    def process_embedding_for_comparison(self, embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Process embedding to match target dimension"""
        if embedding.shape[1] != target_dim:
            # Use linear interpolation to resize to target dimension
            embedding = F.interpolate(
                embedding.unsqueeze(1),  # Add channel dimension
                size=target_dim,
                mode='linear',
                align_corners=False
            ).squeeze(1)  # Remove channel dimension
        return embedding

    def compute_cosine_similarity(self, 
                                query_embedding: torch.Tensor,
                                chunk_embedding: torch.Tensor) -> float:
        """Compute cosine similarity between query and chunk embeddings"""
        try:
            with torch.no_grad():
                # Ensure both are 2D
                if len(query_embedding.shape) == 1:
                    query_embedding = query_embedding.unsqueeze(0)
                if len(chunk_embedding.shape) == 1:
                    chunk_embedding = chunk_embedding.unsqueeze(0)

                # Get the minimum dimension
                min_dim = min(query_embedding.shape[1], chunk_embedding.shape[1])
                
                # Resize both embeddings to the minimum dimension
                query_embedding = self.process_embedding_for_comparison(query_embedding, min_dim)
                chunk_embedding = self.process_embedding_for_comparison(chunk_embedding, min_dim)
                
                # Normalize embeddings
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
                chunk_embedding = F.normalize(chunk_embedding, p=2, dim=1)
                
                # Compute similarity
                similarity = F.cosine_similarity(query_embedding, chunk_embedding, dim=1)
                
                return similarity.item()
                
        except Exception as e:
            logger.error(f"Error in compute_cosine_similarity: {str(e)}")
            raise
    
    def find_most_similar_chunks(self,
                               query_path: str,
                               top_k: int = 1,
                               metadata_path: Optional[str] = None) -> List[SimilarityResult]:
        """Find the most similar chunks to a query"""
        # Load query embedding
        query_embedding = self.load_embedding(query_path)
        
        # Load metadata if available
        chunk_metadata = {}
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                chunk_metadata = json.load(f)
        
        # Process all chunk embeddings
        results = []
        chunk_paths = sorted(self.embeddings_dir.glob("chunk_*.pt"))
        
        for chunk_path in tqdm(chunk_paths, desc="Processing chunks"):
            try:
                metadata = chunk_metadata.get(chunk_path.stem, {})
                chunk_embedding = self.load_embedding(chunk_path)
                similarity_score = self.compute_cosine_similarity(
                    query_embedding,
                    chunk_embedding
                )
                
                result = SimilarityResult(
                    chunk_id=chunk_path.stem,
                    similarity_score=similarity_score,
                    chunk_start_time=metadata.get('start_time', 0.0),
                    chunk_end_time=metadata.get('end_time', 0.0),
                    embedding_path=str(chunk_path)
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk_path}: {str(e)}")
                continue
        
        # Sort by similarity score and get top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]


class SimilarityVisualizer:
    @staticmethod
    def save_results(results: List[SimilarityResult], output_path: str):
        with open(output_path, 'w') as f:
            f.write("Similarity Results:\n")
            f.write("-" * 50 + "\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Rank {i}:\n")
                f.write(f"  Chunk ID: {result.chunk_id}\n")
                f.write(f"  Similarity Score: {result.similarity_score:.4f}\n")
                f.write(f"  Time Range: {result.chunk_start_time:.2f}s - "
                       f"{result.chunk_end_time:.2f}s\n")
                f.write(f"  Embedding: {result.embedding_path}\n")
                f.write("-" * 50 + "\n")
    
    @staticmethod
    def plot_similarities(results: List[SimilarityResult], output_path: str):
        chunk_ids = [r.chunk_id for r in results]
        scores = [r.similarity_score for r in results]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=range(len(chunk_ids)), y=scores)
        plt.title("Chunk Similarity Scores")
        plt.xlabel("Chunk ID")
        plt.ylabel("Cosine Similarity")
        plt.xticks(range(len(chunk_ids)), chunk_ids, rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def process_audio_files(input_file):
    """Process main audio file and extract embeddings"""
    # input_file = "/home/snp2453/slt/CORAAL-QA/DCB_se1_ag1_f_02_1.wav"
    output_dir = "speech_retrieval/data/processed_audio"
    embeddings_dir = "speech_retrieval/data/embeddings/chunks_embedding"
    chunk_duration = 15
    min_chunk_duration = 5

    try:
        # Initialize MIMI model
        initialize_mimi_model()
        
        # Initialize splitter
        splitter = MP3Splitter(
            chunk_duration=chunk_duration,
            output_dir=output_dir,
            min_chunk_duration=min_chunk_duration
        )
        
        # Split audio
        chunk_infos = splitter.split_audio(input_file)
        
        # Save metadata and create embeddings directory
        if chunk_infos:
            output_dir = os.path.dirname(chunk_infos[0].file_path)
            AudioMetadata.save_chunk_metadata(chunk_infos, output_dir)
            Path(embeddings_dir).mkdir(parents=True, exist_ok=True)
            
            # Process each chunk
            for chunk_info in tqdm(chunk_infos, desc="Extracting embeddings"):
                try:
                    embeddings = extract_mimi_embeddings(chunk_info.file_path)
                    output_path = os.path.join(embeddings_dir, f"chunk_{chunk_info.chunk_number:03d}.pt")
                    torch.save(embeddings, output_path)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_info.chunk_number}: {str(e)}")
                    continue
                    
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

def process_query_audio(query_audio_path: str, output_path: str):
    """Process query audio and save embeddings"""
    try:
        initialize_mimi_model()
        embeddings = extract_mimi_embeddings(query_audio_path)
        torch.save(embeddings, output_path)
        logger.info(f"Query embeddings saved to {output_path}")
    except Exception as e:
        logger.error(f"Error processing query audio: {str(e)}")
        raise

def find_similar_chunks():
    """Find chunks similar to query"""
    output_dir = "speech_retrieval/results"
    embeddings_dir = "speech_retrieval/data/embeddings/chunks_embedding"
    query_path = "peech_retrieval/data/embeddings/question_embedding.pt"
    metadata_path = "speech_retrieval/chunk_metadata.json"
    top_k = 20

    # try:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    calculator = EmbeddingSimilarityCalculator(embeddings_dir)
    results = calculator.find_most_similar_chunks(
        query_path,
        top_k=top_k,
        metadata_path=metadata_path
    )
    
    SimilarityVisualizer.save_results(
        results,
        output_dir / "similarity_results.txt"
    )
    
    SimilarityVisualizer.plot_similarities(
        results,
        output_dir / "similarity_plot.png"
    )
    
    logger.info("Similarity search completed successfully!")
    
    return results
        
    # except Exception as e:
    #     logger.error(f"Error: {str(e)}")
    #     raise


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,)

model_id_llm = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline_llm = transformers.pipeline(
    "text-generation",
    model=model_id_llm,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

answer = []
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts",device="cuda")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

initialize_mimi_model()
current_audio_path = None

df = load_dataset("SP2001/CORAAL-QA_DCB_WhisperLargeTranscribed")

for i in tqdm(range(len(df['train']))):
    question = df['train']['question'][i]
    audio_main_path  = df['train']['audio_path'][i]
    
    if current_audio_path != audio_main_path:
        current_audio_path = audio_main_path
        process_audio_files(audio_main_path)

    
    # response = client.audio.speech.create(
    # model="tts-1",
    # voice="alloy",
    # input=question,
    # )
    
    speech = synthesiser(question, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("speech_retrieval/data/raw_audio/machine_output_question.mp3", speech["audio"], samplerate=speech["sampling_rate"])


    query_embedding_path = "speech_retrieval/data/embeddings/question_embedding.pt"

    # Extract MIMI embeddings for query
    embeddings = extract_mimi_embeddings("speech_retrieval/data/raw_audio/machine_output_question.mp3")
    torch.save(embeddings, query_embedding_path)
    # logger.info(f"Query embeddings saved to {query_embedding_path}")
    # logger.info("Finding similar chunks...")
    res = find_similar_chunks()
    # logger.info("All processing completed successfully!")
    
    audio_paths = [res[i].embedding_path.replace("embeddings", "processed_audio").replace(".pt", ".mp3").replace("chunks_embedding/","") for i in range(len(res))]
    textss = pipe(audio_paths)
    texts = [textss[i]['text'] for i in range(len(textss))]
    texts_all = " ".join(texts)

    system_prompt = """You are an expert in answering questions given a context. Use only the context to answer the asked question and nothing else. Dont be versbose and reply with the exact answer.
    If you cant answer the question based on the context, then answer 'NA'
    """

    user_prompt = f""" {{
    "question":{question},
    "sentence":{texts_all}
    }}"""   

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    outputs = pipeline_llm(
    messages,
    max_new_tokens=128,
    )

    print(outputs[0]["generated_text"][-1])
    answer.append(outputs[0]["generated_text"][-1]['content'])

clean_ret_ans = [answer[i] for i in range(len(answer))]

clean_machine_ans = df['train']['clean_answer'][:len(clean_ret_ans)]

judgements = []
for i in range(len(clean_ret_ans)):
    system_prompt = f""" You are an expert LLM Judge, where given a question and a ground truth answer, you have to judge the machine generated answer and decide if it is same as the ground truth answer or not.
    Be linient in your judgement and judge the answer based on the context and not the exact words. If you think the answer is same as the ground truth answer, then answer 'YES' else answer 'NO'
    Dont be versbose and reply with the exact answer.
    """
    user_prompt = f""" 
    Question: {df['train']['question'][i]}
    Ground Truth Answer: {clean_machine_ans[i]}
    Machine Generated Answer: {clean_ret_ans[i]}
    Judgement:
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    outputs = pipeline_llm(
        messages,
        max_new_tokens=128,
    )
    
    judgements.append(outputs[0]["generated_text"][-1]['content'])
    print(outputs[0]["generated_text"][-1])
    
accuracy = len([i for i in range(len(judgements)) if judgements[i] == 'YES'])/len(judgements)

print(f"Accuracy: {accuracy}")