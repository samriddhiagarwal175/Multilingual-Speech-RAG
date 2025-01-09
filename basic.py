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

df = load_from_disk("/home/snp2453/slt/merged_df")

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct",device="cuda:0")

answers = []
for i in tqdm(range(len(df))):
    question = df[i]['question']
    gt_answer = df[i]['answer']
    text_file_path = df[i]['speech_path'].split("/")[-1].split(".")[0]
    full_path = "/home/snp2453/slt/transcriptions/" + text_file_path + ".txt"
    with open(full_path, 'r') as file:
        content = file.readlines()
    
    messages = [
    {"role": "user", "content": f"You are an expert in precise question answering task. Given a context and a question, you need to give a direct answer, dont be verbose. Just give the answer directly. Context: {content}. Question: {question}. Answer: "},
    ]
    answer = pipe(messages,max_length=1024)
    answer = answer[0]['generated_text'][1]['content']
    answers.append(answer)    
    # print(answer)
    
df = df.add_column("LLama_Answers",answers)

df.push_to_hub("SP2001/SLT_merged_df",private=True)