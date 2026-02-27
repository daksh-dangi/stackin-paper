import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import os
import re
from google import genai
from google.genai import types
import random
import json

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
JUDGE_MODEL = "gemini-2.5-flash"

'''
dataset loading related functions
'''
class MathDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def text_collate_fn(batch):
    return {
        "prompts": [item["prompt"] for item in batch],
        "ground_truths": [item["ground_truth"] for item in batch]
    }

def prepare_datasets(json_path, split_ratio=0.9, batch_size=8, seed=42):
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    parsed_data = []
    for _, content in raw_data.items():
        parsed_data.append({
            "prompt": content["question"],
            "ground_truth": content["answer_val"]
        })

    random.seed(seed)
    random.shuffle(parsed_data)
    split_idx = int(len(parsed_data) * split_ratio)

    train_data = parsed_data[:split_idx]
    eval_data_list = parsed_data[split_idx:]
    print(f"Loaded {len(parsed_data)} problems. Train: {len(train_data)}, Eval: {len(eval_data_list)}")

    train_dataset = MathDataset(train_data)
    dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=text_collate_fn
    )

    eval_dataset = {
        "prompts": [item["prompt"] for item in eval_data_list],
        "ground_truths": [item["ground_truth"] for item in eval_data_list]
    }

    return dataloader, eval_dataset


'''
utility functions for FSDP
'''
def setup_distributed():
    dist.init_process_group("nccl")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

def cleanup_distributed():
    dist.destroy_process_group()


'''
utility functions for training loop + rewards
'''
def is_transformer_block(param_name):
    # since Qwen2 groups all attention + MLP layers under 'model.layers', only look for those
    return 'layers.' in param_name

def is_well_formatted_box(text):
    target = "\\boxed{"
    start_idx = text.find(target)
    
    if start_idx == -1:
        return False
        
    content_start = start_idx + len(target)
    brace_count = 1
    current_idx = content_start
    
    while current_idx < len(text):
        char = text[current_idx]
        
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            
        if brace_count == 0:
            return True
        
        current_idx += 1
        
    return False

def format_reward(completions):
    rewards = []
    for comp in completions:
        if is_well_formatted_box(comp):
            rewards.append(0.1)
        else:
            rewards.append(-1.0)
    return rewards

def accuracy_reward(completions, ground_truths):
    rewards = []
    system_instruction = (
        "You are an expert mathematics professor grading advanced exams. "
        "Compare the Student's Answer to the Ground Truth. "
        "The formatting may differ (e.g., different variable names, unsimplified fractions, "
        "or asymptotic equivalents). If the mathematical meaning and value are equivalent, mark it correct. "
        "CRITICAL: You must output EXACTLY and ONLY the number 1.0 if correct, or 0.0 if incorrect. "
        "Do not output any other text, reasoning, or markdown."
    )

    for completion, ground_truth in zip(completions, ground_truths):
        grading_prompt = f"Ground Truth: {ground_truth}\n\nStudent's Answer: {completion}"
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=grading_prompt,
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.0)
            )
            match = re.search(r"(1\.0|0\.0)", response.text)
            if match:
                score = float(match.group(1))
            else:
                score = 0.0 
                
        except Exception as e:
            print(f"API Error during grading: {e}")
            score = 0.0 
            
        rewards.append(score)
        
    return rewards

def combined_reward_fn(completions, ground_truths):
    rewards_format = format_reward(completions, ground_truths)
    rewards_accuracy = accuracy_reward(completions, ground_truths)
    
    total_rewards = []
    for f_rev, a_rev in zip(rewards_format, rewards_accuracy):
        total_rewards.append(f_rev + a_rev)
        
    return total_rewards