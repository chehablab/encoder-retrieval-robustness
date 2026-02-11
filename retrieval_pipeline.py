import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List
from datasets import get_dataset
from encoders import get_encoder, get_features
from metric import retrieval_mean_precision

def exists(path):
    return os.path.exists(path)

def _is_already_evaluated(checkpoint_file_path, encoder_name, dataset_name):
    if exists(checkpoint_file_path):
        checkpoints = json.load(open(checkpoint_file_path))
        for checkpoint in checkpoints:
            if checkpoint['encoder'] == encoder_name and checkpoint['dataset'] == dataset_name:
                return True
    return False

def evaluate_retrieval(encoder_name: str, 
                       dataset_name: str, 
                       target_dim: int,
                       k_list: List[int] = [9],
                       device: str = "cuda",
                       checkpoint_folder: str = "./checkpoints", 
                       checkpoint_name: str = "results",
                       verbose: bool = True):

    if verbose: print("\nVerifying checkpoints....") 
    if not exists(checkpoint_folder): os.mkdir(checkpoint_folder)
        
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name+".json")
    if _is_already_evaluated(checkpoint_file, encoder_name, dataset_name):
        print(f"{encoder_name} already evaluated on {dataset_name}. Skipping evaluation")
        return

    encoder, img_processor = get_encoder(encoder_name, device=device)
    dataset = get_dataset(dataset_name, None, img_processor)
        
    if verbose: print(f"\nGetting image embeddings....")
    num_samples = len(dataset)
    embeddings = torch.empty((num_samples, target_dim), device=device)
    labels = np.empty(num_samples, dtype=np.int32)

    encoder.eval()
    for i, (image, label) in enumerate(tqdm(dataset)):
        image = image.to(device).unsqueeze(0)
        emb = get_features(encoder, image, target_dim, device)
        embeddings[i] = emb
        labels[i] = label

    embeddings = embeddings.cpu().numpy().astype("float32")
    faiss.normalize_L2(embeddings)
            
    if verbose: print("\nEvaluating embeddings....")
    mean_precision=[]
    for k in k_list:
        mp = retrieval_mean_precision(embeddings, labels, k)
        mean_precision.append({k: mp})
        
    if verbose: print("\nSaving checkpoint....")
    
    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'metrics': {
            'mean_precision' : mean_precision
        }
    }
    
    checkpoint = json.load(open(checkpoint_file)) if exists(checkpoint_file) else []
         
    checkpoint.append(results)
    
    json.dump(checkpoint, open(checkpoint_file, 'w'), ensure_ascii=True, indent=4)
