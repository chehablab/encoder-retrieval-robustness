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
from transformations import get_transformation

def exists(path):
    return os.path.exists(path)

def _is_already_evaluated(checkpoint_file_path, encoder_name, dataset_name, transformation_name):
    if exists(checkpoint_file_path):
        checkpoints = json.load(open(checkpoint_file_path))
        for checkpoint in checkpoints:
            if checkpoint['encoder'] == encoder_name and checkpoint['dataset'] == dataset_name and checkpoint['transformation'] == transformation_name:
                return True
    return False

def _apply_transform(image, transformation):
    img = np.asarray(image)
    img = img / 255.0
    augmented_img = (np.array(transformation([img])) * 255).astype(np.uint8)
    return augmented_img

def evaluate_retrieval(encoder_name: str, 
                       dataset_name: str, 
                       target_dim: int,
                       transformation,
                       transformation_name: str = None,
                       k_list: List[int] = [9],
                       batch_size: int = 64,
                       device: str = "cuda",
                       checkpoint_folder: str = "./checkpoints", 
                       checkpoint_name: str = "results",
                       verbose: bool = True):

    if verbose: print("\nVerifying checkpoints....") 
    if not exists(checkpoint_folder): os.mkdir(checkpoint_folder)
        
    checkpoint_file = os.path.join(checkpoint_folder, checkpoint_name+".json")
    if _is_already_evaluated(checkpoint_file, encoder_name, dataset_name, transformation_name):
        print(f"{encoder_name} already evaluated on {dataset_name}. Skipping evaluation")
        return

    encoder, img_processor = get_encoder(encoder_name, device=device)
    dataset = get_dataset(dataset_name)
    num_samples = len(dataset)

    # if transformation:
    #     if verbose: print(f"\n Augmenting images....")
    #     all_images = []
    #     all_labels = []

    #     for image, label in tqdm(dataset):
    #         aug_img = _apply_transform(image, transformation)
    #         all_images.append(aug_img)
    #         all_labels.append(label)
    #     del dataset
    #     dataset = zip(all_images, all_labels)

    if verbose: print(f"\nExtracting data ...")
    images_paths = []
    labels = []
    for image, label in tqdm(dataset):
        images_paths.append(image)
        labels.append(label)
    labels = np.array(labels)
    
    if verbose: print(f"\nGetting image embeddings....")
    embeddings = []
    encoder.eval()
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_images = images_paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        if transformation:
            batch_images = [_apply_transform(image, transformation) for image in batch_images]
        batch_images = img_processor(batch_images, return_tensors="pt")["pixel_values"].to(device)
        batch_emb = get_features(encoder, batch_images, target_dim, device)
        embeddings.append(batch_emb)
    embeddings = torch.cat(embeddings, dim=0)
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
        'transformation': transformation_name,
        'metrics': {
            'mean_precision' : mean_precision
        }
    }
    
    checkpoint = json.load(open(checkpoint_file)) if exists(checkpoint_file) else []
         
    checkpoint.append(results)
    
    json.dump(checkpoint, open(checkpoint_file, 'w'), ensure_ascii=True, indent=4)

def _test_retreival_pipeline():
    encoder_name = "microsoft/resnet-50"
    dataset_name = "flowers102"
    transformation_obj = {
            "id": "jigsaw",
            "nb_rows_min": 2,
            "nb_rows_max": 5,
            "nb_cols_min": 2,
            "nb_cols_max": 5,
            "max_steps_min": 2,
            "max_steps_max": 5,
            "active": True
        }
    transformation = get_transformation(transformation_obj)
    transformation_name = "jigsaw"
    evaluate_retrieval(encoder_name, dataset_name, 2048, transformation, transformation_name, [5])
