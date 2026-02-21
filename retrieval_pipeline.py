import os
import json
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List
from datasets import get_dataset
from encoders import get_encoder, get_features
from metric import retrieval_mean_precision, RetrievalMetrics, mean_average_precision, mean_average_precision_self
from transformations import get_transformation
from torch.utils.data import DataLoader

def exists(path):
    return os.path.exists(path)

def _collate_fn(batch):
    images, labels = zip(*batch) 
    return list(images), torch.tensor(labels)

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

def _get_embeddings(encoder, dataset, transformation, img_processor, target_dim, device):
    embeddings = []
    all_labels = []
    for batch_images, labels in tqdm(dataset):
        if transformation:
                batch_images =  [_apply_transform(image, transformation).squeeze() for image in batch_images]
        batch_images = img_processor(batch_images, return_tensors="pt")["pixel_values"].to(device)
        batch_emb = get_features(encoder, batch_images, target_dim, device)
        embeddings.append(batch_emb)
        all_labels.append(labels)
    embeddings = torch.cat(embeddings)
    labels = torch.cat(all_labels)
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    return embeddings, labels

def evaluate_retrieval(encoder_name: str, 
                       dataset_name: str, 
                       target_dim: int,
                       transformation,
                       transformation_name,
                       metrics = [RetrievalMetrics.MEAN_AVERAGE_PRECISION],
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
        print(f"{encoder_name} already evaluated on {dataset_name} and {transformation_name}. Skipping evaluation")
        return

    encoder, img_processor = get_encoder(encoder_name, device=device)
    dataset = get_dataset(dataset_name)
    dataset = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=_collate_fn
    )
    
    if verbose: print(f"\nGetting clean image embeddings....")

    clean_emb, clean_labels = _get_embeddings(encoder, dataset, None, img_processor, target_dim, device)

    if verbose: print("\n Evaluating embeddings....")

    mAP_results = []
    if RetrievalMetrics.MEAN_AVERAGE_PRECISION in metrics:
        if verbose: print("\n Computing mAP@k....")

        for k in k_list:
            if transformation:
                if verbose: print(f"\nGetting {transformation_name} transformed image embeddings....")
                augmented_emb, augmented_labels = _get_embeddings(encoder, dataset, transformation, img_processor, target_dim, device)
                result = mean_average_precision(augmented_emb, augmented_labels, clean_emb, clean_labels, k)
                mAP_results.append({f'mAP@{k}': result})
            else:
                result = mean_average_precision_self(clean_emb, clean_labels, k)
                mAP_results.append({f'mAP@{k}': result})

    mean_precision=[]
    if RetrievalMetrics.MEAN_PRECISION in metrics:
        if verbose: print("\n computing MP@k....")

        for k in k_list:
            if transformation:
                if verbose: print(f"\nGetting {transformation_name} transformed image query embeddings....")
                augmented_emb, augmented_labels = _get_embeddings(encoder, dataset, transformation, img_processor, target_dim, device)
                result = retrieval_mean_precision(augmented_emb, augmented_labels, k)
                mean_precision.append({f'MP@{k}': result})
            else:
                result = retrieval_mean_precision(clean_emb, clean_labels, k)
                mean_precision.append({f'MP@{k}': result})                
        
    if verbose: print("\nSaving checkpoint....")

    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'transformation': transformation_name,
        'metrics': {
                'mAP': mAP_results,
                'mean_precision': mean_precision
            }
        }
    
    checkpoint = json.load(open(checkpoint_file)) if exists(checkpoint_file) else []
         
    checkpoint.append(results)
    
    json.dump(checkpoint, open(checkpoint_file, 'w'), ensure_ascii=True, indent=4)

def _test_retreival_pipeline():
    encoder_name = "microsoft/resnet-50"
    dataset_name = "gpr1200"
    transformation_obj = {
            "id": "motionblur",
            "kernelsize": 14,
            "angle": 45,
            "direction": 1
        }
    transformation = get_transformation(transformation_obj)
    transformation_name = transformation_obj['id']
    evaluate_retrieval(encoder_name, dataset_name, 2048, transformation, transformation_name, [RetrievalMetrics.MEAN_AVERAGE_PRECISION],[10], batch_size=256)
_test_retreival_pipeline()