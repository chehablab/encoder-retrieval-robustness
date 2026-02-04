import numpy as np
import random, os, json, gc
from metrics import *
from metrics import AltMetric
from encoders import get_features, get_encoder
from datasets import get_dataset
from utils import stratified_sample

def probe(encoder_name, dataset_name, transformation, transformation_name,
          metrics= [AltMetric.LINEAR_CKA_METRIC, AltMetric.RBF_CKA_METRIC,
                    AltMetric.RANK_METRIC, AltMetric.TOP_K_RECALL_METRIC,
                    AltMetric.VARIANCE_METRIC, AltMetric.INITIAL_ALIGNMENT_CLUSTERS_METRIC,
                    AltMetric.INITIAL_ALIGNMENT_NN_METRIC, AltMetric.INITIAL_ALIGNMENT_CLUSTERS_AUC_METRIC,
                    AltMetric.INITIAL_ALIGNMENT_CLUSTERS_AUC_METRIC_WITH_DIM_REDUCTION],
           image_size= 224, n_augmentations=10, sample_size=500, encoder_target_dim=768,
             random_state=42, ks= None,
             chkpt_path="./chkpt", chkpt_name="checkpoint",  verbose=True):
    
    encoder, processor = get_encoder(encoder_name)
    dataset = get_dataset(dataset_name, 'train', processor=None)

    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)

    # Create checkpoint
    if verbose: print("Checking path ...")
    if not os.path.exists(chkpt_path):
        os.mkdir(chkpt_path)
    
    # Take a stratified random subset
    if verbose: print(f"Sampling {sample_size} images ...")
    sample_data = stratified_sample(dataset, sample_size)

    if verbose: print("Clearing dataset from memory ...")
    del dataset
    gc.collect()

    # Apply transformations on each image in the sample
    if verbose: print("Applying transformations ...")
    all_images = []
    image_ids = []
    image_labels= []
    
    for idx, (image, label) in enumerate(sample_data):
        # Original image
        image = image.resize((image_size, image_size))
        image = np.asarray(image)
        image = image / 255.0
        # Generate augmentations
        augmented_images = transformation([image]*n_augmentations)
        # Back to int8
        image = np.uint8(image * 255)
        augmented_images = [np.uint8(img * 255) for img in augmented_images]
        # Add original image to list
        all_images.append(image)
        image_ids.append(idx)
        image_labels.append(label)
        # Add augmentations to list
        all_images.extend(augmented_images)
        image_ids.extend([idx]*n_augmentations)
        image_labels.extend([label]*n_augmentations)
        del image
        del augmented_images

    if verbose: print("Clearing sample from memory ...")
    del sample_data
    gc.collect()

    # Get the features of each image and augmentations
    if verbose: print("Getting images embeddings ...")
    features = []
    batch_size = 64  
    for i in range(0, len(all_images), batch_size):
        batch_images = all_images[i:i+batch_size]
        batch_processed = processor(batch_images, return_tensors='pt')['pixel_values']
        batch_features = get_features(encoder, batch_processed, encoder_target_dim, "cuda")
        batch_features = batch_features.cpu().numpy()
        features.append(batch_features)
    features = np.vstack(features).astype('float64')

    if verbose: print("Clearing images from memory ...")
    del all_images
    gc.collect()

    if verbose: print("Clearing model from memory ...")
    del encoder
    gc.collect()
    
    # Compute metrics
    if verbose: print("Computing metrics ...")
    
    if AltMetric.TOP_K_RECALL_METRIC in metrics:
        top_k_aug_recall_scores = top_k_augmentations_recall(features, image_ids, n_augmentations, n_augmentations)
    else:
        top_k_aug_recall_scores = []

    if AltMetric.RANK_METRIC in metrics:
        aug_avg_rank_scores, aug_min_rank_scores, aug_max_rank_scores = augmentations_rank(features, image_ids)
    else:
        aug_avg_rank_scores= []
        aug_min_rank_scores= []
        aug_max_rank_scores= []

    if AltMetric.RBF_CKA_METRIC in metrics:
        rbf_cka_score = rbf_cka(features, image_ids, n_augmentations)
    else: 
        rbf_cka_score = None
    
    if AltMetric.LINEAR_CKA_METRIC in metrics:
        linear_cka_score = linear_cka(features, image_ids, n_augmentations)
    else:
        linear_cka_score = None

    if AltMetric.VARIANCE_METRIC in metrics:
        var_metric = variance(features, image_ids)
    else:
        var_metric= []

    if AltMetric.INITIAL_ALIGNMENT_NN_METRIC in metrics:
        initial_alignment_nn_scores = initial_alignment_nn(features, image_ids, image_labels, k=100)
    else:
        initial_alignment_nn_scores = []

    if AltMetric.INITIAL_ALIGNMENT_CLUSTERS_METRIC in metrics:
        initial_alignment_clusters_scores = initial_alignment_clusters(features, image_ids, image_labels, n_clusters=20)
    else:
        initial_alignment_clusters_scores = []

    if AltMetric.INITIAL_ALIGNMENT_CLUSTERS_AUC_METRIC in metrics: 
        initial_alignment_clusters_auc_scores = initial_alignment_clusters_auc(features, image_ids, image_labels, ks)
    else:
        initial_alignment_clusters_auc_scores = {}

    if AltMetric.INITIAL_ALIGNMENT_CLUSTERS_AUC_METRIC_WITH_DIM_REDUCTION in metrics:
        initial_alignment_clusters_auc_with_dim_reduction_scores = initial_alignment_clusters_auc_with_dim_reduction(features, image_ids, image_labels, ks)
    else:
        initial_alignment_clusters_auc_with_dim_reduction_scores = {}

    if verbose: print("Clearing embeddings from memory ...")
    del features
    gc.collect()

    # Store the metrics in checkpoint format
    if verbose: print("Saving to chekpoint ...")
    config = {
        'n_augmentations': n_augmentations,
        'sample_size': sample_size,
        'encoder_target_dim': encoder_target_dim,
        'image_size': image_size,
        'random_state': random_state,
    }
    
    results = {
        'encoder': encoder_name,
        'dataset': dataset_name,
        'transformation': transformation_name,
        'config': config,
        'metrics': {
            'top_k_recall': top_k_aug_recall_scores,
            'average_rank': aug_avg_rank_scores,
            'min_rank': aug_min_rank_scores,
            'max_rank': aug_max_rank_scores,
            'rbf_cka': rbf_cka_score,
            'linear_cka': linear_cka_score,
            "var": var_metric,
            "initial_alignment_nn": initial_alignment_nn_scores,
            "initial_alignment_clusters": initial_alignment_clusters_scores,
            "initial_alignment_clusters_auc": initial_alignment_clusters_auc_scores,
            "initial_alignment_clusters_auc_with_dim_reduction": initial_alignment_clusters_auc_with_dim_reduction_scores
        }
    }

    # Write to checkpoint
    chkpt_file = os.path.join(chkpt_path, f"{chkpt_name}.json")
    if os.path.exists(chkpt_file):
        chkpt = json.load(open(chkpt_file, "r"))
    else:
        chkpt = []
    chkpt.append(results)
    json.dump(chkpt, open(chkpt_file, "w"), ensure_ascii=True, indent=4)