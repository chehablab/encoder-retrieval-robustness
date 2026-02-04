import numpy as np
import faiss
from enum import Enum
import pdb, os
import numpy as np
from scipy.stats import entropy
import gc

debug_pdb_traces = bool(os.environ.get("DEBUG_PDB_TRACES", "False"))

class AltMetric(Enum):
  TOP_K_RECALL_METRIC = "top_k_augmentations_recall"
  RANK_METRIC = "augmentations_rank"
  RBF_CKA_METRIC = "rbf_cka"
  LINEAR_CKA_METRIC = "linear_cka"
  VARIANCE_METRIC = "variance"
  INITIAL_ALIGNMENT_NN_METRIC = "initial_alignment_nn"
  INITIAL_ALIGNMENT_CLUSTERS_METRIC = "initial_alignment_clusters"
  INITIAL_ALIGNMENT_CLUSTERS_AUC_METRIC = "initial_alignment_clusters_auc"
  INITIAL_ALIGNMENT_CLUSTERS_AUC_METRIC_WITH_DIM_REDUCTION = "initial_alignment_clusters_auc_with_dim_reduction"

def _normalized_entropy(labels):
    labels = np.asarray(labels)
    if labels.size == 1:
        return 0.0
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    counts = np.bincount(labels)
    active = counts[counts > 0]
    if len(active) <= 1:
        return 0.0
    H = entropy(active)
    H = max(0.0, H)
    Hn = H / np.log(len(active))
    return max(0.0, min(1.0, Hn))

import numpy as np
import faiss
from scipy.stats import entropy

def _binary_entropy(p):
    if p == 0 or p == 1:
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def initial_alignment_clusters_auc_with_dim_reduction(embeddings, ids, labels, ks):
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)

    uniq_embeddings = []
    for uid in np.unique(ids):
        idx = np.where(ids == uid)[0][0]
        uniq_embeddings.append(embeddings[idx])
    X = np.asarray(uniq_embeddings)
    
    # Dimensionality reduction using PCA
    d = X.shape[1]
    output_dim = 70 if d == 768 else 100 if d == 1024 else 200 if d == 2048 else d // 10
    pca_matrix = faiss.PCAMatrix(d, output_dim)
    pca_matrix.train(X)
    X = pca_matrix.apply(X)

    del embeddings
    del uniq_embeddings
    gc.collect()

    return initial_alignment_clusters_auc(X, ids, labels, ks)

def initial_alignment_clusters_auc(embeddings, ids, labels, ks):
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)

    uniq_embeddings = []
    uniq_labels = []
    for uid in np.unique(ids):
        idx = np.where(ids == uid)[0][0]
        uniq_embeddings.append(embeddings[idx])
        uniq_labels.append(labels[idx])
    X = np.asarray(uniq_embeddings)
    Y = np.asarray(uniq_labels)
    N = len(Y)

    del embeddings
    del uniq_embeddings
    gc.collect()

    multi_label = len(Y.shape) > 1
    num_labels = Y.shape[1] if multi_label else 1

    if multi_label:
        H_list = []
        for l in range(num_labels):
            p = np.mean(Y[:, l])
            H_list.append(_binary_entropy(p))
        H_Y = np.mean(H_list)
    else:
        label_counts = np.bincount(Y.astype(int))
        H_Y = entropy(label_counts)

    if H_Y == 0:
        return {k: 1.0 for k in ks}

    alignments = {}
    for k in ks:
        kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=k,
            niter=20,
            seed=42,
            verbose=False,
        )
        kmeans.train(X)
        cluster_ids = kmeans.index.search(X, 1)[1].flatten()

        H_Y_given_C = 0.0
        for c in range(k):
            idx = np.where(cluster_ids == c)[0]
            if len(idx) == 0:
                continue
            labels_c = Y[idx]

            if multi_label:
                H_list = []
                for l in range(num_labels):
                    p = np.mean(labels_c[:, l])
                    H_list.append(_binary_entropy(p))
                H_c = np.mean(H_list)
            else:
                counts_c = np.bincount(labels_c.astype(int))
                H_c = entropy(counts_c)

            H_Y_given_C += (len(idx) / N) * H_c

        alignment = 1.0 - H_Y_given_C / H_Y
        alignments[k] = max(0.0, min(1.0, alignment))

    del X
    del Y
    gc.collect()

    return alignments

# to do adapt to multi-class
def initial_alignment_clusters(embeddings, ids, labels, n_clusters=100):
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)

    original_embeddings = []
    original_labels = []
    for uid in np.unique(ids):
        idx = np.where(ids == uid)[0][0]
        original_embeddings.append(embeddings[idx].copy())
        original_labels.append(labels[idx].copy())

    original_embeddings = np.asarray(original_embeddings)
    original_labels = np.asarray(original_labels)

    kmeans = faiss.Kmeans(d=original_embeddings.shape[1], k=n_clusters, niter=20, verbose=False)
    kmeans.train(original_embeddings)
    cluster_labels = kmeans.index.search(original_embeddings, 1)[1].flatten()

    # compute per cluster entropy!
    initial_alignments = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        if len(cluster_indices) == 1:
            initial_alignments.append(1.0)
            continue
        cluster_labels_list = original_labels[cluster_indices]
        if len(cluster_labels_list.shape)>1: # multi-label case
            # to do adapt to multi-class
            cluster_labels_list = [l.item() for label in cluster_labels_list for l in label]
        entropy = _normalized_entropy(cluster_labels_list)
        initial_alignments.append(1 - entropy)  # higher is better
    return initial_alignments

# to do adapt to multi-class
def initial_alignment_nn(embeddings, ids, labels, k=100):
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)

    original_embeddings = []
    original_labels = []
    for uid in np.unique(ids):
        idx = np.where(ids == uid)[0][0]
        original_embeddings.append(embeddings[idx].copy())
        original_labels.append(labels[idx].copy())

    original_embeddings = np.asarray(original_embeddings)
    original_labels = np.asarray(original_labels)

    index = faiss.IndexFlatL2(original_embeddings.shape[1])
    index.add(original_embeddings)
    _, neighbors = index.search(original_embeddings, k)

    # compute acc for each embedding
    initial_alignments = []
    for i in range(len(original_embeddings)):
        neighbor_labels = original_labels[neighbors[i][1:]]  # exclude self
        initial_alignments.append(np.sum(neighbor_labels == original_labels[i])/(k-1))

    return initial_alignments

def variance(embeddings, ids):
    ids = np.asarray(ids)
    embeddings = np.asarray(embeddings)

    original_embeddings = []
    for uid in np.unique(ids):
        idx = np.where(ids == uid)[0][0]
        original_embeddings.append(embeddings[idx].copy())

    original_embeddings = np.asarray(original_embeddings)
    var = np.var(original_embeddings, axis=0)
    return {
        "variance": list(var),
        "erank": _variance_erank(var),
        "entropy": _variance_entropy(var),
        "gini": _variance_gini(var),
        "top10_ratio": _variance_top10_ratio(var),
        "tail_mass": _variance_tail_mass(var),
        "max_median_ratio": _variance_max_median_ratio(var),
        "spectral_flatness": _variance_spectral_flatness(var),
    }

def _variance_erank(var):
    var = np.asarray(var)
    var = var[var > 0]
    p = var / var.sum()
    return 1.0 / np.sum(p ** 2)

def _variance_entropy(var):
    var = np.asarray(var)
    var = var[var > 0]
    p = var / var.sum()
    return -np.sum(p * np.log(p)) / np.log(len(p))

def _variance_gini(var):
    var = np.asarray(var)
    var = var[var > 0]
    d = len(var)
    diff_sum = np.sum(np.abs(var[:, None] - var[None, :]))
    return diff_sum / (2 * d * var.sum())

def _variance_top10_ratio(var):
    var = np.asarray(var)
    var = var[var > 0]
    k = max(1, int(0.1 * len(var)))
    return np.sort(var)[-k:].sum() / var.sum()

def _variance_tail_mass(var):
    var = np.asarray(var)
    var = var[var > 0]
    p = var / var.sum()
    return p[p < (1 / len(p))].sum()

def _variance_max_median_ratio(var):
    var = np.asarray(var)
    var = var[var > 0]
    return var.max() / np.median(var)

def _variance_spectral_flatness(var):
    var = np.asarray(var)
    var = var[var > 0]
    return np.exp(np.mean(np.log(var))) / np.mean(var)


def rbf_cka(embeddings, ids, n):
    ids = np.array(ids)
    embeddings = np.array(embeddings)
    original_embeddings = []
    augmented_embeddings= []
    uids = set(ids)
    for id in uids:
       id_original_index = np.where(ids==id)[0][0]
       id_augmented_indices = np.where(ids==id)[0][1:]

       id_original_embeddings = [embeddings[id_original_index].copy()]*n
       id_augmented_embeddings = embeddings[id_augmented_indices]

       assert len(id_original_embeddings) == len(id_augmented_embeddings)

       original_embeddings.extend(id_original_embeddings)
       augmented_embeddings.extend(id_augmented_embeddings)
    original_embeddings = np.array(original_embeddings)
    augmented_embeddings = np.array(augmented_embeddings)
    return float(_rbf_cka(original_embeddings, augmented_embeddings, True))

def linear_cka(embeddings, ids, n):
    ids = np.array(ids)
    embeddings = np.array(embeddings)
    original_embeddings = []
    augmented_embeddings= []
    uids = set(ids)
    for id in uids:
       id_original_index = np.where(ids==id)[0][0]
       id_augmented_indices = np.where(ids==id)[0][1:]

       id_original_embeddings = [embeddings[id_original_index].copy()]*n
       id_augmented_embeddings = embeddings[id_augmented_indices]

       assert len(id_original_embeddings) == len(id_augmented_embeddings)

       original_embeddings.extend(id_original_embeddings)
       augmented_embeddings.extend(id_augmented_embeddings)
    original_embeddings = np.array(original_embeddings)
    augmented_embeddings = np.array(augmented_embeddings)
    return float(_linear_cka(original_embeddings, augmented_embeddings, True))


def top_k_augmentations_recall(embeddings, ids, k, n):
    """
    ids: list of original and augmented images ids .e.g. 111111222222...
    """
    ids = np.array(ids)
    embeddings = np.array(embeddings)
    # Generate index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    recalls = []
    uids = set(ids)
    for id in uids:
        original_index = np.where(ids==id)[0][0] # The first is always the original
        _, neighbors = index.search(embeddings[original_index:original_index+1], k + 1)  # +1 for self
        neighbors = neighbors[0][1:]  # skip self
        hits = sum(ids[n] == id for n in neighbors)
        recalls.append((hits / n).item())
    return list(recalls)

def augmentations_rank(embeddings, ids):
    ids = np.array(ids)
    embeddings = np.array(embeddings)
    # Generate index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    avg_ranks, min_ranks, max_ranks = [], [], []
    uids = set(ids)
    for id in uids:
        original_idx= np.where(ids==id)[0][0]
        transformed_idx = np.where(ids==id)[0][1:]
        _, neighbors = index.search(embeddings[original_idx:original_idx + 1], len(embeddings))
        neighbors = neighbors[0][1:] # skip self
        ranks = [np.where(neighbors == t)[0][0].item() + 1 for t in transformed_idx] # +1 to rank from 1
        avg_ranks.append(np.mean(ranks).item())
        min_ranks.append(np.min(ranks).item())
        max_ranks.append(np.max(ranks).item())

    return avg_ranks, min_ranks, max_ranks

def _gram_linear(x):
  return x.dot(x.T)

def _gram_rbf(x, threshold=1.0):
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def _center_gram(gram, unbiased=False):
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()
  if unbiased:
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram

def _cka(gram_x, gram_y, debiased=False):
  gram_x = _center_gram(gram_x, unbiased=debiased)
  gram_y = _center_gram(gram_y, unbiased=debiased)
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def _rbf_cka(x, y, debiased=True):
   gram_x = _gram_rbf(x)
   gram_y = _gram_rbf(y)
   return _cka(gram_x, gram_y, debiased)

def _linear_cka(x, y, debiased=True):
    gram_x = _gram_linear(x)
    gram_y = _gram_linear(y)
    return _cka(gram_x, gram_y, debiased)  

def _test_metrics():
    embeddings = np.random.random((20, 512))
    embeddings = [[embeddings[i,:]]*5 for i in range(0, 20)]
    embeddings = [e for e5 in embeddings for e in e5]
    embeddings = np.array(embeddings)
    assert embeddings.shape == (100, 512)
    
    ids = [ [i]*5 for i in range(1,21)]
    ids = [j for i in ids for j in i]
    
    recalls = top_k_augmentations_recall(embeddings, ids, 4, 4)
    assert recalls == [1]*20
    
    avg_ranks, min_ranks, max_ranks = augmentations_rank(embeddings, ids)
    assert avg_ranks == [np.mean([1,2,3,4])]*20
    assert min_ranks == [1]*20
    assert max_ranks == [4]*20

    rbf_cka_value = rbf_cka(embeddings, ids, 4)
    linear_cka_value = linear_cka(embeddings, ids, 4)

    assert rbf_cka_value == 1.0
    assert linear_cka_value == 1.0

    var_var, mean_var = variance(embeddings, ids)

    assert var_var>0
    assert mean_var>0

