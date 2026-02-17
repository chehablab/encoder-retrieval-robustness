import faiss
from enum import Enum
import numpy as np

class RetrievalMetrics(Enum):
    MEAN_PRECISION = "mean_precision"
    MEAN_AVERAGE_PRECISION = "mAP"

def retrieval_mean_precision(embeddings, labels, k):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, neighbors = index.search(embeddings, k + 1)
    neighbors = neighbors[:, 1:]
    same_labels = labels[neighbors] == labels[:, None]
    return same_labels.mean().item()

def mean_average_precision(queries, q_labels, embeddings, labels, k):
    faiss.normalize_L2(queries)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    average_precision = []

    for i, q in enumerate(queries):
        q = q.reshape(1, -1)

        if queries is embeddings:
            _, neighbors = index.search(q, k + 1)
            neighbors = neighbors[0, 1:]
        else:
            _, neighbors = index.search(q, k)
            neighbors = neighbors[0]
        total_relevant = np.sum(labels == q_labels[i])
        normalizer = min(total_relevant, k)

        ap = 0.0
        rel = 0

        for rank, idx in enumerate(neighbors):
            if labels[idx] == q_labels[i]:
                rel += 1
                ap += rel / (rank + 1)

        if rel > 0:
            ap /= normalizer
        else:
            ap = 0.0

        average_precision.append(ap)
    return np.mean(average_precision)


def _test_mean_average_precision():
    embeddings = np.array([
        [1.0], [0.95], [0.9], [0.85], [0.8], [0.75],
        [-1.0], [-0.95], [-0.9], [-0.85], [-0.8], [-0.75],
    ], dtype="float32")

    labels = np.array([0]*6 + [1]*6)

    query = embeddings.copy()
    q_labels = labels.copy()

    k = 5

    mAP = mean_average_precision(query, q_labels, embeddings, labels, k)

    print("mAP@5:", mAP)

    assert np.isclose(mAP, 1.0), "mAP@5 should be exactly 1.0"
    print("Test passed")

def _test_retrieval_mean_precision():
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3], [-4, -5, -6]], dtype="float32")
    labels = np.array([0, 0, 1, 1])
    k = 2
    assert retrieval_mean_precision(embeddings, labels, k) == 0.5
    print("Test passed")