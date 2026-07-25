import faiss  # type: ignore
import numpy as np

from app.ingestion.ingest import _build_faiss_index


def test_hnsw_flat_metric_is_inner_product():
    """
    (C-3) IndexHNSWFlat for corpora >100 chunks must use METRIC_INNER_PRODUCT.
    If it defaults to METRIC_L2, then descending sort in retriever.py (treating
    L2 distance as a cosine similarity) silently inverts retrieval.
    This test verifies that the built index has metric_type set to METRIC_INNER_PRODUCT.
    """
    dim = 384
    n_vectors = 120  # >100 to trigger HNSW path

    # Create random vectors
    vectors = np.random.rand(n_vectors, dim).astype("float32")

    # Build index using the real ingestion pipeline function
    index = _build_faiss_index(vectors)

    # Assert that the index uses METRIC_INNER_PRODUCT (which is value 0)
    # rather than the default METRIC_L2 (which is value 1)
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT, (
        f"Expected index metric_type to be METRIC_INNER_PRODUCT ({faiss.METRIC_INNER_PRODUCT}), "
        f"but got {index.metric_type} (METRIC_L2 is {faiss.METRIC_L2})"
    )


def test_logical_proof_sorting_disagreement():
    """
    Logical Proof: Prove that sorting by METRIC_L2 in descending order on normalised vectors
    yields completely inverted/incorrect results compared to METRIC_INNER_PRODUCT (cosine similarity).
    """
    # 1. Create a query vector and two candidate vectors as lists to avoid
    # NumPy array truthiness issues in list comparison.
    q = np.array([1.0, 0.0], dtype=np.float32)
    v_close = [0.9, 0.43588989]  # cosine ~ 0.90
    v_far = [-0.8, 0.6]          # cosine ~ -0.80

    # 2. Compute inner products (cosine similarities for normalised vectors)
    ip_close = np.dot(q, np.array(v_close, dtype=np.float32))
    ip_far = np.dot(q, np.array(v_far, dtype=np.float32))

    # Sorted by inner product descending (best first):
    # v_close should be first, then v_far
    ip_ranking = [v_close, v_far] if ip_close > ip_far else [v_far, v_close]

    # 3. Compute L2 distances squared
    l2_close = np.sum((q - np.array(v_close, dtype=np.float32)) ** 2)
    l2_far = np.sum((q - np.array(v_far, dtype=np.float32)) ** 2)

    # If we incorrectly sort descending by L2 distance (treating larger values as better):
    # since l2_far (3.24) > l2_close (0.20), v_far will be ranked first!
    l2_desc_ranking = [v_close, v_far] if l2_close > l2_far else [v_far, v_close]

    # They MUST disagree. The L2 descending ranking is the inverse of correct similarity ranking!
    assert ip_ranking != l2_desc_ranking, (
        "Expected ranking by inner-product descending to disagree with "
        "ranking by L2-distance descending (which inverts retrieval)."
    )

    # The correct ranking (inner product descending) puts v_close first:
    assert ip_ranking[0] == v_close

    # The inverted ranking (L2 descending) puts v_far first:
    assert l2_desc_ranking[0] == v_far
