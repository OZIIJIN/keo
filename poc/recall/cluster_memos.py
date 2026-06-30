"""
HDBSCAN clustering on raw text embeddings of past_memos.
Usage: python3 cluster_memos.py [--min-cluster-size N] [--min-samples N]
"""
import json
import argparse
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize

EMBED_CACHE = "cache/embed_cache.qwen3-embedding.json"
PAST_MEMOS  = "eval/past_memos.json"


def load_data():
    with open(EMBED_CACHE, encoding="utf-8") as f:
        cache = json.load(f)
    with open(PAST_MEMOS, encoding="utf-8") as f:
        memos = json.load(f)
    return cache, memos


def build_matrix(cache, memos):
    vecs, valid = [], []
    for m in memos:
        v = cache.get(m["text"])
        if v:
            vecs.append(v)
            valid.append(m)
    X = normalize(np.array(vecs, dtype=np.float32))  # cosine → 유클리드 근사
    return X, valid


def run_hdbscan(X, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(X)
    return labels


def print_results(labels, memos):
    unique = sorted(set(labels))
    n_clusters = sum(1 for l in unique if l >= 0)
    n_noise    = sum(1 for l in labels if l == -1)

    print(f"\n{'='*60}")
    print(f"총 메모: {len(memos)}개")
    print(f"군집 수: {n_clusters}개")
    print(f"노이즈:  {n_noise}개 ({n_noise/len(memos)*100:.1f}%)")
    print(f"{'='*60}\n")

    for cid in unique:
        if cid == -1:
            continue
        members = [memos[i] for i, l in enumerate(labels) if l == cid]
        print(f"── Cluster {cid} ({len(members)}개) ──")
        for m in members:
            text = m["text"]
            print(f"  [{m['id']}] {text[:60]}{'...' if len(text)>60 else ''}")
        print()

    noise_memos = [memos[i] for i, l in enumerate(labels) if l == -1]
    print(f"── Noise ({len(noise_memos)}개) ──")
    for m in noise_memos[:10]:
        print(f"  [{m['id']}] {m['text'][:60]}")
    if len(noise_memos) > 10:
        print(f"  ... 외 {len(noise_memos)-10}개")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="군집으로 인정할 최소 메모 수 (default: 5)")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="핵심 포인트 판정 최소 이웃 수 (default: 3)")
    args = parser.parse_args()

    print(f"min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}")
    print("데이터 로딩...")
    cache, memos = load_data()

    print("임베딩 행렬 구성...")
    X, valid_memos = build_matrix(cache, memos)
    print(f"유효 메모: {len(valid_memos)}개, 벡터 차원: {X.shape[1]}")

    print("HDBSCAN 실행...")
    labels = run_hdbscan(X, args.min_cluster_size, args.min_samples)

    print_results(labels, valid_memos)


if __name__ == "__main__":
    main()
