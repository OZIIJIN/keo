import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from rank_bm25 import BM25Okapi


DEFAULT_MODEL = "qwen3-embedding"
DEFAULT_MODE = "hybrid"
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_MULTIPLIER = 3
DEFAULT_RRF_K = 60
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_LEGACY_EMBEDDINGS_URL = "http://localhost:11434/api/embeddings"
DATA_PATH = Path(__file__).parent / "eval" / "past_memos.json"
EVAL_PATH = Path(__file__).parent / "eval" / "eval_cases.json"
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
EMBED_CACHE: dict[tuple[str, str], list[float]] = {}


def load_past_memos(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        memos = json.load(file)

    if not isinstance(memos, list):
        raise ValueError("past memos file must contain a JSON list")

    for memo in memos:
        if "id" not in memo or "text" not in memo:
            raise ValueError("each past memo must have id and text")

    return memos


def load_eval_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("eval cases file must contain a JSON list")

    for case in cases:
        if "id" not in case or "current_memo" not in case or "expected_ids" not in case:
            raise ValueError("each eval case must have id, current_memo, and expected_ids")

    return cases


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def embed_text(text: str, model: str) -> list[float]:
    cache_key = (model, text)
    cached = EMBED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = post_json(OLLAMA_EMBED_URL, {"model": model, "input": text})
        embedding = payload["embeddings"][0]
        EMBED_CACHE[cache_key] = embedding
        return embedding
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError(
                "Ollama embedding request failed. "
                f"Run `ollama serve` and `ollama pull {model}` first."
            ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama embedding request failed. "
            f"Run `ollama serve` and `ollama pull {model}` first."
        ) from error

    try:
        payload = post_json(
            OLLAMA_LEGACY_EMBEDDINGS_URL,
            {"model": model, "prompt": text},
        )
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama embedding request failed. "
            f"Run `ollama serve` and `ollama pull {model}` first."
        ) from error

    embedding = payload["embedding"]
    EMBED_CACHE[cache_key] = embedding
    return embedding


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def build_dense_index(past_memos: list[dict], model: str) -> list[dict]:
    indexed_memos = []

    for memo in past_memos:
        indexed_memos.append({**memo, "embedding": embed_text(memo["text"], model)})

    return indexed_memos


def build_sparse_index(past_memos: list[dict]) -> dict:
    indexed_memos = []
    tokenized_corpus = []

    for memo in past_memos:
        tokens = tokenize_text(memo["text"])
        tokenized_corpus.append(tokens)
        indexed_memos.append(
            {
                **memo,
                "tokens": tokens,
            }
        )

    return {
        "memos": indexed_memos,
        "bm25": BM25Okapi(tokenized_corpus) if tokenized_corpus else None,
    }


def find_dense_memos(current_memo: str, past_memos: list[dict], top_k: int, model: str) -> list[dict]:
    current_embedding = embed_text(current_memo, model)
    results = []

    for memo in past_memos:
        score = cosine_similarity(current_embedding, memo["embedding"])
        results.append({**memo, "dense_score": score})

    return sorted(results, key=lambda item: item["dense_score"], reverse=True)[:top_k]


def find_sparse_memos(current_memo: str, sparse_index: dict, top_k: int) -> list[dict]:
    query_tokens = tokenize_text(current_memo)
    bm25 = sparse_index["bm25"]
    if not query_tokens or bm25 is None:
        return []

    scores = bm25.get_scores(query_tokens)
    results = []

    for memo, score in zip(sparse_index["memos"], scores):
        if score <= 0:
            continue
        results.append({**memo, "sparse_score": score})

    return sorted(results, key=lambda item: item["sparse_score"], reverse=True)[:top_k]


def rrf_fuse_results(
    dense_results: list[dict],
    sparse_results: list[dict],
    *,
    rrf_k: int,
    top_k: int,
) -> list[dict]:
    fused: dict[str, dict] = {}

    for rank, memo in enumerate(dense_results, start=1):
        item = fused.setdefault(
            memo["id"],
            {
                **memo,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
                "fusion_score": 0.0,
            },
        )
        item["dense_rank"] = rank
        item["dense_score"] = memo["dense_score"]
        item["fusion_score"] += 1.0 / (rrf_k + rank)

    for rank, memo in enumerate(sparse_results, start=1):
        item = fused.setdefault(
            memo["id"],
            {
                **memo,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_score": None,
                "sparse_score": None,
                "fusion_score": 0.0,
            },
        )
        item["sparse_rank"] = rank
        item["sparse_score"] = memo["sparse_score"]
        item["fusion_score"] += 1.0 / (rrf_k + rank)

    return sorted(
        fused.values(),
        key=lambda item: (
            item["fusion_score"],
            item["dense_score"] if item["dense_score"] is not None else -1.0,
            item["sparse_score"] if item["sparse_score"] is not None else -1.0,
        ),
        reverse=True,
    )[:top_k]


def run_retrieval(
    current_memo: str,
    dense_index: list[dict],
    sparse_index: dict,
    *,
    top_k: int,
    candidate_k: int,
    model: str,
    mode: str,
    rrf_k: int,
) -> dict:
    dense_results = find_dense_memos(current_memo, dense_index, candidate_k, model)

    if mode == "dense":
        return {
            "mode": mode,
            "dense_results": dense_results,
            "sparse_results": [],
            "final_results": dense_results[:top_k],
        }

    sparse_results = find_sparse_memos(current_memo, sparse_index, candidate_k)
    fused_results = rrf_fuse_results(
        dense_results,
        sparse_results,
        rrf_k=rrf_k,
        top_k=top_k,
    )
    return {
        "mode": mode,
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "final_results": fused_results,
    }


def print_results(
    model: str,
    current_memo: str,
    retrieval: dict,
    expected_ids: set[str] | None = None,
) -> None:
    print("\n=== Recall Retrieval POC ===")
    print(f"Mode: {retrieval['mode']}")
    print(f"Embedding model: {model}")
    print("\n[Current memo]")
    print(current_memo)

    if expected_ids is not None:
        print("\n[Expected related memo ids]")
        print(", ".join(sorted(expected_ids)))

    if retrieval["mode"] == "hybrid":
        print("\n[Dense candidates]")
        for rank, memo in enumerate(retrieval["dense_results"], start=1):
            print(f"{rank}. dense_score={memo['dense_score']:.4f} id={memo['id']}")

        print("\n[Sparse candidates]")
        if retrieval["sparse_results"]:
            for rank, memo in enumerate(retrieval["sparse_results"], start=1):
                print(f"{rank}. sparse_score={memo['sparse_score']:.4f} id={memo['id']}")
        else:
            print("-")

        print("\n[Fused top results]")
    else:
        print("\n[Top similar past memos]")

    for rank, memo in enumerate(retrieval["final_results"], start=1):
        hit = ""
        if expected_ids is not None:
            hit = f" hit={'YES' if memo['id'] in expected_ids else 'NO'}"

        score_parts = []
        if retrieval["mode"] == "hybrid":
            score_parts.append(f"fusion={memo['fusion_score']:.4f}")
            if memo.get("dense_rank") is not None:
                score_parts.append(f"dense_rank={memo['dense_rank']}")
            if memo.get("dense_score") is not None:
                score_parts.append(f"dense_score={memo['dense_score']:.4f}")
            if memo.get("sparse_rank") is not None:
                score_parts.append(f"sparse_rank={memo['sparse_rank']}")
            if memo.get("sparse_score") is not None:
                score_parts.append(f"sparse_score={memo['sparse_score']:.4f}")
        else:
            score_parts.append(f"dense_score={memo['dense_score']:.4f}")

        print(f"\n{rank}. {' '.join(score_parts)}{hit} id={memo['id']}")
        if memo.get("date"):
            print(f"   date: {memo['date']}")
        print(f"   text: {memo['text']}")


def summarize_hits(expected_ids: set[str], results: list[dict]) -> dict:
    result_ids = {memo["id"] for memo in results}
    hits = expected_ids & result_ids
    missed = expected_ids - result_ids
    return {
        "hits": hits,
        "missed": missed,
        "num_hits": len(hits),
        "num_expected": len(expected_ids),
    }


def print_eval_summary(case: dict, retrieval: dict, top_k: int) -> None:
    expected_ids = set(case["expected_ids"])
    final_summary = summarize_hits(expected_ids, retrieval["final_results"])

    print("\n[Eval summary]")
    print(f"case: {case['id']}")
    print(f"mode: {retrieval['mode']}")
    print(f"hits@{top_k}: {final_summary['num_hits']}/{final_summary['num_expected']}")
    print(f"hit ids: {', '.join(sorted(final_summary['hits'])) if final_summary['hits'] else '-'}")
    print(
        f"missed ids: {', '.join(sorted(final_summary['missed'])) if final_summary['missed'] else '-'}"
    )

    if retrieval["mode"] == "hybrid":
        dense_summary = summarize_hits(expected_ids, retrieval["dense_results"][:top_k])
        sparse_summary = summarize_hits(expected_ids, retrieval["sparse_results"][:top_k])
        print(
            f"dense hits@{top_k}: {dense_summary['num_hits']}/{dense_summary['num_expected']}"
        )
        print(
            f"sparse hits@{top_k}: {sparse_summary['num_hits']}/{sparse_summary['num_expected']}"
        )


def empty_stats() -> dict:
    return {
        "total_hits": 0,
        "total_expected": 0,
        "zero_hit_cases": 0,
        "full_hit_cases": 0,
        "partial_hit_cases": 0,
    }


def update_stats(stats: dict, expected_ids: set[str], results: list[dict]) -> None:
    summary = summarize_hits(expected_ids, results)
    stats["total_hits"] += summary["num_hits"]
    stats["total_expected"] += summary["num_expected"]

    if summary["num_hits"] == 0:
        stats["zero_hit_cases"] += 1
    elif summary["num_hits"] == summary["num_expected"]:
        stats["full_hit_cases"] += 1
    else:
        stats["partial_hit_cases"] += 1


def evaluate_cases(
    cases: list[dict],
    dense_index: list[dict],
    sparse_index: dict,
    *,
    top_k: int,
    candidate_k: int,
    model: str,
    mode: str,
    rrf_k: int,
    print_details: bool,
) -> dict:
    final_stats = empty_stats()
    dense_stats = empty_stats()
    sparse_stats = empty_stats()

    for case in cases:
        expected_ids = set(case["expected_ids"])
        retrieval = run_retrieval(
            case["current_memo"],
            dense_index,
            sparse_index,
            top_k=top_k,
            candidate_k=candidate_k,
            model=model,
            mode=mode,
            rrf_k=rrf_k,
        )

        update_stats(final_stats, expected_ids, retrieval["final_results"])
        update_stats(dense_stats, expected_ids, retrieval["dense_results"][:top_k])
        if mode == "hybrid":
            update_stats(sparse_stats, expected_ids, retrieval["sparse_results"][:top_k])

        if print_details:
            print_results(model, case["current_memo"], retrieval, expected_ids)
            print_eval_summary(case, retrieval, top_k)

    return {
        "final": final_stats,
        "dense": dense_stats,
        "sparse": sparse_stats,
    }


def print_overall_stats(label: str, stats: dict, top_k: int) -> None:
    total_expected = stats["total_expected"]
    hit_rate = (stats["total_hits"] / total_expected * 100) if total_expected else 0.0
    print(f"\n[{label}]")
    print(f"hits@{top_k}: {stats['total_hits']}/{total_expected}")
    print(f"hit rate: {hit_rate:.2f}%")
    print(f"zero-hit cases: {stats['zero_hit_cases']}")
    print(f"partial-hit cases: {stats['partial_hit_cases']}")
    print(f"full-hit cases: {stats['full_hit_cases']}")


def run_eval(args: argparse.Namespace) -> None:
    past_memos = load_past_memos(args.data)
    dense_index = build_dense_index(past_memos, args.model)
    sparse_index = build_sparse_index(past_memos)
    cases = load_eval_cases(args.eval_data)

    stats = evaluate_cases(
        cases,
        dense_index,
        sparse_index,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        model=args.model,
        mode=args.mode,
        rrf_k=args.rrf_k,
        print_details=not args.summary_only,
    )

    print("\n=== Overall Eval Summary ===")
    print(f"mode: {args.mode}")
    print(f"model: {args.model}")
    print(f"cases: {len(cases)}")
    print(f"candidate_k: {args.candidate_k}")

    print_overall_stats("final", stats["final"], args.top_k)
    print_overall_stats("dense", stats["dense"], args.top_k)
    if args.mode == "hybrid":
        print_overall_stats("sparse", stats["sparse"], args.top_k)


def read_current_memo(args: argparse.Namespace) -> str:
    if args.memo:
        return args.memo.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("현재 메모를 입력하세요: ").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find past memos that are relevant to the current memo."
    )
    parser.add_argument("--memo", help="current memo text")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="number of final results")
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="number of dense/sparse candidates before fusion",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF smoothing constant",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "dense"),
        default=DEFAULT_MODE,
        help="retrieval mode",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama embedding model to use",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="run fixture eval cases instead of a single memo search",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only overall eval summary",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="path to test past memo JSON data",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=EVAL_PATH,
        help="path to eval case JSON data",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than 0.")

    if args.rrf_k <= 0:
        raise SystemExit("--rrf-k must be greater than 0.")

    if args.candidate_k is None:
        args.candidate_k = max(args.top_k, args.top_k * DEFAULT_CANDIDATE_MULTIPLIER)

    if args.candidate_k < args.top_k:
        raise SystemExit("--candidate-k must be greater than or equal to --top-k.")

    return args


def main() -> None:
    args = parse_args()
    if args.eval:
        run_eval(args)
        return

    current_memo = read_current_memo(args)

    if not current_memo:
        raise SystemExit("현재 메모가 비어 있습니다.")

    past_memos = load_past_memos(args.data)
    dense_index = build_dense_index(past_memos, args.model)
    sparse_index = build_sparse_index(past_memos)
    retrieval = run_retrieval(
        current_memo,
        dense_index,
        sparse_index,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        model=args.model,
        mode=args.mode,
        rrf_k=args.rrf_k,
    )
    print_results(args.model, current_memo, retrieval)


if __name__ == "__main__":
    main()
