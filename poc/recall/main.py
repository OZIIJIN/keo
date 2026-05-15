import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_LEGACY_EMBEDDINGS_URL = "http://localhost:11434/api/embeddings"
DEFAULT_TOP_K = 10
DATA_PATH = Path(__file__).parent / "eval" / "past_memos.json"
EVAL_PATH = Path(__file__).parent / "eval" / "eval_cases.json"
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def build_past_memo_index(past_memos: list[dict], model: str) -> list[dict]:
    indexed_memos = []

    for memo in past_memos:
        indexed_memos.append({**memo, "embedding": embed_text(memo["text"], model)})

    return indexed_memos


def find_similar_memos(current_memo: str, past_memos: list[dict], top_k: int, model: str) -> list[dict]:
    current_embedding = embed_text(current_memo, model)
    results = []

    for memo in past_memos:
        past_embedding = memo.get("embedding")
        if past_embedding is None:
            past_embedding = embed_text(memo["text"], model)

        score = cosine_similarity(current_embedding, past_embedding)
        results.append({**memo, "score": score})

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def print_results(
    model: str,
    current_memo: str,
    results: list[dict],
    expected_ids: set[str] | None = None,
) -> None:
    print("\n=== Recall Retrieval POC ===")
    print(f"Embedding model: {model}")
    print("\n[Current memo]")
    print(current_memo)
    if expected_ids is not None:
        print("\n[Expected related memo ids]")
        print(", ".join(sorted(expected_ids)))
    print("\n[Top similar past memos]")

    for rank, memo in enumerate(results, start=1):
        hit = ""
        if expected_ids is not None:
            hit = f" hit={'YES' if memo['id'] in expected_ids else 'NO'}"
        print(f"\n{rank}. score={memo['score']:.4f}{hit} id={memo['id']}")
        if memo.get("date"):
            print(f"   date: {memo['date']}")
        print(f"   text: {memo['text']}")


def print_eval_summary(case: dict, results: list[dict], top_k: int) -> None:
    expected_ids = set(case["expected_ids"])
    result_ids = {memo["id"] for memo in results}
    hits = expected_ids & result_ids

    print("\n[Eval summary]")
    print(f"case: {case['id']}")
    print(f"hits@{top_k}: {len(hits)}/{len(expected_ids)}")
    print(f"hit ids: {', '.join(sorted(hits)) if hits else '-'}")
    missed = expected_ids - result_ids
    print(f"missed ids: {', '.join(sorted(missed)) if missed else '-'}")


def evaluate_cases(
    cases: list[dict],
    past_memos: list[dict],
    top_k: int,
    model: str,
    print_details: bool,
) -> tuple[int, int]:
    total_hits = 0
    total_expected = 0

    for case in cases:
        expected_ids = set(case["expected_ids"])
        results = find_similar_memos(case["current_memo"], past_memos, top_k, model)
        result_ids = {memo["id"] for memo in results}
        hits = expected_ids & result_ids

        total_hits += len(hits)
        total_expected += len(expected_ids)

        if print_details:
            print_results(model, case["current_memo"], results, expected_ids)
            print_eval_summary(case, results, top_k)

    return total_hits, total_expected


def run_eval(args: argparse.Namespace) -> None:
    past_memos = build_past_memo_index(load_past_memos(args.data), args.model)
    cases = load_eval_cases(args.eval_data)
    total_hits, total_expected = evaluate_cases(
        cases,
        past_memos,
        args.top_k,
        args.model,
        not args.summary_only,
    )

    print("\n=== Overall Eval Summary ===")
    print(f"model: {args.model}")
    print(f"cases: {len(cases)}")
    print(f"total hits@{args.top_k}: {total_hits}/{total_expected}")


def read_current_memo(args: argparse.Namespace) -> str:
    if args.memo:
        return args.memo.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("현재 메모를 입력하세요: ").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find past memos that are semantically similar to the current memo."
    )
    parser.add_argument("--memo", help="current memo text")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="number of results")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval:
        run_eval(args)
        return

    current_memo = read_current_memo(args)

    if not current_memo:
        raise SystemExit("현재 메모가 비어 있습니다.")

    past_memos = load_past_memos(args.data)
    indexed_past_memos = build_past_memo_index(past_memos, args.model)
    results = find_similar_memos(current_memo, indexed_past_memos, args.top_k, args.model)
    print_results(args.model, current_memo, results)


if __name__ == "__main__":
    main()
