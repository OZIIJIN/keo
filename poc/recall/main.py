import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path


MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_LEGACY_EMBEDDINGS_URL = "http://localhost:11434/api/embeddings"
DEFAULT_TOP_K = 10
DATA_PATH = Path(__file__).parent / "eval" / "past_memos.json"
EVAL_PATH = Path(__file__).parent / "eval" / "eval_cases.json"


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


def embed_text(text: str) -> list[float]:
    try:
        payload = post_json(OLLAMA_EMBED_URL, {"model": MODEL, "input": text})
        return payload["embeddings"][0]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError(
                "Ollama embedding request failed. "
                "Run `ollama serve` and `ollama pull nomic-embed-text` first."
            ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama embedding request failed. "
            "Run `ollama serve` and `ollama pull nomic-embed-text` first."
        ) from error

    try:
        payload = post_json(
            OLLAMA_LEGACY_EMBEDDINGS_URL,
            {"model": MODEL, "prompt": text},
        )
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama embedding request failed. "
            "Run `ollama serve` and `ollama pull nomic-embed-text` first."
        ) from error

    return payload["embedding"]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def find_similar_memos(current_memo: str, past_memos: list[dict], top_k: int) -> list[dict]:
    current_embedding = embed_text(current_memo)
    results = []

    for memo in past_memos:
        past_embedding = embed_text(memo["text"])
        score = cosine_similarity(current_embedding, past_embedding)
        results.append({**memo, "score": score})

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def print_results(
    current_memo: str,
    results: list[dict],
    expected_ids: set[str] | None = None,
) -> None:
    print("\n=== Recall Retrieval POC ===")
    print(f"Embedding model: {MODEL}")
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


def run_eval(args: argparse.Namespace) -> None:
    past_memos = load_past_memos(args.data)
    cases = load_eval_cases(args.eval_data)

    total_hits = 0
    total_expected = 0

    for case in cases:
        expected_ids = set(case["expected_ids"])
        results = find_similar_memos(case["current_memo"], past_memos, args.top_k)
        result_ids = {memo["id"] for memo in results}
        hits = expected_ids & result_ids

        total_hits += len(hits)
        total_expected += len(expected_ids)

        print_results(case["current_memo"], results, expected_ids)
        print_eval_summary(case, results, args.top_k)

    print("\n=== Overall Eval Summary ===")
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
        "--eval",
        action="store_true",
        help="run fixture eval cases instead of a single memo search",
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
    results = find_similar_memos(current_memo, past_memos, args.top_k)
    print_results(current_memo, results)


if __name__ == "__main__":
    main()
