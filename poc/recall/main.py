import argparse
import atexit
import http.client
import json
import math
import re
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "qwen3-embedding"
DEFAULT_TAGGING_MODEL = "qwen3:8b"
DEFAULT_RERANK_MODEL = "qwen3:8b"
DEFAULT_TOP_K = 10
EMBED_TIMEOUT_SECONDS = 60
GENERATE_TIMEOUT_SECONDS = 900
DEFAULT_RRF_K = 60
DEFAULT_RELATION_CANDIDATE_K = 30
DEFAULT_RERANK_CANDIDATE_K = 30
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_LEGACY_EMBEDDINGS_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DATA_PATH = Path(__file__).parent / "eval" / "past_memos.json"
EVAL_PATH = Path(__file__).parent / "eval" / "eval_cases.json"
TAG_SCHEMA_PATH = Path(__file__).parent / "tag_schema.json"
PAST_TAGS_PATH = Path(__file__).parent / "eval" / "past_memo_tags.json"
EVAL_CASE_TAGS_PATH = Path(__file__).parent / "eval" / "eval_case_tags.json"
CACHE_DIR = Path(__file__).parent / "cache"
TAGGING_FIXED = "fixed"
TAGGING_LLM = "llm"
RETRIEVAL_TEXT = "text"
RETRIEVAL_RELATION = "relation"
RETRIEVAL_RRF = "rrf"
RETRIEVAL_RELATION_GATE = "relation_gate"
RETRIEVAL_TEXT_RERANK = "text_rerank"
EMBED_CACHE: dict[tuple[str, str], list[float]] = {}
EMBED_CACHE_LOADED_MODELS: set[str] = set()
EMBED_CACHE_DIRTY_MODELS: set[str] = set()
RERANK_LABELS = {
    "회피": "해야 할 본체 대신 주변으로 샘",
    "미룸": "할 일을 다음으로 계속 넘김",
    "흐름끊김": "일정이나 알림으로 집중이 깨짐",
    "피로": "몸 상태가 판단과 실행에 영향을 줌",
    "쉼불안": "쉬어도 불안하거나 회복이 안 됨",
    "생활밀림": "자잘한 생활 일이 쌓임",
    "돈걱정": "돈 생각이 올라와 집중이 흐려짐",
    "여운": "장면, 문장, 사건의 잔상이 오래 감",
    "겉속다름": "겉으로 보인 나와 실제 내가 어긋남",
    "패턴발견": "반복되는 흐름을 알아차림",
    "아이디어": "앱이나 구조에 대한 통찰",
    "잘됐다": "흐름이 좋거나 뭔가 잘 풀린 날",
    "회복": "쉬거나 환경을 바꿔서 나아진 느낌",
    "발견": "읽다가 또는 우연히 뭔가를 새로 알게 됨",
    "몰입": "흐름이 잘 타서 깊게 들어간 날",
    "관찰": "판단 없이 있는 그대로 기록한 메모",
}
RERANK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "reason": {"type": "string"},
        "results": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["label", "reason", "results"],
}


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def get_embed_cache_path(model: str) -> Path:
    return CACHE_DIR / f"embed_cache.{sanitize_model_name(model)}.json"


def load_embed_cache_for_model(model: str) -> None:
    if model in EMBED_CACHE_LOADED_MODELS:
        return

    cache_path = get_embed_cache_path(model)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError(f"embed cache file must contain a JSON object: {cache_path}")

        for text, embedding in payload.items():
            if isinstance(text, str) and isinstance(embedding, list):
                EMBED_CACHE[(model, text)] = embedding

    EMBED_CACHE_LOADED_MODELS.add(model)


def save_embed_cache_for_model(model: str) -> None:
    if model not in EMBED_CACHE_DIRTY_MODELS:
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_embed_cache_path(model)
    payload = {
        text: embedding
        for (cached_model, text), embedding in EMBED_CACHE.items()
        if cached_model == model
    }
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)

    EMBED_CACHE_DIRTY_MODELS.discard(model)


def save_all_embed_caches() -> None:
    for model in list(EMBED_CACHE_DIRTY_MODELS):
        save_embed_cache_for_model(model)


atexit.register(save_all_embed_caches)


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


def load_json_dict(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")

    return payload


def load_tag_schema(path: Path) -> dict:
    schema = load_json_dict(path)
    relation_tags = schema.get("relation_tags")

    if not isinstance(relation_tags, dict):
        raise ValueError("tag schema must have relation_tags object")

    return schema


def normalize_annotation(annotation: dict, schema: dict) -> dict:
    relation_tags = annotation.get("relation_tags", [])
    known_relation_tags = schema["relation_tags"]

    if not isinstance(relation_tags, list) or any(not isinstance(tag, str) for tag in relation_tags):
        raise ValueError("relation_tags must be a list of strings")

    unknown_relation_tags = [tag for tag in relation_tags if tag not in known_relation_tags]
    if unknown_relation_tags:
        raise ValueError(f"unknown relation tags: {', '.join(unknown_relation_tags)}")

    return {"relation_tags": relation_tags}


def load_tag_annotations(path: Path, schema: dict) -> dict[str, dict]:
    raw_annotations = load_json_dict(path)
    annotations = {}

    for item_id, annotation in raw_annotations.items():
        if not isinstance(annotation, dict):
            raise ValueError(f"tag annotation for {item_id} must be an object")
        annotations[item_id] = normalize_annotation(annotation, schema)

    return annotations


def parse_relation_tags(raw: str | None, schema: dict) -> list[str]:
    if not raw:
        return []

    relation_tags = [tag.strip() for tag in raw.split(",") if tag.strip()]
    normalized = normalize_annotation({"relation_tags": relation_tags}, schema)
    return normalized["relation_tags"]


def build_relation_text(relation_tags: list[str]) -> str:
    if not relation_tags:
        return ""
    return "[relation_tags]\n" + "\n".join(relation_tags)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def generate_text(
    prompt: str,
    model: str,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
    response_format: str | dict | None = None,
) -> str:
    request_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    if response_format is not None:
        request_payload["format"] = response_format

    try:
        payload = post_json(
            OLLAMA_GENERATE_URL,
            request_payload,
            timeout=timeout,
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.RemoteDisconnected) as error:
        raise RuntimeError(
            "Ollama generation request failed. "
            f"Run `ollama serve` and `ollama pull {model}` first."
        ) from error

    response = payload.get("response")
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("Ollama generation response was empty")

    return response.strip()


def build_tagging_prompt(text: str, schema: dict) -> str:
    relation_lines = [
        f"- {tag}: {description}"
        for tag, description in schema["relation_tags"].items()
    ]

    return "\n".join(
        [
            "You classify a Korean memo into fixed KEO relation tags.",
            "Choose up to two relation tags.",
            "The goal is to expose retrieval signals, not memo type.",
            "Use only signals that are directly supported by the memo text.",
            "Do not infer hidden causes unless the memo clearly states them.",
            "Prefer concrete, observable tags over abstract or interpretive tags.",
            "If a tag needs extra interpretation, do not choose it.",
            "If two relation tags are similar, choose the more literal one.",
            "Return JSON only. Do not add markdown or explanation.",
            'JSON schema: {"relation_tags":["<tag1>","<tag2>"]}',
            "",
            "[Relation tags]",
            *relation_lines,
            "",
            "[Memo]",
            text.strip(),
        ]
    )


def build_rerank_prompt(current_memo: str, candidates: list[dict], top_k: int) -> str:
    candidate_lines = []
    for candidate in candidates:
        candidate_lines.append(f"- {candidate['id']}: {candidate['text']}")
    label_lines = [f"- {label}: {description}" for label, description in RERANK_LABELS.items()]

    return "\n".join(
        [
            "/no_think",
            "You review Korean memo retrieval candidates for KEO.",
            "Return one JSON object only.",
            "Do not write markdown, bullets, headings, or explanation outside JSON.",
            "Rank the candidates by relevance to the current memo.",
            "Prefer deeper pattern similarity over surface word overlap.",
            "A candidate can still be relevant if the domain differs but the repeated pattern is similar.",
            f"Return exactly {min(top_k, len(candidates))} candidate ids unless there are fewer candidates.",
            "Then choose one label from the fixed label list below.",
            'Required JSON shape: {"label":"패턴발견","reason":"짧은 한 문장","results":["memo-123","memo-045"]}',
            'The "results" field is required.',
            'Never return an empty object.',
            'Never return an empty "results" list when candidates are provided.',
            'Do not put memo ids inside "reason". Put memo ids only inside "results".',
            'Do not rename keys. Use exactly these keys: "label", "reason", "results".',
            "Use only candidate ids from the list.",
            "Write label and reason in Korean.",
            "The label must be chosen from the fixed label list below.",
            "The label should describe the shared pattern of the selected set, not each memo separately.",
            "Use '회피' when the memo avoids the main thing by moving sideways.",
            "Use '미룸' when the main thing is simply pushed to later.",
            "Use '발견' for a new realization and '패턴발견' for noticing a repeated personal pattern.",
            "If the shared pattern is weak or mixed, still rank the closest candidates and use an empty label.",
            "Keep reason short, one sentence max.",
            'Example valid output: {"label":"패턴발견","reason":"시작을 미루는 반복 패턴이 보인다.","results":["memo-006","memo-022"]}',
            'Example valid output with weak pattern: {"label":"","reason":"","results":["memo-006","memo-022"]}',
            "",
            "[Fixed labels]",
            *label_lines,
            "",
            "[Current memo]",
            current_memo.strip(),
            "",
            "[Candidates]",
            *candidate_lines,
        ]
    )


def extract_json_value(text: str) -> object:
    start = text.find("{")
    array_start = text.find("[")

    if start == -1 and array_start == -1:
        raise ValueError("no JSON object found in LLM response")

    if array_start != -1 and (start == -1 or array_start < start):
        start = array_start

    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(text[start:])
    return payload


def extract_json_object(text: str) -> dict:
    payload = extract_json_value(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def sanitize_relation_tags(annotation: dict, schema: dict) -> dict:
    raw_relation_tags = annotation.get("relation_tags", [])
    if not isinstance(raw_relation_tags, list):
        return {"relation_tags": []}

    known_relation_tags = schema["relation_tags"]
    relation_tags = [
        tag for tag in raw_relation_tags
        if isinstance(tag, str) and tag in known_relation_tags
    ]
    return {"relation_tags": relation_tags[:2]}


def annotate_with_llm(text: str, schema: dict, model: str, timeout: int = GENERATE_TIMEOUT_SECONDS) -> dict:
    prompt = build_tagging_prompt(text, schema)
    response = generate_text(prompt, model, timeout=timeout, response_format="json")
    annotation = extract_json_object(response)
    return sanitize_relation_tags(annotation, schema)


def rerank_with_llm(
    current_memo: str,
    candidates: list[dict],
    model: str,
    top_k: int,
    timeout: int = GENERATE_TIMEOUT_SECONDS,
) -> tuple[list[dict], dict | None]:
    if not candidates:
        return [], None

    prompt = build_rerank_prompt(current_memo, candidates, top_k)
    response = generate_text(prompt, model, timeout=timeout, response_format=RERANK_RESPONSE_SCHEMA)
    try:
        payload = extract_json_value(response)
    except Exception as error:
        raise RuntimeError(
            "LLM reranker returned invalid JSON. "
            f"Raw response: {response[:1000]}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError("LLM reranker response must be a JSON object")

    raw_results = payload.get("results", [])

    if not isinstance(raw_results, list):
        raise RuntimeError("LLM reranker response must contain a results list")

    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    reranked = []
    seen_ids: set[str] = set()

    for memo_id in raw_results:
        if not isinstance(memo_id, str) or memo_id not in candidate_by_id or memo_id in seen_ids:
            continue
        reranked.append(
            {
                **candidate_by_id[memo_id],
                "score": candidate_by_id[memo_id]["score"],
            }
        )
        seen_ids.add(memo_id)

        if len(reranked) >= top_k:
            break

    if not reranked:
        raise RuntimeError(
            "LLM reranker did not return any valid candidate ids. "
            f"Raw response: {response[:1000]}"
        )

    label = payload.get("label")
    reason = payload.get("reason")
    summary = None
    if isinstance(label, str) and label not in RERANK_LABELS:
        label = ""
    if isinstance(label, str) or isinstance(reason, str):
        summary = {
            "label": label if isinstance(label, str) else "",
            "reason": reason if isinstance(reason, str) else "",
        }

    return reranked, summary


def embed_text(text: str, model: str) -> list[float]:
    load_embed_cache_for_model(model)
    cache_key = (model, text)
    cached = EMBED_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = post_json(
            OLLAMA_EMBED_URL,
            {"model": model, "input": text},
            timeout=EMBED_TIMEOUT_SECONDS,
        )
        embedding = payload["embeddings"][0]
        EMBED_CACHE[cache_key] = embedding
        EMBED_CACHE_DIRTY_MODELS.add(model)
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
            timeout=EMBED_TIMEOUT_SECONDS,
        )
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama embedding request failed. "
            f"Run `ollama serve` and `ollama pull {model}` first."
        ) from error

    embedding = payload["embedding"]
    EMBED_CACHE[cache_key] = embedding
    EMBED_CACHE_DIRTY_MODELS.add(model)
    return embedding


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def build_past_memo_index(
    past_memos: list[dict],
    model: str,
    annotations: dict[str, dict],
    retrieval_mode: str,
) -> list[dict]:
    indexed_memos = []

    for memo in past_memos:
        annotation = annotations.get(memo["id"])
        relation_tags = annotation.get("relation_tags", []) if annotation else []
        relation_text = build_relation_text(relation_tags)
        text_embedding = None
        relation_embedding = None

        if retrieval_mode in {RETRIEVAL_TEXT, RETRIEVAL_RRF, RETRIEVAL_RELATION_GATE, RETRIEVAL_TEXT_RERANK}:
            text_embedding = embed_text(memo["text"], model)

        if retrieval_mode in {RETRIEVAL_RELATION, RETRIEVAL_RRF, RETRIEVAL_RELATION_GATE} and relation_text:
            relation_embedding = embed_text(relation_text, model)

        indexed_memos.append(
            {
                **memo,
                "relation_tags": relation_tags,
                "relation_text": relation_text,
                "text_embedding": text_embedding,
                "relation_embedding": relation_embedding,
            }
        )

    return indexed_memos


def rank_by_embedding(
    query_embedding: list[float] | None,
    past_memos: list[dict],
    embedding_field: str,
) -> list[dict]:
    if query_embedding is None:
        return []

    results = []
    for memo in past_memos:
        memo_embedding = memo.get(embedding_field)
        if memo_embedding is None:
            continue
        score = cosine_similarity(query_embedding, memo_embedding)
        results.append({**memo, "score": score})

    return sorted(results, key=lambda item: item["score"], reverse=True)


def rrf_fuse(rankings: list[list[dict]], top_k: int, rrf_k: int) -> list[dict]:
    fused_scores: dict[str, float] = {}
    memo_by_id: dict[str, dict] = {}

    for ranking in rankings:
        for rank, memo in enumerate(ranking, start=1):
            memo_id = memo["id"]
            fused_scores[memo_id] = fused_scores.get(memo_id, 0.0) + 1.0 / (rrf_k + rank)
            if memo_id not in memo_by_id:
                memo_by_id[memo_id] = memo

    fused_results = []
    for memo_id, score in fused_scores.items():
        fused_results.append({**memo_by_id[memo_id], "score": score})

    return sorted(fused_results, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_by_text(
    candidates: list[dict],
    query_embedding: list[float] | None,
    top_k: int,
) -> list[dict]:
    if query_embedding is None:
        return []

    reranked = []
    for memo in candidates:
        memo_embedding = memo.get("text_embedding")
        if memo_embedding is None:
            continue
        score = cosine_similarity(query_embedding, memo_embedding)
        reranked.append({**memo, "score": score})

    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def find_similar_memos(
    current_memo: str,
    past_memos: list[dict],
    top_k: int,
    model: str,
    query_annotation: dict | None,
    retrieval_mode: str,
    rrf_k: int,
    relation_candidate_k: int,
    rerank_model: str | None,
    rerank_candidate_k: int,
    generate_timeout: int,
) -> tuple[list[dict], str | None, dict | None]:
    relation_tags = query_annotation.get("relation_tags", []) if query_annotation else []
    relation_text = build_relation_text(relation_tags)

    text_query_embedding = None
    relation_query_embedding = None

    if retrieval_mode in {RETRIEVAL_TEXT, RETRIEVAL_RRF, RETRIEVAL_RELATION_GATE, RETRIEVAL_TEXT_RERANK}:
        text_query_embedding = embed_text(current_memo, model)

    if retrieval_mode in {RETRIEVAL_RELATION, RETRIEVAL_RRF, RETRIEVAL_RELATION_GATE} and relation_text:
        relation_query_embedding = embed_text(relation_text, model)

    text_results = rank_by_embedding(text_query_embedding, past_memos, "text_embedding")
    relation_results = rank_by_embedding(relation_query_embedding, past_memos, "relation_embedding")

    if retrieval_mode == RETRIEVAL_TEXT:
        return text_results[:top_k], None, None

    if retrieval_mode == RETRIEVAL_RELATION:
        return relation_results[:top_k], relation_text or None, None

    if retrieval_mode == RETRIEVAL_RELATION_GATE:
        relation_candidates = relation_results[:relation_candidate_k]
        results = rerank_by_text(relation_candidates, text_query_embedding, top_k)
        return results, relation_text or None, None

    if retrieval_mode == RETRIEVAL_TEXT_RERANK:
        if rerank_model is None:
            raise RuntimeError("LLM reranker model is required for text_rerank mode")
        text_candidates = text_results[:rerank_candidate_k]
        results, summary = rerank_with_llm(
            current_memo,
            text_candidates,
            rerank_model,
            top_k,
            timeout=generate_timeout,
        )
        return results, None, summary

    results = rrf_fuse([text_results, relation_results], top_k, rrf_k)
    return results, relation_text or None, None


def print_results(
    model: str,
    current_memo: str,
    relation_text: str | None,
    results: list[dict],
    rerank_summary: dict | None = None,
    expected_ids: set[str] | None = None,
) -> None:
    print("\n=== Recall Retrieval POC ===")
    print(f"Embedding model: {model}")
    print("\n[Current memo]")
    print(current_memo)
    if relation_text:
        print("\n[Relation query text]")
        print(relation_text)
    if rerank_summary and (rerank_summary.get("label") or rerank_summary.get("reason")):
        print("\n[Shared pattern]")
        if rerank_summary.get("label"):
            print(f"label: {rerank_summary['label']}")
        if rerank_summary.get("reason"):
            print(f"reason: {rerank_summary['reason']}")
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
        if memo.get("relation_tags"):
            print(f"   relation_tags: {', '.join(memo['relation_tags'])}")
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


def print_eval_summary(case: dict, results: list[dict], top_k: int) -> None:
    summary = summarize_hits(set(case["expected_ids"]), results)

    print("\n[Eval summary]")
    print(f"case: {case['id']}")
    print(f"hits@{top_k}: {summary['num_hits']}/{summary['num_expected']}")
    print(f"hit ids: {', '.join(sorted(summary['hits'])) if summary['hits'] else '-'}")
    print(f"missed ids: {', '.join(sorted(summary['missed'])) if summary['missed'] else '-'}")


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
    past_memos: list[dict],
    top_k: int,
    model: str,
    print_details: bool,
    case_annotations: dict[str, dict],
    tagging_mode: str,
    tagging_model: str | None,
    schema: dict,
    retrieval_mode: str,
    rrf_k: int,
    relation_candidate_k: int,
    rerank_model: str | None,
    rerank_candidate_k: int,
    generate_timeout: int,
) -> dict:
    stats = empty_stats()

    total_cases = len(cases)
    for index, case in enumerate(cases, start=1):
        if not print_details:
            print(f"[eval] {index}/{total_cases} {case['id']}", file=sys.stderr, flush=True)

        expected_ids = set(case["expected_ids"])
        try:
            query_annotation = resolve_query_annotation(
                case["current_memo"],
                case_annotations.get(case["id"]),
                tagging_mode,
                tagging_model,
                schema,
                generate_timeout,
            )
            results, retrieval_text, rerank_summary = find_similar_memos(
                case["current_memo"],
                past_memos,
                top_k,
                model,
                query_annotation,
                retrieval_mode,
                rrf_k,
                relation_candidate_k,
                rerank_model,
                rerank_candidate_k,
                generate_timeout,
            )
        except Exception as error:
            print("\n[Eval failure]")
            print(f"case: {case['id']}")
            print(f"current_memo: {case['current_memo']}")
            raise
        update_stats(stats, expected_ids, results)

        if print_details:
            print_results(model, case["current_memo"], retrieval_text, results, rerank_summary, expected_ids)
            print_eval_summary(case, results, top_k)

    return stats


def print_overall_stats(stats: dict, top_k: int) -> None:
    total_expected = stats["total_expected"]
    hit_rate = (stats["total_hits"] / total_expected * 100) if total_expected else 0.0
    print(f"hits@{top_k}: {stats['total_hits']}/{total_expected}")
    print(f"hit rate: {hit_rate:.2f}%")
    print(f"zero-hit cases: {stats['zero_hit_cases']}")
    print(f"partial-hit cases: {stats['partial_hit_cases']}")
    print(f"full-hit cases: {stats['full_hit_cases']}")


def run_eval(args: argparse.Namespace) -> None:
    schema = load_tag_schema(args.tag_schema)
    past_tags = load_tag_annotations(args.past_tags, schema)
    case_tags = load_tag_annotations(args.eval_case_tags, schema)
    past_memos = build_past_memo_index(
        load_past_memos(args.data),
        args.model,
        past_tags,
        args.retrieval_mode,
    )
    cases = load_eval_cases(args.eval_data)
    stats = evaluate_cases(
        cases,
        past_memos,
        args.top_k,
        args.model,
        not args.summary_only,
        case_tags,
        args.tagging_mode,
        args.tagging_model,
        schema,
        args.retrieval_mode,
        args.rrf_k,
        args.relation_candidate_k,
        args.rerank_model,
        args.rerank_candidate_k,
        args.generate_timeout,
    )

    print("\n=== Overall Eval Summary ===")
    print(f"model: {args.model}")
    print(f"retrieval mode: {args.retrieval_mode}")
    print(f"tagging mode: {args.tagging_mode}")
    if args.tagging_model:
        print(f"tagging model: {args.tagging_model}")
    print(f"cases: {len(cases)}")
    print_overall_stats(stats, args.top_k)


def read_current_memo(args: argparse.Namespace) -> str:
    if args.memo:
        return args.memo.strip()

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    return input("현재 메모를 입력하세요: ").strip()


def resolve_query_annotation(
    current_memo: str,
    fixed_annotation: dict | None,
    tagging_mode: str,
    tagging_model: str | None,
    schema: dict,
    generate_timeout: int,
) -> dict | None:
    if tagging_mode == TAGGING_FIXED:
        return fixed_annotation

    if tagging_model is None:
        raise SystemExit("--tagging-model is required when --tagging-mode llm")

    try:
        return annotate_with_llm(current_memo, schema, tagging_model, timeout=generate_timeout)
    except RuntimeError as error:
        print(f"[warn] LLM tagging failed for query: {error}", file=sys.stderr)
        return {"relation_tags": []}


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
    parser.add_argument(
        "--tag-schema",
        type=Path,
        default=TAG_SCHEMA_PATH,
        help="path to tag schema JSON data",
    )
    parser.add_argument(
        "--past-tags",
        type=Path,
        default=PAST_TAGS_PATH,
        help="path to past memo tag annotations",
    )
    parser.add_argument(
        "--eval-case-tags",
        type=Path,
        default=EVAL_CASE_TAGS_PATH,
        help="path to eval case tag annotations",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=[RETRIEVAL_TEXT, RETRIEVAL_RELATION, RETRIEVAL_RRF, RETRIEVAL_RELATION_GATE, RETRIEVAL_TEXT_RERANK],
        default=RETRIEVAL_TEXT,
        help="retrieval channel to use: text-only, relation-only, text+relation RRF, relation-gated text rerank, or dense candidates plus LLM rerank",
    )
    parser.add_argument(
        "--relation-tags",
        help="comma-separated relation tags for single memo retrieval",
    )
    parser.add_argument(
        "--tagging-mode",
        choices=[TAGGING_FIXED, TAGGING_LLM],
        default=TAGGING_FIXED,
        help="how to produce query relation tags for single retrieval and eval cases",
    )
    parser.add_argument(
        "--tagging-model",
        default=DEFAULT_TAGGING_MODEL,
        help="Ollama generation model to use for LLM tagging",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="RRF constant used when --retrieval-mode rrf",
    )
    parser.add_argument(
        "--relation-candidate-k",
        type=int,
        default=DEFAULT_RELATION_CANDIDATE_K,
        help="candidate set size used when --retrieval-mode relation_gate",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help="Ollama generation model used when --retrieval-mode text_rerank",
    )
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=DEFAULT_RERANK_CANDIDATE_K,
        help="dense candidate set size used when --retrieval-mode text_rerank",
    )
    parser.add_argument(
        "--generate-timeout",
        type=int,
        default=GENERATE_TIMEOUT_SECONDS,
        help="timeout in seconds for Ollama generation requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_tag_schema(args.tag_schema)

    if args.eval:
        run_eval(args)
        return

    current_memo = read_current_memo(args)
    if not current_memo:
        raise SystemExit("현재 메모가 비어 있습니다.")

    fixed_annotation = {
        "relation_tags": parse_relation_tags(args.relation_tags, schema),
    }
    query_annotation = resolve_query_annotation(
        current_memo,
        fixed_annotation,
        args.tagging_mode,
        args.tagging_model,
        schema,
        args.generate_timeout,
    )
    past_tags = load_tag_annotations(args.past_tags, schema)
    past_memos = build_past_memo_index(
        load_past_memos(args.data),
        args.model,
        past_tags,
        args.retrieval_mode,
    )
    results, retrieval_text, rerank_summary = find_similar_memos(
        current_memo,
        past_memos,
        args.top_k,
        args.model,
        query_annotation,
        args.retrieval_mode,
        args.rrf_k,
        args.relation_candidate_k,
        args.rerank_model,
        args.rerank_candidate_k,
        args.generate_timeout,
    )
    print_results(args.model, current_memo, retrieval_text, results, rerank_summary)


if __name__ == "__main__":
    main()
