import argparse
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import main as recall


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MEMOS_PATH = DATA_DIR / "memos.json"
MEMO_ANNOTATIONS_PATH = DATA_DIR / "memo_annotations.json"
MEMORY_NODES_PATH = DATA_DIR / "memory_nodes.json"
MEMORY_EVIDENCE_PATH = DATA_DIR / "memory_evidence.json"
MEMORY_LINKS_PATH = DATA_DIR / "memory_links.json"

DEFAULT_DENSE_TOP_K = 5
DEFAULT_ATTACH_DENSE_NODE_MIN_HITS = 2
DEFAULT_NODE_TOP_K = 5
DEFAULT_EVAL_DENSE_TOP_K = 30
DEFAULT_LINK_MIN_SUPPORT = 3
DEFAULT_LINK_WINDOW_DAYS = 14
DEFAULT_NODE_SIMILARITY_THRESHOLD = 0.82
DEFAULT_NODE_EVIDENCE_LIMIT = 8
DEFAULT_CREATE_MIN_SUPPORTING_EVIDENCE = 2
DEFAULT_PENDING_PROMOTION_THRESHOLD = 3
PENDING_NODE_SIMILARITY_THRESHOLD = 0.80
NODE_EMBEDDING_MIN_SCORE = 0.70
MIN_EVIDENCE_FOR_EVOLUTION = 10

SEMANTIC_NODE_TYPES = {"pattern", "state", "question", "product_insight"}


def state_paths(state_dir: Path) -> dict[str, Path]:
    return {
        "memos": state_dir / "memos.json",
        "memo_annotations": state_dir / "memo_annotations.json",
        "memory_nodes": state_dir / "memory_nodes.json",
        "memory_evidence": state_dir / "memory_evidence.json",
        "memory_links": state_dir / "memory_links.json",
    }


def read_json(path: Path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def next_memo_id(memos: list[dict]) -> str:
    max_number = 0
    prefix = "user-memo-"

    for memo in memos:
        memo_id = memo.get("id", "")
        if not isinstance(memo_id, str) or not memo_id.startswith(prefix):
            continue
        raw_number = memo_id.removeprefix(prefix)
        if raw_number.isdigit():
            max_number = max(max_number, int(raw_number))

    return f"{prefix}{max_number + 1:03d}"





def normalize_identity_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    normalized = re.sub(r"[^\w가-힣 ,|:-]+", "", normalized)
    return normalized


def is_semantic_node(node: dict) -> bool:
    relation_tags = node.get("relation_tags")

    return (
        node.get("status") != "pending"
        and node.get("type") in SEMANTIC_NODE_TYPES
        and isinstance(node.get("title"), str)
        and bool(node.get("title", "").strip())
        and isinstance(node.get("summary"), str)
        and bool(node.get("summary", "").strip())
        and isinstance(relation_tags, list)
    )


def node_identity_text(node: dict) -> str:
    relation_tags = node.get("relation_tags", [])
    if not isinstance(relation_tags, list):
        relation_tags = []
    return "\n".join(
        [
            str(node.get("type", "")),
            str(node.get("title", "")),
            str(node.get("summary", "")),
            ",".join(sorted(tag for tag in relation_tags if isinstance(tag, str))),
        ]
    )


def build_node_identity_key(node_candidate: dict, evidence_ids: list[str]) -> str:
    relation_tags = node_candidate.get("relation_tags", [])
    if not isinstance(relation_tags, list):
        relation_tags = []

    parts = [
        normalize_identity_text(str(node_candidate.get("type", ""))),
        ",".join(sorted(normalize_identity_text(tag) for tag in relation_tags if isinstance(tag, str))),
        normalize_identity_text(str(node_candidate.get("title", ""))),
        ",".join(sorted(evidence_ids[:8])),
    ]
    return "|".join(parts)


def make_semantic_node_id(node_candidate: dict, evidence_ids: list[str]) -> str:
    node_type = normalize_identity_text(str(node_candidate.get("type", "pattern"))) or "pattern"
    digest = hashlib.sha1(
        build_node_identity_key(node_candidate, evidence_ids).encode("utf-8")
    ).hexdigest()[:8]
    return f"node-{node_type}-{digest}"


def load_state() -> tuple[list[dict], dict[str, dict], dict[str, dict], list[dict]]:
    return load_state_from_dir(DATA_DIR)


def load_state_from_dir(state_dir: Path) -> tuple[list[dict], dict[str, dict], dict[str, dict], list[dict]]:
    paths = state_paths(state_dir)
    memos = read_json(paths["memos"], [])
    memo_annotations = read_json(paths["memo_annotations"], {})
    memory_nodes = read_json(paths["memory_nodes"], {})
    memory_evidence = read_json(paths["memory_evidence"], [])

    if not isinstance(memos, list):
        raise ValueError(f"{paths['memos']} must contain a JSON list")
    if not isinstance(memo_annotations, dict):
        raise ValueError(f"{paths['memo_annotations']} must contain a JSON object")
    if not isinstance(memory_nodes, dict):
        raise ValueError(f"{paths['memory_nodes']} must contain a JSON object")
    if not isinstance(memory_evidence, list):
        raise ValueError(f"{paths['memory_evidence']} must contain a JSON list")

    return memos, memo_annotations, memory_nodes, memory_evidence


def save_state(
    memos: list[dict],
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
) -> None:
    save_state_to_dir(DATA_DIR, memos, memo_annotations, memory_nodes, memory_evidence)


def save_state_to_dir(
    state_dir: Path,
    memos: list[dict],
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
) -> None:
    paths = state_paths(state_dir)
    write_json(paths["memos"], memos)
    write_json(paths["memo_annotations"], memo_annotations)
    write_json(paths["memory_nodes"], memory_nodes)
    write_json(paths["memory_evidence"], memory_evidence)


def load_memory_links() -> list[dict]:
    return load_memory_links_from_dir(DATA_DIR)


def load_memory_links_from_dir(state_dir: Path) -> list[dict]:
    path = state_paths(state_dir)["memory_links"]
    links = read_json(path, [])
    if not isinstance(links, list):
        raise ValueError(f"{path} must contain a JSON list")
    return links


def save_memory_links(memory_links: list[dict]) -> None:
    save_memory_links_to_dir(DATA_DIR, memory_links)


def save_memory_links_to_dir(state_dir: Path, memory_links: list[dict]) -> None:
    write_json(state_paths(state_dir)["memory_links"], memory_links)




def find_evidence(
    memory_evidence: list[dict],
    node_id: str,
    memo_id: str,
) -> dict | None:
    for item in memory_evidence:
        if item.get("node_id") == node_id and item.get("memo_id") == memo_id:
            return item
    return None


def add_evidence(
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    node_id: str,
    memo_id: str,
    reason: str,
    source: str,
    weight: float,
    now: str,
) -> bool:
    node = memory_nodes.get(node_id)
    if node is None:
        return False

    existing = find_evidence(memory_evidence, node_id, memo_id)

    if existing is not None:
        existing_weight = float(existing.get("weight", 0.0))

        if weight > existing_weight:
            existing["reason"] = reason
            existing["source"] = source
            existing["weight"] = weight
            existing["updated_at"] = now

            node["last_seen_at"] = now
            node["updated_at"] = now
            node["confidence"] = min(
                0.95,
                round(0.5 + int(node.get("evidence_count", 0)) * 0.05, 2),
            )

        return False

    memory_evidence.append(
        {
            "node_id": node_id,
            "memo_id": memo_id,
            "reason": reason,
            "source": source,
            "weight": weight,
            "created_at": now,
        }
    )

    node["evidence_count"] = int(node.get("evidence_count", 0)) + 1
    node["last_seen_at"] = now
    node["updated_at"] = now
    node["confidence"] = min(0.95, round(0.5 + node["evidence_count"] * 0.05, 2))
    return True


def build_all_memos_for_dense(
    stored_memos: list[dict],
    stored_annotations: dict[str, dict],
    schema: dict,
) -> tuple[list[dict], dict[str, dict]]:
    eval_memos = recall.load_past_memos(recall.DATA_PATH)
    eval_annotations = recall.load_tag_annotations(recall.PAST_TAGS_PATH, schema)

    annotations = dict(eval_annotations)
    annotations.update(stored_annotations)

    memo_by_id = {memo["id"]: memo for memo in eval_memos}
    for memo in stored_memos:
        memo_by_id[memo["id"]] = memo

    return list(memo_by_id.values()), annotations


def dense_search(
    memo: dict,
    stored_memos: list[dict],
    stored_annotations: dict[str, dict],
    schema: dict,
    model: str,
    top_k: int,
) -> list[dict]:
    all_memos, annotations = build_all_memos_for_dense(stored_memos, stored_annotations, schema)
    candidates = [item for item in all_memos if item["id"] != memo["id"]]
    indexed = recall.build_past_memo_index(
        candidates,
        model,
        annotations,
        recall.RETRIEVAL_TEXT,
    )
    query_embedding = recall.embed_text(memo["text"], model)
    return recall.rank_by_embedding(query_embedding, indexed, "text_embedding")[:top_k]


def dense_search_against_memos(
    memo: dict,
    candidate_memos: list[dict],
    annotations: dict[str, dict],
    model: str,
    top_k: int,
) -> list[dict]:
    candidates = [item for item in candidate_memos if item["id"] != memo["id"]]
    if not candidates:
        return []

    indexed = recall.build_past_memo_index(
        candidates,
        model,
        annotations,
        recall.RETRIEVAL_TEXT,
    )
    query_embedding = recall.embed_text(memo["text"], model)
    return recall.rank_by_embedding(query_embedding, indexed, "text_embedding")[:top_k]


def count_dense_node_hits(
    dense_matches: list[dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
) -> dict[str, dict]:
    memo_to_node_ids: dict[str, list[str]] = {}
    for evidence in memory_evidence:
        memo_id = evidence.get("memo_id")
        node_id = evidence.get("node_id")
        if isinstance(memo_id, str) and isinstance(node_id, str):
            node = memory_nodes.get(node_id)
            if node is None or not is_semantic_node(node):
                continue
            memo_to_node_ids.setdefault(memo_id, []).append(node_id)

    node_hits: dict[str, dict] = {}
    for match in dense_matches:
        memo_id = match["id"]
        for node_id in memo_to_node_ids.get(memo_id, []):
            hit = node_hits.setdefault(node_id, {"count": 0, "memo_ids": [], "best_score": 0.0})
            hit["count"] += 1
            hit["memo_ids"].append(memo_id)
            hit["best_score"] = max(hit["best_score"], float(match.get("score", 0.0)))

    return node_hits


def get_node_evidence_examples(
    node_id: str,
    memory_evidence: list[dict],
    memo_by_id: dict[str, dict],
    annotations: dict[str, dict],
    limit: int,
) -> list[dict]:
    examples = []
    for evidence in memory_evidence:
        if evidence.get("node_id") != node_id:
            continue
        memo_id = evidence.get("memo_id")
        if not isinstance(memo_id, str) or memo_id not in memo_by_id:
            continue
        memo = memo_by_id[memo_id]
        examples.append(
            {
                "memo_id": memo_id,
                "date": memo.get("date"),
                "text": memo.get("text", ""),
                "relation_tags": annotations.get(memo_id, {}).get("relation_tags", []),
                "source": evidence.get("source"),
                "reason": evidence.get("reason"),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def collect_candidate_semantic_nodes(
    relation_tags: list[str],
    dense_node_hits: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    memo_by_id: dict[str, dict],
    annotations: dict[str, dict],
    limit: int,
    memo_text: str = "",
    model: str = "",
) -> list[dict]:
    query_tag_set = set(relation_tags)

    # Path 3: memo text embedding ↔ node (title+summary) embedding — exposure only
    embedding_scores: dict[str, float] = {}
    if memo_text and model:
        semantic_nodes_for_embed = [node for node in memory_nodes.values() if is_semantic_node(node)]
        if semantic_nodes_for_embed:
            memo_embedding = recall.embed_text(memo_text, model)
            indexed_nodes = [
                {
                    **node,
                    "text_embedding": recall.embed_text(
                        f"{node.get('title', '')}\n{node.get('summary', '')}", model
                    ),
                }
                for node in semantic_nodes_for_embed
            ]
            ranked = recall.rank_by_embedding(memo_embedding, indexed_nodes, "text_embedding")
            for rank_item in ranked[:DEFAULT_NODE_TOP_K]:
                score = float(rank_item.get("score", 0.0))
                if score >= NODE_EMBEDDING_MIN_SCORE:
                    embedding_scores[rank_item["id"]] = score

    candidates = []
    for node_id, node in memory_nodes.items():
        if not is_semantic_node(node):
            continue
        node_tags = set(node.get("relation_tags", []))
        dense_hit = dense_node_hits.get(node_id, {})
        tag_overlap = len(query_tag_set & node_tags)
        dense_count = int(dense_hit.get("count", 0))
        emb_score = embedding_scores.get(node_id, 0.0)
        if tag_overlap == 0 and dense_count == 0 and emb_score == 0.0:
            continue

        candidates.append(
            {
                "id": node_id,
                "type": node.get("type"),
                "title": node.get("title"),
                "summary": node.get("summary"),
                "relation_tags": node.get("relation_tags", []),
                "evidence_count": node.get("evidence_count", 0),
                "confidence": node.get("confidence", 0.0),
                "candidate_score": tag_overlap * 2 + dense_count + emb_score,
                "tag_overlap": sorted(query_tag_set & node_tags),
                "dense_memo_ids": dense_hit.get("memo_ids", []),
                "embedding_score": round(emb_score, 4),
                "evidence_examples": get_node_evidence_examples(
                    node_id,
                    memory_evidence,
                    memo_by_id,
                    annotations,
                    3,
                ),
            }
        )

    return sorted(candidates, key=lambda item: item["candidate_score"], reverse=True)[:limit]


def build_candidate_evidence(
    current_memo_id: str,
    dense_matches: list[dict],
    annotations: dict[str, dict],
    limit: int,
) -> list[dict]:
    evidence = []
    for match in dense_matches[:limit]:
        memo_id = match["id"]
        if memo_id == current_memo_id:
            continue
        evidence.append(
            {
                "memo_id": memo_id,
                "date": match.get("date"),
                "text": match.get("text", ""),
                "score": round(float(match.get("score", 0.0)), 4),
                "relation_tags": annotations.get(memo_id, match).get("relation_tags", []),
            }
        )
    return evidence


def build_node_judge_prompt(
    memo: dict,
    relation_tags: list[str],
    candidate_nodes: list[dict],
    candidate_evidence: list[dict],
    schema: dict,
) -> str:
    tag_lines = [
        f"- {tag}: {schema['relation_tags'][tag]}"
        for tag in relation_tags
        if tag in schema["relation_tags"]
    ]
    payload = {
        "memo": {
            "memo_id": memo["id"],
            "date": memo.get("date"),
            "text": memo["text"],
            "relation_tags": relation_tags,
        },
        "candidate_nodes": candidate_nodes,
        "candidate_evidence": candidate_evidence,
    }

    return "\n".join(
        [
            "You are building KEO memory_nodes.",
            "A memory_node is a self-understanding card — a specific recurring pattern or state observed across multiple memos.",
            "Use relation_tags only as routing metadata, not as node concepts.",
            "",
            "ATTACH rule:",
            "  - Attach ONLY when this memo describes the EXACT same trigger and outcome as the node's title.",
            "  - The node's core trigger must be explicitly present in this memo, not just implied or thematically related.",
            "  - If the memo is about a similar but different trigger, prefer create over attach.",
            "  - Shared topic or category alone is NOT enough. The specific situation must match.",
            "",
            "CREATE rule:",
            "  - Node title must follow 'specific trigger + recurring result' structure.",
            "    Good: '회의 후 원래 목표를 잊어버리는 상태', '메신저 확인 후 오전 흐름이 끊기는 패턴'",
            "    Bad: '맥락 분할로 인한 일정 방해' (abstract category), '흐름 끊김' (too vague)",
            "  - Title must NOT be a Korean translation of a relation_tag.",
            "  - Summary must describe the specific observed pattern in concrete terms — do not use abstract generalizations.",
            "  - Do not create a node for a one-off observation unless evidence suggests a repeatable pattern.",
            "",
            "Do not invent relation tags outside the fixed list.",
            "Do not return a node id. The system will create ids.",
            "Return JSON only.",
            'JSON schema: {"decision":"attach|create|attach_and_create|ignore","attach_node_ids":["node-id"],"create_node":{"type":"pattern|state|question|product_insight","title":"짧은 한국어 제목","summary":"근거 기반 한국어 요약","relation_tags":["tag"],"confidence":0.0},"evidence_memo_ids":["memo-id"],"evidence_reason":"짧은 한국어 이유"}',
            "",
            "[Allowed relation tags for this memo]",
            *tag_lines,
            "",
            "[Input JSON]",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def sanitize_node_judgement(payload: dict, schema: dict, candidate_node_ids: set[str], evidence_ids: set[str]) -> dict:
    decision = payload.get("decision")
    if decision not in {"attach", "create", "attach_and_create", "ignore"}:
        decision = "ignore"

    attach_node_ids = payload.get("attach_node_ids", [])
    if not isinstance(attach_node_ids, list):
        attach_node_ids = []
    attach_node_ids = [
        node_id
        for node_id in attach_node_ids
        if isinstance(node_id, str) and node_id in candidate_node_ids
    ]

    evidence_memo_ids = payload.get("evidence_memo_ids", [])
    if not isinstance(evidence_memo_ids, list):
        evidence_memo_ids = []
    evidence_memo_ids = [
        memo_id
        for memo_id in evidence_memo_ids
        if isinstance(memo_id, str) and memo_id in evidence_ids
    ]

    create_node = payload.get("create_node")
    if not isinstance(create_node, dict) or decision not in {"create", "attach_and_create"}:
        create_node = None
    else:
        node_type = create_node.get("type")
        if node_type not in SEMANTIC_NODE_TYPES:
            node_type = "pattern"

        title = create_node.get("title")
        summary = create_node.get("summary")
        relation_tags = create_node.get("relation_tags", [])
        confidence = create_node.get("confidence", 0.5)
        if not isinstance(title, str) or not title.strip():
            create_node = None
        elif not isinstance(summary, str) or not summary.strip():
            create_node = None
        elif not isinstance(relation_tags, list):
            create_node = None
        else:
            known_tags = schema["relation_tags"]
            clean_tags = [
                tag
                for tag in relation_tags
                if isinstance(tag, str) and tag in known_tags
            ]
            if not clean_tags:
                create_node = None
            else:
                if not isinstance(confidence, (int, float)):
                    confidence = 0.5
                create_node = {
                    "type": node_type,
                    "title": title.strip()[:80],
                    "summary": summary.strip()[:500],
                    "relation_tags": sorted(set(clean_tags)),
                    "confidence": max(0.0, min(0.95, float(confidence))),
                }

    reason = payload.get("evidence_reason")
    if not isinstance(reason, str):
        reason = ""

    return {
        "decision": decision,
        "attach_node_ids": attach_node_ids,
        "create_node": create_node,
        "evidence_memo_ids": evidence_memo_ids,
        "evidence_reason": reason.strip()[:500],
    }


def judge_memory_node_with_llm(
    memo: dict,
    relation_tags: list[str],
    candidate_nodes: list[dict],
    candidate_evidence: list[dict],
    schema: dict,
    model: str,
    timeout: int,
) -> dict:
    prompt = build_node_judge_prompt(memo, relation_tags, candidate_nodes, candidate_evidence, schema)
    response = recall.generate_text(prompt, model, timeout=timeout, response_format="json")
    try:
        payload = recall.extract_json_object(response)
    except (ValueError, Exception) as error:
        raise RuntimeError(f"LLM node judge returned invalid JSON: {error}") from error
    candidate_node_ids = {node["id"] for node in candidate_nodes}
    evidence_ids = {item["memo_id"] for item in candidate_evidence}
    evidence_ids.add(memo["id"])
    judgement = sanitize_node_judgement(payload, schema, candidate_node_ids, evidence_ids)
    create_node = judgement.get("create_node")
    if create_node is not None:
        create_node["relation_tags"] = sorted(
            set(create_node.get("relation_tags", [])) | set(relation_tags)
        )[:3]
    return judgement


def find_similar_semantic_node(
    node_candidate: dict,
    memory_nodes: dict[str, dict],
    model: str,
    threshold: float,
) -> tuple[str | None, float]:
    semantic_nodes = [
        node
        for node in memory_nodes.values()
        if is_semantic_node(node)
    ]
    if not semantic_nodes:
        return None, 0.0

    query_embedding = recall.embed_text(node_identity_text(node_candidate), model)
    indexed_nodes = [
        {
            **node,
            "text_embedding": recall.embed_text(node_identity_text(node), model),
        }
        for node in semantic_nodes
    ]
    ranked = recall.rank_by_embedding(query_embedding, indexed_nodes, "text_embedding")
    if not ranked:
        return None, 0.0

    best = ranked[0]
    score = float(best.get("score", 0.0))
    if score >= threshold:
        return best["id"], score
    return None, score


def create_semantic_node(
    node_candidate: dict,
    evidence_ids: list[str],
    memory_nodes: dict[str, dict],
    now: str,
) -> dict:
    node_id = make_semantic_node_id(node_candidate, evidence_ids)
    suffix = 1
    base_node_id = node_id
    while node_id in memory_nodes:
        suffix += 1
        node_id = f"{base_node_id}-{suffix}"

    node = {
        "id": node_id,
        "source": "llm",
        "type": node_candidate["type"],
        "title": node_candidate["title"],
        "summary": node_candidate["summary"],
        "relation_tags": node_candidate["relation_tags"],
        "representative_evidence_ids": sorted(set(evidence_ids)),
        "identity_key": build_node_identity_key(node_candidate, evidence_ids),
        "evidence_count": 0,
        "confidence": node_candidate.get("confidence", 0.5),
        "last_seen_at": None,
        "created_at": now,
        "updated_at": now,
    }
    memory_nodes[node_id] = node
    return node


def create_pending_node(
    node_candidate: dict,
    initial_evidence_ids: list[str],
    memory_nodes: dict[str, dict],
    now: str,
) -> dict:
    node_id = make_semantic_node_id(node_candidate, initial_evidence_ids)
    suffix = 1
    base_node_id = node_id
    while node_id in memory_nodes:
        suffix += 1
        node_id = f"{base_node_id}-{suffix}"

    node = {
        "id": node_id,
        "source": "llm",
        "status": "pending",
        "type": node_candidate["type"],
        "title": node_candidate["title"],
        "summary": node_candidate["summary"],
        "relation_tags": node_candidate["relation_tags"],
        "pending_evidence_ids": list(dict.fromkeys(initial_evidence_ids)),
        "identity_key": build_node_identity_key(node_candidate, initial_evidence_ids),
        "evidence_count": 0,
        "confidence": node_candidate.get("confidence", 0.5),
        "last_seen_at": None,
        "created_at": now,
        "updated_at": now,
    }
    memory_nodes[node_id] = node
    return node


def find_similar_pending_node(
    node_candidate: dict,
    memory_nodes: dict[str, dict],
    model: str,
    threshold: float,
) -> tuple[str | None, float]:
    pending_nodes = [
        node
        for node in memory_nodes.values()
        if node.get("status") == "pending" and node.get("type") in SEMANTIC_NODE_TYPES
    ]
    if not pending_nodes:
        return None, 0.0

    query_embedding = recall.embed_text(node_identity_text(node_candidate), model)
    indexed = [
        {
            **node,
            "text_embedding": recall.embed_text(node_identity_text(node), model),
        }
        for node in pending_nodes
    ]
    ranked = recall.rank_by_embedding(query_embedding, indexed, "text_embedding")
    if not ranked:
        return None, 0.0

    best = ranked[0]
    score = float(best.get("score", 0.0))
    if score >= threshold:
        return best["id"], score
    return None, score


def promote_pending_node(
    pending_node: dict,
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    now: str,
) -> None:
    evidence_ids = pending_node.pop("pending_evidence_ids", [])
    pending_node.pop("status", None)
    pending_node["representative_evidence_ids"] = sorted(set(evidence_ids))
    pending_node["last_seen_at"] = now
    pending_node["updated_at"] = now

    for memo_id in evidence_ids:
        add_evidence(
            memory_nodes,
            memory_evidence,
            pending_node["id"],
            memo_id,
            "보류 패턴 반복 확인으로 승격",
            "llm",
            0.75,
            now,
        )


def build_node_evolve_prompt(node: dict, evidence_memos: list[dict], schema: dict) -> str:
    allowed_tags = list(schema.get("relation_tags", {}).keys())
    evidence_list = [
        {"memo_id": m["id"], "date": m.get("date"), "text": m["text"]}
        for m in evidence_memos
    ]
    payload = {
        "node": {
            "id": node["id"],
            "type": node["type"],
            "title": node["title"],
            "summary": node["summary"],
            "relation_tags": node.get("relation_tags", []),
        },
        "evidence_memos": evidence_list,
    }
    return "\n".join(
        [
            "You are reviewing a KEO memory_node that has accumulated many evidence memos.",
            "Decide if this node should be split into more specific sub-patterns, refined (title/summary/tags updated), or confirmed as-is.",
            "",
            "SPLIT rules:",
            "  - For each evidence memo, ask: does this memo's text explicitly mention the trigger in the node title? If not, it does not belong to this node.",
            "  - If more than 30% of memos do NOT explicitly match the node's core trigger, you MUST split.",
            "  - You MUST produce at least 2 split_nodes. A split with only 1 node is invalid — use refine instead.",
            "  - Each split_node title must follow 'specific trigger + recurring result' structure. Do NOT copy the original title.",
            "  - Each split_node summary must be written fresh based on its assigned memos. Do NOT copy or paraphrase the original node's summary.",
            "  - Each split_node's relation_tags: choose max 3 tags from the allowed list. Select freely — do NOT copy the original node's tags.",
            "  - Every evidence_memo_id must be assigned to exactly one split_node.",
            "",
            "REFINE rules:",
            "  - Choose when title/summary/tags don't accurately represent the evidence, but all memos belong to one pattern.",
            "  - Rewrite title in 'specific trigger + recurring result' structure.",
            "  - Write summary fresh — describe the concrete observed pattern. Do NOT copy or paraphrase the original summary.",
            "  - Update relation_tags: max 3 tags, select freely from the full allowed list.",
            "",
            "CONFIRM: choose ONLY when the node title's specific trigger is explicitly present in ALL evidence memos.",
            "Return JSON only.",
            'JSON schema: {"action":"split|refine|confirm","split_nodes":[{"type":"pattern|state|question|product_insight","title":"한국어 제목","summary":"한국어 요약","relation_tags":["tag"],"evidence_memo_ids":["memo-id"]}],"refined_node":{"type":"pattern|state|question|product_insight","title":"한국어 제목","summary":"한국어 요약","relation_tags":["tag"]},"reason":"짧은 한국어 이유"}',
            "",
            f"[Allowed relation tags]: {', '.join(allowed_tags)}",
            "",
            "[Input JSON]",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
    )


def apply_node_evolution(
    node: dict,
    action: str,
    split_nodes: list[dict],
    refined_node: dict | None,
    memo_by_id: dict[str, dict],
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    schema: dict,
    model: str,
    similarity_threshold: float,
    now: str,
) -> list[dict]:
    original_id = node["id"]
    decisions = []

    if action == "confirm":
        decisions.append({"node_id": original_id, "action": "confirm"})
        return decisions

    if action == "refine" and refined_node:
        known_tags = set(schema.get("relation_tags", {}).keys())
        clean_tags = [t for t in refined_node.get("relation_tags", []) if t in known_tags]
        if not clean_tags:
            decisions.append({"node_id": original_id, "action": "confirm", "reason": "refine tags invalid"})
            return decisions
        node["title"] = refined_node.get("title", node["title"])
        node["summary"] = refined_node.get("summary", node["summary"])
        node["relation_tags"] = clean_tags[:3]
        node["type"] = refined_node.get("type", node["type"])
        node["updated_at"] = now
        decisions.append({"node_id": original_id, "action": "refined"})
        return decisions

    if action == "split" and split_nodes:
        known_tags = set(schema.get("relation_tags", {}).keys())
        original_evidence_ids = {
            e["memo_id"] for e in memory_evidence if e.get("node_id") == original_id
        }

        # split_nodes must have ≥2 valid candidates — otherwise treat as refine/confirm
        valid_candidates = []
        for sc in split_nodes:
            if not isinstance(sc, dict):
                continue
            raw_tags = sc.get("relation_tags", [])
            if not isinstance(raw_tags, list):
                raw_tags = []
            clean_tags = [t for t in raw_tags if isinstance(t, str) and t in known_tags][:3]
            memo_ids = sc.get("evidence_memo_ids", [])
            if not isinstance(memo_ids, list):
                memo_ids = []
            mids = [m for m in memo_ids if isinstance(m, str) and m in original_evidence_ids]
            if clean_tags and mids:
                sc["relation_tags"] = clean_tags
                valid_candidates.append(sc)

        if len(valid_candidates) < 2:
            decisions.append({"node_id": original_id, "action": "confirm",
                               "reason": "split candidates < 2, skipped"})
            return decisions

        assigned_memo_ids: set[str] = set()

        for split_candidate in valid_candidates:
            evidence_memo_ids = [
                mid for mid in split_candidate.get("evidence_memo_ids", [])
                if mid in original_evidence_ids
            ]

            similar_id, sim = find_similar_semantic_node(
                split_candidate, memory_nodes, model, similarity_threshold
            )
            if similar_id and similar_id != original_id:
                for mid in evidence_memo_ids:
                    add_evidence(memory_nodes, memory_evidence, similar_id, mid,
                                 f"split에서 기존 유사 node로 재배분 (유사도 {sim:.4f})",
                                 "evolve", 0.75, now)
                assigned_memo_ids.update(evidence_memo_ids)
                decisions.append({"node_id": similar_id, "action": "split_merged_existing",
                                   "evidence_memo_ids": evidence_memo_ids})
            else:
                new_node = create_semantic_node(split_candidate, evidence_memo_ids, memory_nodes, now)
                for mid in evidence_memo_ids:
                    add_evidence(memory_nodes, memory_evidence, new_node["id"], mid,
                                 "split으로 새 node 생성", "evolve", 0.75, now)
                assigned_memo_ids.update(evidence_memo_ids)
                decisions.append({"node_id": new_node["id"], "action": "split_created",
                                   "evidence_memo_ids": evidence_memo_ids})

        if assigned_memo_ids:
            # remove original node's evidence and the node itself
            memory_evidence[:] = [
                e for e in memory_evidence
                if not (e.get("node_id") == original_id and e.get("memo_id") in assigned_memo_ids)
            ]
            remaining = [
                e for e in memory_evidence if e.get("node_id") == original_id
            ]
            if not remaining:
                del memory_nodes[original_id]
                decisions.append({"node_id": original_id, "action": "split_removed_original"})
            else:
                # update original node's tags to reflect only remaining evidence memos
                remaining_memo_ids = {e["memo_id"] for e in remaining}
                remaining_tags: set[str] = set()
                for mid in remaining_memo_ids:
                    ann_tags = memo_annotations.get(mid, {}).get("relation_tags", [])
                    remaining_tags.update(t for t in ann_tags if t in known_tags)
                if remaining_tags:
                    node["relation_tags"] = sorted(remaining_tags)[:3]
                node["evidence_count"] = len(remaining)
                node["updated_at"] = now
                decisions.append({"node_id": original_id, "action": "split_original_kept",
                                   "remaining_evidence": len(remaining)})

    return decisions


def evolve_single_node(
    node: dict,
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    memo_by_id: dict[str, dict],
    memo_annotations: dict[str, dict],
    schema: dict,
    model: str,
    node_judge_model: str,
    similarity_threshold: float,
    generate_timeout: int,
    now: str,
) -> list[dict]:
    node_id = node["id"]
    evidence_memo_ids = [
        e["memo_id"] for e in memory_evidence
        if e.get("node_id") == node_id and isinstance(e.get("memo_id"), str)
    ]
    evidence_memos = [memo_by_id[mid] for mid in evidence_memo_ids if mid in memo_by_id]
    if not evidence_memos:
        return []

    prompt = build_node_evolve_prompt(node, evidence_memos, schema)
    try:
        response = recall.generate_text(
            prompt, node_judge_model, timeout=generate_timeout, response_format="json"
        )
        payload = recall.extract_json_object(response)
    except Exception:
        return []

    action = payload.get("action")
    if action not in {"split", "refine", "confirm"}:
        return []

    decisions = apply_node_evolution(
        node,
        action,
        payload.get("split_nodes") or [],
        payload.get("refined_node"),
        memo_by_id,
        memo_annotations,
        memory_nodes,
        memory_evidence,
        schema,
        model,
        similarity_threshold,
        now,
    )

    if node_id in memory_nodes:
        memory_nodes[node_id]["last_evolved_evidence_count"] = memory_nodes[node_id].get("evidence_count", 0)

    return decisions


def should_evolve_node(node: dict) -> bool:
    ec = node.get("evidence_count", 0)
    last_ec = node.get("last_evolved_evidence_count", 0)
    return ec >= MIN_EVIDENCE_FOR_EVOLUTION and (ec - last_ec) >= MIN_EVIDENCE_FOR_EVOLUTION


def evolve_semantic_nodes(
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    memo_by_id: dict[str, dict],
    memo_annotations: dict[str, dict],
    schema: dict,
    model: str,
    node_judge_model: str,
    similarity_threshold: float,
    generate_timeout: int,
    now: str,
) -> list[dict]:
    all_decisions = []
    candidates = [
        node for node in list(memory_nodes.values())
        if is_semantic_node(node) and should_evolve_node(node)
    ]
    for node in candidates:
        decisions = evolve_single_node(
            node, memory_nodes, memory_evidence, memo_by_id, memo_annotations,
            schema, model, node_judge_model, similarity_threshold, generate_timeout, now,
        )
        all_decisions.extend(decisions)
    return all_decisions


def merge_similar_nodes(
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    model: str,
    similarity_threshold: float,
    now: str,
) -> list[dict]:
    active = [n for n in memory_nodes.values() if is_semantic_node(n)]
    if len(active) < 2:
        return []

    ev_count: dict[str, int] = defaultdict(int)
    for e in memory_evidence:
        nid = e.get("node_id")
        if nid:
            ev_count[nid] += 1

    indexed = [
        {**node, "text_embedding": recall.embed_text(node_identity_text(node), model)}
        for node in active
    ]

    decisions = []
    merged_ids: set[str] = set()

    for i, node_a in enumerate(indexed):
        if node_a["id"] in merged_ids:
            continue
        others = [n for n in indexed[i + 1:] if n["id"] not in merged_ids]
        if not others:
            continue
        ranked = recall.rank_by_embedding(node_a["text_embedding"], others, "text_embedding")
        for match in ranked:
            if float(match.get("score", 0)) < similarity_threshold:
                break
            victim_id = match["id"]
            if victim_id in merged_ids:
                continue

            if ev_count[node_a["id"]] >= ev_count[victim_id]:
                survivor_id, loser_id = node_a["id"], victim_id
            else:
                survivor_id, loser_id = victim_id, node_a["id"]

            for e in memory_evidence:
                if e.get("node_id") == loser_id:
                    e["node_id"] = survivor_id

            # deduplicate evidence for survivor
            seen_mids: set[str] = set()
            deduped = []
            for e in memory_evidence:
                if e.get("node_id") == survivor_id:
                    mid = e.get("memo_id", "")
                    if mid in seen_mids:
                        continue
                    seen_mids.add(mid)
                deduped.append(e)
            memory_evidence[:] = deduped

            merged_ids.add(loser_id)
            if loser_id in memory_nodes:
                del memory_nodes[loser_id]

            survivor = memory_nodes.get(survivor_id)
            if survivor:
                survivor["evidence_count"] = sum(
                    1 for e in memory_evidence if e.get("node_id") == survivor_id
                )
                survivor["updated_at"] = now

            decisions.append({
                "action": "merged",
                "survivor_id": survivor_id,
                "loser_id": loser_id,
                "similarity": round(float(match.get("score", 0)), 4),
            })
            break

    return decisions


def apply_node_judgement(
    judgement: dict,
    memo: dict,
    relation_tags: list[str],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    model: str,
    similarity_threshold: float,
    now: str,
) -> list[dict]:
    decisions = []
    evidence_reason = judgement.get("evidence_reason") or "LLM node judge 판단"
    evidence_ids = [memo["id"], *judgement.get("evidence_memo_ids", [])]
    evidence_ids = list(dict.fromkeys(evidence_ids))
    memo_tag_set = set(relation_tags)

    for node_id in judgement.get("attach_node_ids", []):
        if node_id not in memory_nodes:
            continue
        if not is_semantic_node(memory_nodes[node_id]):
            continue
        node_tags = set(memory_nodes[node_id].get("relation_tags", []))
        if not (memo_tag_set & node_tags):
            decisions.append({
                "node_id": node_id,
                "source": "llm",
                "decision": "attach_rejected_no_tag_overlap",
                "reason": "메모 relation_tags와 node relation_tags 겹침 없음",
                "weight": 0.0,
            })
            continue

        attached_any = False
        for evidence_memo_id in evidence_ids:
            added = add_evidence(
                memory_nodes,
                memory_evidence,
                node_id,
                evidence_memo_id,
                evidence_reason,
                "llm",
                0.85 if evidence_memo_id == memo["id"] else 0.65,
                now,
            )
            attached_any = attached_any or added

        decisions.append(
            {
                "node_id": node_id,
                "source": "llm",
                "decision": "attach" if attached_any else "already_attached",
                "reason": evidence_reason,
                "weight": 0.85,
                "evidence_memo_ids": evidence_ids,
            }
        )

    create_node = judgement.get("create_node")
    if create_node is None:
        return decisions

    similar_node_id, similarity = find_similar_semantic_node(
        create_node,
        memory_nodes,
        model,
        similarity_threshold,
    )
    if similar_node_id is not None:
        added = False
        for evidence_memo_id in evidence_ids:
            added = add_evidence(
                memory_nodes,
                memory_evidence,
                similar_node_id,
                evidence_memo_id,
                f"{evidence_reason} 기존 semantic node와 유사도 {similarity:.4f}로 병합",
                "llm",
                0.85 if evidence_memo_id == memo["id"] else 0.65,
                now,
            ) or added
        decisions.append(
            {
                "node_id": similar_node_id,
                "source": "llm",
                "decision": "attach_existing_similar",
                "reason": f"{evidence_reason} 기존 node와 유사도 {similarity:.4f}",
                "weight": 0.85,
                "evidence_memo_ids": evidence_ids,
            }
        )
        return decisions

    # Create guard: require supporting evidence from other memos
    other_evidence = [mid for mid in evidence_ids if mid != memo["id"]]
    if len(other_evidence) < DEFAULT_CREATE_MIN_SUPPORTING_EVIDENCE:
        similar_pending_id, pending_sim = find_similar_pending_node(
            create_node, memory_nodes, model, PENDING_NODE_SIMILARITY_THRESHOLD
        )
        if similar_pending_id is not None:
            pending_node = memory_nodes[similar_pending_id]
            existing = pending_node.get("pending_evidence_ids", [])
            merged = list(dict.fromkeys([*existing, *evidence_ids]))
            pending_node["pending_evidence_ids"] = merged
            pending_node["updated_at"] = now
            if len(merged) >= DEFAULT_PENDING_PROMOTION_THRESHOLD:
                promote_pending_node(pending_node, memory_nodes, memory_evidence, now)
                decisions.append(
                    {
                        "node_id": pending_node["id"],
                        "source": "llm",
                        "decision": "promoted_from_pending",
                        "reason": f"보류 패턴 반복 확인 (유사도 {pending_sim:.4f}), evidence {len(merged)}개",
                        "weight": 0.75,
                        "evidence_memo_ids": merged,
                    }
                )
            else:
                decisions.append(
                    {
                        "node_id": similar_pending_id,
                        "source": "llm",
                        "decision": "pending_merged",
                        "reason": f"기존 보류 패턴에 추가 (유사도 {pending_sim:.4f}), evidence {len(merged)}개",
                        "weight": 0.0,
                        "evidence_memo_ids": merged,
                    }
                )
        else:
            pending_node = create_pending_node(create_node, evidence_ids, memory_nodes, now)
            decisions.append(
                {
                    "node_id": pending_node["id"],
                    "source": "llm",
                    "decision": "pending_created",
                    "reason": f"반복 근거 부족 (supporting evidence {len(other_evidence)}개), 보류 패턴으로 저장",
                    "weight": 0.0,
                    "evidence_memo_ids": evidence_ids,
                }
            )
        return decisions

    node = create_semantic_node(create_node, evidence_ids, memory_nodes, now)
    for evidence_memo_id in evidence_ids:
        add_evidence(
            memory_nodes,
            memory_evidence,
            node["id"],
            evidence_memo_id,
            evidence_reason,
            "llm",
            0.85 if evidence_memo_id == memo["id"] else 0.65,
            now,
        )
    decisions.append(
        {
            "node_id": node["id"],
            "source": "llm",
            "decision": "create",
            "reason": evidence_reason,
            "weight": 0.85,
            "created_node": node,
            "evidence_memo_ids": evidence_ids,
        }
    )
    return decisions


def build_fixture_memory_graph(
    past_memos: list[dict],
    annotations: dict[str, dict],
    schema: dict,
) -> tuple[dict[str, dict], list[dict]]:
    # Legacy fixture seed graph is disabled.
    # Evaluate semantic graph state via `build-eval-graph` + `--use-state`.
    return {}, []


def score_node_candidates(
    query_tags: list[str],
    dense_matches: list[dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    tag_weight: float,
    dense_weight: float,
) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    query_tag_set = set(query_tags)

    for node_id, node in memory_nodes.items():
        if not is_semantic_node(node):
            continue
        node_tags = set(node.get("relation_tags", []))
        overlap = sorted(query_tag_set & node_tags)
        if not overlap:
            continue
        candidate = candidates.setdefault(
            node_id,
            {"score": 0.0, "tag_hits": [], "dense_memo_ids": [], "dense_hits": 0},
        )
        candidate["score"] += len(overlap) * tag_weight
        candidate["tag_hits"].extend(overlap)

    memo_to_node_ids: dict[str, list[str]] = {}
    for evidence in memory_evidence:
        memo_id = evidence.get("memo_id")
        node_id = evidence.get("node_id")
        if isinstance(memo_id, str) and isinstance(node_id, str):
            node = memory_nodes.get(node_id)
            if node is None or not is_semantic_node(node):
                continue
            memo_to_node_ids.setdefault(memo_id, []).append(node_id)

    for rank, match in enumerate(dense_matches, start=1):
        rank_score = dense_weight / rank
        for node_id in memo_to_node_ids.get(match["id"], []):
            if node_id not in memory_nodes:
                continue
            if not is_semantic_node(memory_nodes[node_id]):
                continue
            candidate = candidates.setdefault(
                node_id,
                {"score": 0.0, "tag_hits": [], "dense_memo_ids": [], "dense_hits": 0},
            )
            candidate["score"] += rank_score
            candidate["dense_hits"] += 1
            candidate["dense_memo_ids"].append(match["id"])

    return candidates


def rank_memory_evidence_memos(
    node_candidates: dict[str, dict],
    memory_evidence: list[dict],
    memo_by_id: dict[str, dict],
    annotations: dict[str, dict],
    dense_matches: list[dict],
    query_tags: list[str],
    node_top_k: int,
    top_k: int,
    include_dense_fallback: bool,
) -> list[dict]:
    selected_node_ids = [
        node_id
        for node_id, _ in sorted(
            node_candidates.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:node_top_k]
    ]
    dense_scores = {match["id"]: float(match["score"]) for match in dense_matches}
    query_tag_set = set(query_tags)
    memo_scores: dict[str, float] = {}
    memo_sources: dict[str, set[str]] = {}

    for evidence in memory_evidence:
        node_id = evidence.get("node_id")
        memo_id = evidence.get("memo_id")
        if node_id not in selected_node_ids or memo_id not in memo_by_id:
            continue

        node_score = float(node_candidates[node_id]["score"])
        memo_tags = set(annotations.get(memo_id, {}).get("relation_tags", []))
        tag_overlap = len(query_tag_set & memo_tags)
        dense_score = dense_scores.get(memo_id, 0.0)

        score = node_score * 0.35 + tag_overlap * 0.3 + dense_score * 2.0
        memo_scores[memo_id] = memo_scores.get(memo_id, 0.0) + score
        memo_sources.setdefault(memo_id, set()).add(node_id)

    if include_dense_fallback:
        for rank, match in enumerate(dense_matches[:top_k], start=1):
            memo_id = match["id"]
            memo_scores[memo_id] = memo_scores.get(memo_id, 0.0) + float(match["score"]) * 1.5 + 0.2 / rank
            memo_sources.setdefault(memo_id, set()).add("dense")

    ranked_memo_ids = sorted(memo_scores, key=lambda memo_id: memo_scores[memo_id], reverse=True)
    results = []
    for memo_id in ranked_memo_ids[:top_k]:
        memo = memo_by_id[memo_id]
        results.append(
            {
                **memo,
                "score": memo_scores[memo_id],
                "relation_tags": annotations.get(memo_id, {}).get("relation_tags", []),
                "memory_sources": sorted(memo_sources[memo_id]),
            }
        )

    return results


def classify_relation_tags(args: argparse.Namespace, schema: dict) -> list[str]:
    if args.relation_tags:
        return recall.parse_relation_tags(args.relation_tags, schema)

    if args.tagging_mode == "llm":
        annotation = recall.annotate_with_llm(
            args.memo,
            schema,
            args.tagging_model,
            timeout=args.generate_timeout,
        )
        return annotation["relation_tags"]

    return []


def process_memo_into_graph(
    memo: dict,
    relation_tags: list[str],
    memos: list[dict],
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    schema: dict,
    model: str,
    dense_top_k: int,
    attach_dense_node_min_hits: int,
    use_llm_node_judge: bool,
    node_judge_model: str,
    node_candidate_limit: int,
    node_evidence_limit: int,
    node_similarity_threshold: float,
    generate_timeout: int,
    now: str,
    use_evolve: bool = False,
) -> dict:
    if any(item["id"] == memo["id"] for item in memos):
        raise ValueError(f"memo already exists: {memo['id']}")

    previous_memos = list(memos)
    memos.append(memo)
    memo_annotations[memo["id"]] = {
        "relation_tags": relation_tags,
        "short_summary": memo["text"][:80],
    }

    linked_nodes: set[str] = set()
    connection_decisions = []
    dense_matches = dense_search_against_memos(
        memo,
        previous_memos,
        memo_annotations,
        model,
        dense_top_k,
    )
    dense_node_hits = count_dense_node_hits(dense_matches, memory_nodes, memory_evidence)

    if use_llm_node_judge:
        memo_by_id = {item["id"]: item for item in memos}
        candidate_nodes = collect_candidate_semantic_nodes(
            relation_tags,
            dense_node_hits,
            memory_nodes,
            memory_evidence,
            memo_by_id,
            memo_annotations,
            node_candidate_limit,
            memo_text=memo["text"],
            model=model,
        )
        candidate_evidence = build_candidate_evidence(
            memo["id"],
            dense_matches,
            memo_annotations,
            node_evidence_limit,
        )
        try:
            node_judgement = judge_memory_node_with_llm(
                memo,
                relation_tags,
                candidate_nodes,
                candidate_evidence,
                schema,
                node_judge_model,
                generate_timeout,
            )
        except RuntimeError as error:
            connection_decisions.append(
                {
                    "node_id": None,
                    "source": "llm",
                    "decision": "ignore",
                    "reason": f"LLM node judge failed: {error}",
                    "weight": 0.0,
                }
            )
        else:
            llm_decisions = apply_node_judgement(
                node_judgement,
                memo,
                relation_tags,
                memory_nodes,
                memory_evidence,
                model,
                node_similarity_threshold,
                now,
            )
            for decision in llm_decisions:
                node_id = decision.get("node_id")
                if isinstance(node_id, str):
                    linked_nodes.add(node_id)
            connection_decisions.extend(llm_decisions)

            if use_evolve:
                memo_by_id = {item["id"]: item for item in memos}
                affected_node_ids = {
                    d["node_id"] for d in llm_decisions
                    if d.get("decision") in {"attach", "attach_existing_similar", "promoted_from_pending"}
                    and isinstance(d.get("node_id"), str)
                }
                for nid in affected_node_ids:
                    node = memory_nodes.get(nid)
                    if node and is_semantic_node(node) and should_evolve_node(node):
                        evolve_decisions = evolve_single_node(
                            node, memory_nodes, memory_evidence, memo_by_id, memo_annotations,
                            schema, model, node_judge_model, node_similarity_threshold,
                            generate_timeout, now,
                        )
                        connection_decisions.extend(evolve_decisions)

    return {
        "memo": memo,
        "annotation": memo_annotations[memo["id"]],
        "connection_decisions": connection_decisions,
        "linked_node_ids": sorted(
            linked_nodes
        ),
        "dense_matches": dense_matches,
        "dense_node_candidates": dense_node_hits,
    }


def ingest(args: argparse.Namespace) -> None:
    schema = recall.load_tag_schema(args.tag_schema)
    memos, memo_annotations, memory_nodes, memory_evidence = load_state()
    now = utc_now()

    memo_id = args.memo_id or next_memo_id(memos)
    relation_tags = classify_relation_tags(args, schema)
    memo = {
        "id": memo_id,
        "date": args.date or now[:10],
        "text": args.memo.strip(),
        "created_at": now,
    }
    try:
        processed = process_memo_into_graph(
            memo,
            relation_tags,
            memos,
            memo_annotations,
            memory_nodes,
            memory_evidence,
            schema,
            args.model,
            args.dense_top_k,
            args.attach_dense_node_min_hits,
            args.use_llm_node_judge,
            args.node_judge_model,
            args.node_candidate_limit,
            args.node_evidence_limit,
            args.node_similarity_threshold,
            args.generate_timeout,
            now,
            use_evolve=not getattr(args, "skip_evolve", False),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    save_state(memos, memo_annotations, memory_nodes, memory_evidence)

    result = {
        "memo": memo,
        "annotation": memo_annotations[memo_id],
        "connection_decisions": processed["connection_decisions"],
        "linked_nodes": [memory_nodes[node_id] for node_id in processed["linked_node_ids"]],
        "dense_matches": [
            {
                "id": match["id"],
                "score": round(float(match["score"]), 4),
                "date": match.get("date"),
                "relation_tags": match.get("relation_tags", []),
                "text": match["text"],
            }
            for match in processed["dense_matches"]
        ],
        "dense_node_candidates": processed["dense_node_candidates"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def show_nodes(args: argparse.Namespace) -> None:
    _, _, memory_nodes, _ = load_state()
    nodes = sorted(
        memory_nodes.values(),
        key=lambda node: (node.get("evidence_count", 0), node.get("updated_at", "")),
        reverse=True,
    )
    print(json.dumps(nodes[: args.limit], ensure_ascii=False, indent=2))


def seed_from_eval(args: argparse.Namespace) -> None:
    existing_memos, _, _, _ = load_state()
    if existing_memos and not args.force:
        raise SystemExit("data already exists. Re-run with --force to replace it.")

    schema = recall.load_tag_schema(args.tag_schema)
    past_memos = recall.load_past_memos(args.data)
    past_tags = recall.load_tag_annotations(args.past_tags, schema)
    memory_nodes, memory_evidence = build_fixture_memory_graph(past_memos, past_tags, schema)

    memo_annotations = {}
    for memo in past_memos:
        annotation = past_tags.get(memo["id"], {"relation_tags": []})
        memo_annotations[memo["id"]] = {
            "relation_tags": annotation["relation_tags"],
            "short_summary": memo["text"][:80],
        }

    memos = [
        {
            **memo,
            "created_at": memo.get("date"),
        }
        for memo in past_memos
    ]
    save_state(memos, memo_annotations, memory_nodes, memory_evidence)
    save_memory_links([])

    print(
        json.dumps(
            {
                "memos": len(memos),
                "memo_annotations": len(memo_annotations),
                "memory_nodes": len(memory_nodes),
                "memory_evidence": len(memory_evidence),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_eval_graph(args: argparse.Namespace) -> None:
    state_dir = args.state_dir
    paths = state_paths(state_dir)
    if paths["memos"].exists() and not args.force:
        raise SystemExit(f"{state_dir} already exists. Re-run with --force to replace eval graph state.")

    schema = recall.load_tag_schema(args.tag_schema)
    past_memos = recall.load_past_memos(args.data)
    past_tags = recall.load_tag_annotations(args.past_tags, schema)
    memos: list[dict] = []
    memo_annotations: dict[str, dict] = {}
    memory_nodes: dict[str, dict] = {}
    memory_evidence: list[dict] = []
    limit = args.limit or len(past_memos)
    now = utc_now()

    for index, source_memo in enumerate(past_memos[:limit], start=1):
        if args.summary_only:
            print(f"[build-eval-graph] {index}/{limit} {source_memo['id']}", flush=True)

        relation_tags = past_tags.get(source_memo["id"], {}).get("relation_tags", [])
        memo = {
            **source_memo,
            "created_at": source_memo.get("date", now),
        }
        process_memo_into_graph(
            memo,
            relation_tags,
            memos,
            memo_annotations,
            memory_nodes,
            memory_evidence,
            schema,
            args.model,
            args.dense_top_k,
            args.attach_dense_node_min_hits,
            not args.skip_llm_node_judge,
            args.node_judge_model,
            args.node_candidate_limit,
            args.node_evidence_limit,
            args.node_similarity_threshold,
            args.generate_timeout,
            memo.get("created_at") or now,
            use_evolve=not args.skip_evolve,
        )

        if args.save_every and index % args.save_every == 0:
            save_state_to_dir(state_dir, memos, memo_annotations, memory_nodes, memory_evidence)
            save_memory_links_to_dir(state_dir, [])

    # post-build pass: catch any nodes that crossed threshold on the very last memo
    if not args.skip_evolve:
        if args.summary_only:
            print("[build-eval-graph] post-build evolve pass...", flush=True)
        memo_by_id = {m["id"]: m for m in memos}
        evolve_semantic_nodes(
            memory_nodes, memory_evidence, memo_by_id, memo_annotations,
            schema, args.model, args.node_judge_model,
            args.node_similarity_threshold, args.generate_timeout, now,
        )

        if args.summary_only:
            print("[build-eval-graph] merging similar nodes...", flush=True)
        while merge_similar_nodes(
            memory_nodes, memory_evidence, args.model, args.node_similarity_threshold, now,
        ):
            pass

    save_state_to_dir(state_dir, memos, memo_annotations, memory_nodes, memory_evidence)
    save_memory_links_to_dir(state_dir, [])

    print(
        json.dumps(
            {
                "state_dir": str(state_dir),
                "memos": len(memos),
                "memo_annotations": len(memo_annotations),
                "memory_nodes": len(memory_nodes),
                "semantic_nodes": sum(1 for node in memory_nodes.values() if is_semantic_node(node)),
                "memory_evidence": len(memory_evidence),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_day(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None


def build_memo_node_map(
    memory_evidence: list[dict],
    memory_nodes: dict[str, dict] | None = None,
) -> dict[str, set[str]]:
    memo_node_ids: dict[str, set[str]] = defaultdict(set)
    for evidence in memory_evidence:
        memo_id = evidence.get("memo_id")
        node_id = evidence.get("node_id")
        if isinstance(memo_id, str) and isinstance(node_id, str):
            if memory_nodes is not None:
                node = memory_nodes.get(node_id)
                if node is None or not is_semantic_node(node):
                    continue
            memo_node_ids[memo_id].add(node_id)
    return memo_node_ids


def add_link_candidate(
    candidates: dict[tuple[str, str, str], dict],
    from_node_id: str,
    to_node_id: str,
    relation: str,
    memo_ids: list[str],
    support: int,
) -> None:
    if from_node_id == to_node_id:
        return

    key = (from_node_id, to_node_id, relation)
    candidate = candidates.setdefault(
        key,
        {
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "relation": relation,
            "support_count": 0,
            "evidence_memo_ids": [],
        },
    )
    candidate["support_count"] += support
    candidate["evidence_memo_ids"].extend(memo_ids)
    candidate["evidence_memo_ids"] = sorted(set(candidate["evidence_memo_ids"]))


def build_link_candidates(
    memos: list[dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
    min_support: int,
    window_days: int,
    include_temporal: bool,
) -> list[dict]:
    memo_by_id = {memo["id"]: memo for memo in memos}
    memo_node_ids = build_memo_node_map(memory_evidence, memory_nodes)
    candidates: dict[tuple[str, str, str], dict] = {}

    for memo_id, node_ids in memo_node_ids.items():
        for left_node_id, right_node_id in combinations(sorted(node_ids), 2):
            add_link_candidate(
                candidates,
                left_node_id,
                right_node_id,
                "overlaps",
                [memo_id],
                1,
            )

    if include_temporal:
        memo_events = []
        for memo_id, node_ids in memo_node_ids.items():
            memo = memo_by_id.get(memo_id)
            if memo is None:
                continue
            day = parse_day(memo.get("date") or memo.get("created_at"))
            if day is None:
                continue
            memo_events.append((day, memo_id, node_ids))

        memo_events.sort(key=lambda item: item[0])
        for index, (left_day, left_memo_id, left_node_ids) in enumerate(memo_events):
            for right_day, right_memo_id, right_node_ids in memo_events[index + 1:]:
                day_diff = (right_day - left_day).days
                if day_diff > window_days:
                    break
                if day_diff <= 0 or left_memo_id == right_memo_id:
                    continue
                for left_node_id in left_node_ids:
                    for right_node_id in right_node_ids:
                        add_link_candidate(
                            candidates,
                            left_node_id,
                            right_node_id,
                            "follows",
                            [left_memo_id, right_memo_id],
                            1,
                        )

    return sorted(
        [
            candidate
            for candidate in candidates.values()
            if candidate["support_count"] >= min_support
        ],
        key=lambda item: item["support_count"],
        reverse=True,
    )


def heuristic_link_reason(link: dict, memory_nodes: dict[str, dict]) -> str:
    from_node = memory_nodes.get(link["from_node_id"], {})
    to_node = memory_nodes.get(link["to_node_id"], {})
    from_title = from_node.get("title", link["from_node_id"])
    to_title = to_node.get("title", link["to_node_id"])

    if link["relation"] == "overlaps":
        return f"같은 메모에서 반복적으로 함께 나타남: {from_title} / {to_title}"
    if link["relation"] == "follows":
        return f"{from_title} 이후 {to_title}가 가까운 시점에 반복적으로 나타남"
    return f"{from_title} / {to_title} 사이의 반복 관계가 관찰됨"


def build_link_judge_prompt(link: dict, memory_nodes: dict[str, dict]) -> str:
    from_node = memory_nodes[link["from_node_id"]]
    to_node = memory_nodes[link["to_node_id"]]

    return "\n".join(
        [
            "You review candidate links between KEO memory nodes.",
            "Choose exactly one relation: reinforces, contrasts, causes, follows, overlaps, none.",
            "Use none if the evidence is not strong enough.",
            "Return JSON only.",
            'JSON schema: {"relation":"overlaps","reason":"짧은 한국어 문장","confidence":0.0}',
            "",
            "[Node A]",
            f"id: {from_node['id']}",
            f"title: {from_node['title']}",
            f"summary: {from_node['summary']}",
            "",
            "[Node B]",
            f"id: {to_node['id']}",
            f"title: {to_node['title']}",
            f"summary: {to_node['summary']}",
            "",
            "[Observed candidate]",
            f"candidate_relation: {link['relation']}",
            f"support_count: {link['support_count']}",
            f"evidence_memo_ids: {', '.join(link['evidence_memo_ids'][:12])}",
        ]
    )


def judge_link_with_llm(link: dict, memory_nodes: dict[str, dict], model: str, timeout: int) -> dict | None:
    prompt = build_link_judge_prompt(link, memory_nodes)
    response = recall.generate_text(prompt, model, timeout=timeout, response_format="json")
    payload = recall.extract_json_object(response)

    relation = payload.get("relation")
    reason = payload.get("reason")
    confidence = payload.get("confidence")
    if relation not in {"reinforces", "contrasts", "causes", "follows", "overlaps"}:
        return None
    if not isinstance(reason, str):
        reason = ""
    if not isinstance(confidence, (int, float)):
        confidence = 0.0

    return {
        **link,
        "relation": relation,
        "reason": reason,
        "confidence": float(confidence),
    }


def build_links(args: argparse.Namespace) -> None:
    memos, _, memory_nodes, memory_evidence = load_state()
    candidates = build_link_candidates(
        memos,
        memory_nodes,
        memory_evidence,
        args.min_support,
        args.window_days,
        args.include_temporal,
    )
    now = utc_now()
    memory_links = []

    for candidate in candidates[: args.limit]:
        if args.use_llm:
            judged = judge_link_with_llm(
                candidate,
                memory_nodes,
                args.link_model,
                args.generate_timeout,
            )
            if judged is None or judged["confidence"] < args.min_confidence:
                continue
            link = judged
        else:
            link = {
                **candidate,
                "reason": heuristic_link_reason(candidate, memory_nodes),
                "confidence": min(0.95, round(0.45 + candidate["support_count"] * 0.05, 2)),
            }
            if link["confidence"] < args.min_confidence:
                continue

        link["created_at"] = now
        memory_links.append(link)

    save_memory_links(memory_links)
    print(json.dumps(memory_links, ensure_ascii=False, indent=2))


def show_links(args: argparse.Namespace) -> None:
    memory_links = load_memory_links()
    memory_links = sorted(
        memory_links,
        key=lambda link: (link.get("support_count", 0), link.get("confidence", 0.0)),
        reverse=True,
    )
    print(json.dumps(memory_links[: args.limit], ensure_ascii=False, indent=2))


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "avg": 0,
            "median": 0,
        }

    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "avg": round(mean(values), 4),
        "median": round(median(values), 4),
    }


def count_by(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item] += 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))



def calculate_attach_quality(
    memos: list[dict],
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
) -> dict:
    actual_pairs = {
        (evidence.get("memo_id"), evidence.get("node_id"))
        for evidence in memory_evidence
        if isinstance(evidence.get("memo_id"), str)
        and isinstance(evidence.get("node_id"), str)
        and is_semantic_node(memory_nodes.get(evidence.get("node_id"), {}))
    }
    attached_memo_ids = set(
        memo_id
        for memo_id, node_ids in build_memo_node_map(memory_evidence, memory_nodes).items()
        if node_ids
    )

    return {
        "basis": "semantic node attachment in live graph",
        "attached_pair_count": len(actual_pairs),
        "attached_memo_count": len(attached_memo_ids),
        "attachment_coverage": round(len(attached_memo_ids) / len(memos), 4) if memos else 0.0,
        "source_counts": count_by(
            [
                evidence.get("source", "unknown")
                for evidence in memory_evidence
                if isinstance(evidence.get("memo_id"), str)
                and isinstance(evidence.get("node_id"), str)
                and is_semantic_node(memory_nodes.get(evidence.get("node_id"), {}))
            ]
        ),
    }


def calculate_node_coherence(
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
) -> dict:
    node_scores = []
    low_coherence_nodes = []

    for node_id, node in memory_nodes.items():
        if not is_semantic_node(node):
            continue
        node_tags = set(node.get("relation_tags", []))
        evidence_memo_ids = [
            evidence["memo_id"]
            for evidence in memory_evidence
            if evidence.get("node_id") == node_id and isinstance(evidence.get("memo_id"), str)
        ]
        if not evidence_memo_ids:
            continue

        matching_count = 0
        for memo_id in evidence_memo_ids:
            memo_tags = set(memo_annotations.get(memo_id, {}).get("relation_tags", []))
            if node_tags & memo_tags:
                matching_count += 1

        coherence = matching_count / len(evidence_memo_ids)
        node_scores.append(coherence)
        if coherence < 0.8:
            low_coherence_nodes.append(
                {
                    "node_id": node_id,
                    "title": node.get("title", node_id),
                    "coherence": round(coherence, 4),
                    "evidence_count": len(evidence_memo_ids),
                }
            )

    return {
        "basis": "node relation_tags와 evidence memo relation_tags의 overlap 비율",
        "node_coherence": summarize_numeric(node_scores),
        "low_coherence_nodes": sorted(
            low_coherence_nodes,
            key=lambda item: item["coherence"],
        )[:10],
    }


def audit(args: argparse.Namespace) -> None:
    state_dir = getattr(args, "state_dir", DATA_DIR)
    memos, memo_annotations, memory_nodes, memory_evidence = load_state_from_dir(state_dir)
    memory_links = load_memory_links_from_dir(state_dir)
    memo_node_ids = build_memo_node_map(memory_evidence, memory_nodes)

    memo_count = len(memos)
    node_counts_per_memo = [len(memo_node_ids.get(memo["id"], set())) for memo in memos]
    evidence_sources = [
        evidence.get("source", "unknown")
        for evidence in memory_evidence
        if isinstance(evidence.get("source", "unknown"), str)
    ]
    dense_evidence_count = sum(1 for source in evidence_sources if source == "dense")
    dense_added_node_ratio = dense_evidence_count / len(memory_evidence) if memory_evidence else 0.0

    evidence_counts = [
        int(node.get("evidence_count", 0))
        for node in memory_nodes.values()
    ]
    orphan_nodes = [
        {
            "node_id": node["id"],
            "title": node.get("title", node["id"]),
            "evidence_count": int(node.get("evidence_count", 0)),
        }
        for node in memory_nodes.values()
        if int(node.get("evidence_count", 0)) <= args.orphan_threshold
    ]
    over_broad_nodes = [
        {
            "node_id": node["id"],
            "title": node.get("title", node["id"]),
            "evidence_count": int(node.get("evidence_count", 0)),
        }
        for node in memory_nodes.values()
        if int(node.get("evidence_count", 0)) >= args.over_broad_threshold
    ]

    link_supports = [
        int(link.get("support_count", 0))
        for link in memory_links
    ]
    link_confidences = [
        float(link.get("confidence", 0.0))
        for link in memory_links
        if isinstance(link.get("confidence", 0.0), (int, float))
    ]
    low_confidence_links = [
        link
        for link in memory_links
        if float(link.get("confidence", 0.0)) < args.low_confidence_threshold
    ]
    relation_values = [
        link.get("relation", "unknown")
        for link in memory_links
        if isinstance(link.get("relation", "unknown"), str)
    ]

    report = {
        "counts": {
            "memos": memo_count,
            "memo_annotations": len(memo_annotations),
            "memory_nodes": len(memory_nodes),
            "semantic_nodes": sum(1 for node in memory_nodes.values() if is_semantic_node(node)),
            "memory_evidence": len(memory_evidence),
            "memory_links": len(memory_links),
        },
        "node_attach_quality": calculate_attach_quality(memos, memo_annotations, memory_nodes, memory_evidence),
        "node_attach_shape": {
            "avg_nodes_per_memo": round(mean(node_counts_per_memo), 4) if node_counts_per_memo else 0,
            "nodes_per_memo": summarize_numeric([float(value) for value in node_counts_per_memo]),
            "evidence_by_source": count_by(evidence_sources),
            "dense_added_node_ratio": round(dense_added_node_ratio, 4),
        },
        "node_quality": {
            **calculate_node_coherence(memo_annotations, memory_nodes, memory_evidence),
            "evidence_count_distribution": summarize_numeric([float(value) for value in evidence_counts]),
            "orphan_node_count": len(orphan_nodes),
            "orphan_nodes": orphan_nodes[:10],
            "over_broad_node_count": len(over_broad_nodes),
            "over_broad_nodes": sorted(
                over_broad_nodes,
                key=lambda item: item["evidence_count"],
                reverse=True,
            )[:10],
            "duplicate_node_rate": "not_measured_in_mvp_semantic_nodes",
        },
        "link_quality": {
            "link_precision": "needs_human_review",
            "relation_accuracy": "needs_human_or_llm_judged_labels",
            "relation_counts": count_by(relation_values),
            "support_count_distribution": summarize_numeric([float(value) for value in link_supports]),
            "confidence_distribution": summarize_numeric(link_confidences),
            "low_confidence_threshold": args.low_confidence_threshold,
            "low_confidence_link_count": len(low_confidence_links),
            "low_confidence_link_ratio": round(len(low_confidence_links) / len(memory_links), 4) if memory_links else 0,
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    schema = recall.load_tag_schema(args.tag_schema)
    past_memos = recall.load_past_memos(args.data)
    cases = recall.load_eval_cases(args.eval_data)
    past_tags = recall.load_tag_annotations(args.past_tags, schema)
    case_tags = recall.load_tag_annotations(args.eval_case_tags, schema)
    if args.use_state:
        stored_memos, stored_annotations, memory_nodes, memory_evidence = load_state_from_dir(args.state_dir)
        indexed_source_memos = stored_memos
        annotations = stored_annotations
    else:
        memory_nodes, memory_evidence = build_fixture_memory_graph(past_memos, past_tags, schema)
        indexed_source_memos = past_memos
        annotations = past_tags

    memo_by_id = {memo["id"]: memo for memo in indexed_source_memos}

    indexed_memos = recall.build_past_memo_index(
        indexed_source_memos,
        args.model,
        annotations,
        recall.RETRIEVAL_TEXT,
    )
    dense_stats = recall.empty_stats()
    memory_stats = recall.empty_stats()

    for index, case in enumerate(cases, start=1):
        if args.summary_only:
            print(f"[memory-eval] {index}/{len(cases)} {case['id']}")

        expected_ids = set(case["expected_ids"])
        query_embedding = recall.embed_text(case["current_memo"], args.model)
        dense_matches = recall.rank_by_embedding(query_embedding, indexed_memos, "text_embedding")
        dense_results = dense_matches[: args.top_k]
        query_tags = case_tags.get(case["id"], {}).get("relation_tags", [])

        node_candidates = score_node_candidates(
            query_tags,
            dense_matches[: args.dense_top_k],
            memory_nodes,
            memory_evidence,
            args.tag_weight,
            args.dense_weight,
        )
        memory_results = rank_memory_evidence_memos(
            node_candidates,
            memory_evidence,
            memo_by_id,
            annotations,
            dense_matches[: args.dense_top_k],
            query_tags,
            args.node_top_k,
            args.top_k,
            not args.no_dense_fallback,
        )

        recall.update_stats(dense_stats, expected_ids, dense_results)
        recall.update_stats(memory_stats, expected_ids, memory_results)

        if not args.summary_only:
            print(f"\n=== {case['id']} ===")
            print(case["current_memo"])
            print(f"query_tags: {', '.join(query_tags) if query_tags else '-'}")
            print("\n[node candidates]")
            for node_id, payload in sorted(node_candidates.items(), key=lambda item: item[1]["score"], reverse=True)[: args.node_top_k]:
                node = memory_nodes.get(node_id, {"title": node_id})
                print(f"- {node_id} score={payload['score']:.4f} title={node['title']}")
            recall.print_eval_summary(case, memory_results, args.top_k)

    print("\n=== Memory Graph Eval Summary ===")
    print(f"cases: {len(cases)}")
    print(f"expected: {sum(len(case['expected_ids']) for case in cases)}")
    print(f"memory nodes: {len(memory_nodes)}")
    print(f"semantic nodes: {sum(1 for node in memory_nodes.values() if is_semantic_node(node))}")
    print(f"memory evidence: {len(memory_evidence)}")
    print(f"graph source: {str(args.state_dir) if args.use_state else 'fixture'}")
    print(f"dense_top_k for node routing: {args.dense_top_k}")
    print(f"node_top_k: {args.node_top_k}")
    print("\n[dense baseline]")
    recall.print_overall_stats(dense_stats, args.top_k)
    print("\n[memory graph]")
    recall.print_overall_stats(memory_stats, args.top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KEO memory graph PoC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="store a memo and update memory nodes")
    ingest_parser.add_argument("--memo", required=True, help="memo text to ingest")
    ingest_parser.add_argument("--memo-id", help="optional memo id")
    ingest_parser.add_argument("--date", help="memo date in YYYY-MM-DD")
    ingest_parser.add_argument("--relation-tags", help="comma-separated fixed relation tags")
    ingest_parser.add_argument("--tagging-mode", choices=["fixed", "llm"], default="fixed")
    ingest_parser.add_argument("--tagging-model", default=recall.DEFAULT_TAGGING_MODEL)
    ingest_parser.add_argument("--tag-schema", type=Path, default=recall.TAG_SCHEMA_PATH)
    ingest_parser.add_argument("--model", default=recall.DEFAULT_MODEL)
    ingest_parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    ingest_parser.add_argument("--attach-dense-node-min-hits", type=int, default=DEFAULT_ATTACH_DENSE_NODE_MIN_HITS)
    ingest_parser.add_argument("--use-llm-node-judge", action="store_true")
    ingest_parser.add_argument("--node-judge-model", default=recall.DEFAULT_TAGGING_MODEL)
    ingest_parser.add_argument("--node-candidate-limit", type=int, default=8)
    ingest_parser.add_argument("--node-evidence-limit", type=int, default=DEFAULT_NODE_EVIDENCE_LIMIT)
    ingest_parser.add_argument("--node-similarity-threshold", type=float, default=DEFAULT_NODE_SIMILARITY_THRESHOLD)
    ingest_parser.add_argument("--generate-timeout", type=int, default=recall.GENERATE_TIMEOUT_SECONDS)

    show_parser = subparsers.add_parser("nodes", help="print memory nodes")
    show_parser.add_argument("--limit", type=int, default=20)

    links_parser = subparsers.add_parser("links", help="print memory links")
    links_parser.add_argument("--limit", type=int, default=20)

    audit_parser = subparsers.add_parser("audit", help="print memory graph quality metrics")
    audit_parser.add_argument("--state-dir", type=Path, default=DATA_DIR)
    audit_parser.add_argument("--orphan-threshold", type=int, default=1)
    audit_parser.add_argument("--over-broad-threshold", type=int, default=25)
    audit_parser.add_argument("--low-confidence-threshold", type=float, default=0.6)

    build_links_parser = subparsers.add_parser("build-links", help="build memory links from accumulated evidence")
    build_links_parser.add_argument("--min-support", type=int, default=DEFAULT_LINK_MIN_SUPPORT)
    build_links_parser.add_argument("--window-days", type=int, default=DEFAULT_LINK_WINDOW_DAYS)
    build_links_parser.add_argument("--include-temporal", action="store_true")
    build_links_parser.add_argument("--limit", type=int, default=50)
    build_links_parser.add_argument("--min-confidence", type=float, default=0.0)
    build_links_parser.add_argument("--use-llm", action="store_true")
    build_links_parser.add_argument("--link-model", default=recall.DEFAULT_TAGGING_MODEL)
    build_links_parser.add_argument("--generate-timeout", type=int, default=recall.GENERATE_TIMEOUT_SECONDS)

    seed_parser = subparsers.add_parser("seed-eval", help="legacy: write eval past memos without seed nodes")
    seed_parser.add_argument("--force", action="store_true", help="replace existing data files")
    seed_parser.add_argument("--data", type=Path, default=recall.DATA_PATH)
    seed_parser.add_argument("--tag-schema", type=Path, default=recall.TAG_SCHEMA_PATH)
    seed_parser.add_argument("--past-tags", type=Path, default=recall.PAST_TAGS_PATH)

    build_eval_parser = subparsers.add_parser("build-eval-graph", help="build an isolated memory graph from eval past memos")
    build_eval_parser.add_argument("--state-dir", type=Path, default=BASE_DIR / "data_eval")
    build_eval_parser.add_argument("--force", action="store_true")
    build_eval_parser.add_argument("--limit", type=int)
    build_eval_parser.add_argument("--summary-only", action="store_true")
    build_eval_parser.add_argument("--save-every", type=int, default=25)
    build_eval_parser.add_argument("--model", default=recall.DEFAULT_MODEL)
    build_eval_parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    build_eval_parser.add_argument("--attach-dense-node-min-hits", type=int, default=DEFAULT_ATTACH_DENSE_NODE_MIN_HITS)
    build_eval_parser.add_argument("--skip-llm-node-judge", action="store_true")
    build_eval_parser.add_argument("--skip-evolve", action="store_true")
    build_eval_parser.add_argument("--node-judge-model", default=recall.DEFAULT_TAGGING_MODEL)
    build_eval_parser.add_argument("--node-candidate-limit", type=int, default=8)
    build_eval_parser.add_argument("--node-evidence-limit", type=int, default=DEFAULT_NODE_EVIDENCE_LIMIT)
    build_eval_parser.add_argument("--node-similarity-threshold", type=float, default=DEFAULT_NODE_SIMILARITY_THRESHOLD)
    build_eval_parser.add_argument("--generate-timeout", type=int, default=recall.GENERATE_TIMEOUT_SECONDS)
    build_eval_parser.add_argument("--data", type=Path, default=recall.DATA_PATH)
    build_eval_parser.add_argument("--tag-schema", type=Path, default=recall.TAG_SCHEMA_PATH)
    build_eval_parser.add_argument("--past-tags", type=Path, default=recall.PAST_TAGS_PATH)

    eval_parser = subparsers.add_parser("eval", help="evaluate semantic memory graph retrieval")
    eval_parser.add_argument("--model", default=recall.DEFAULT_MODEL)
    eval_parser.add_argument("--top-k", type=int, default=recall.DEFAULT_TOP_K)
    eval_parser.add_argument("--dense-top-k", type=int, default=DEFAULT_EVAL_DENSE_TOP_K)
    eval_parser.add_argument("--node-top-k", type=int, default=DEFAULT_NODE_TOP_K)
    eval_parser.add_argument("--tag-weight", type=float, default=2.0)
    eval_parser.add_argument("--dense-weight", type=float, default=1.0)
    eval_parser.add_argument("--summary-only", action="store_true")
    eval_parser.add_argument("--use-state", action="store_true", help="evaluate persisted semantic memory graph instead of empty fixture graph")
    eval_parser.add_argument("--state-dir", type=Path, default=DATA_DIR)
    eval_parser.add_argument("--no-dense-fallback", action="store_true")
    eval_parser.add_argument("--data", type=Path, default=recall.DATA_PATH)
    eval_parser.add_argument("--eval-data", type=Path, default=recall.EVAL_PATH)
    eval_parser.add_argument("--tag-schema", type=Path, default=recall.TAG_SCHEMA_PATH)
    eval_parser.add_argument("--past-tags", type=Path, default=recall.PAST_TAGS_PATH)
    eval_parser.add_argument("--eval-case-tags", type=Path, default=recall.EVAL_CASE_TAGS_PATH)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "ingest":
        ingest(args)
    elif args.command == "nodes":
        show_nodes(args)
    elif args.command == "links":
        show_links(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "build-links":
        build_links(args)
    elif args.command == "seed-eval":
        seed_from_eval(args)
    elif args.command == "build-eval-graph":
        build_eval_graph(args)
    elif args.command == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
