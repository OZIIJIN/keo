import argparse
import json
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from pathlib import Path

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
DEFAULT_LINK_WINDOW_DAYS = 2


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


def slugify_tag(tag: str) -> str:
    return tag.replace("_", "-")


def node_id_for_tag(tag: str) -> str:
    return f"node-{slugify_tag(tag)}"


def title_for_tag(tag: str, description: str) -> str:
    title_overrides = {
        "start_avoidance": "시작 회피",
        "execution_delay": "실행 지연",
        "safe_task_selection": "안전한 일 선택",
        "reactive_work": "반응형 작업",
        "context_fragmentation": "흐름 끊김",
        "overplanning": "과계획",
        "unfinished_loop": "미해결 루프",
        "priority_confusion": "우선순위 혼탁",
        "decision_fog": "판단 흐림",
        "self_deception": "자기기만",
        "avoidance_rationalization": "회피 합리화",
        "evaluation_fear": "평가 두려움",
        "control_seeking": "통제 추구",
        "social_response_delay": "답장 미룸",
        "life_admin_overload": "생활 일 과부하",
        "low_energy_drift": "에너지 저하",
        "fatigue_spillover": "피로 번짐",
        "rhythm_disruption": "리듬 붕괴",
        "rest_guilt": "쉼 불안",
        "recovery_failure": "회복 실패",
        "schedule_disruption": "일정 흔들림",
        "external_priority_capture": "외부 우선순위 잠식",
        "event_aftereffect": "사건 여파",
        "pattern_recognition": "패턴 인식",
        "event_reflection_link": "사건-회고 연결",
        "decision_trace": "판단 흐름 기록",
        "identity_gap": "정체성 간극",
        "self_understanding_product": "자기이해 제품 통찰",
    }
    return title_overrides.get(tag, description)


def load_state() -> tuple[list[dict], dict[str, dict], dict[str, dict], list[dict]]:
    memos = read_json(MEMOS_PATH, [])
    memo_annotations = read_json(MEMO_ANNOTATIONS_PATH, {})
    memory_nodes = read_json(MEMORY_NODES_PATH, {})
    memory_evidence = read_json(MEMORY_EVIDENCE_PATH, [])

    if not isinstance(memos, list):
        raise ValueError(f"{MEMOS_PATH} must contain a JSON list")
    if not isinstance(memo_annotations, dict):
        raise ValueError(f"{MEMO_ANNOTATIONS_PATH} must contain a JSON object")
    if not isinstance(memory_nodes, dict):
        raise ValueError(f"{MEMORY_NODES_PATH} must contain a JSON object")
    if not isinstance(memory_evidence, list):
        raise ValueError(f"{MEMORY_EVIDENCE_PATH} must contain a JSON list")

    return memos, memo_annotations, memory_nodes, memory_evidence


def save_state(
    memos: list[dict],
    memo_annotations: dict[str, dict],
    memory_nodes: dict[str, dict],
    memory_evidence: list[dict],
) -> None:
    write_json(MEMOS_PATH, memos)
    write_json(MEMO_ANNOTATIONS_PATH, memo_annotations)
    write_json(MEMORY_NODES_PATH, memory_nodes)
    write_json(MEMORY_EVIDENCE_PATH, memory_evidence)


def load_memory_links() -> list[dict]:
    links = read_json(MEMORY_LINKS_PATH, [])
    if not isinstance(links, list):
        raise ValueError(f"{MEMORY_LINKS_PATH} must contain a JSON list")
    return links


def save_memory_links(memory_links: list[dict]) -> None:
    write_json(MEMORY_LINKS_PATH, memory_links)


def ensure_node_for_tag(
    tag: str,
    schema: dict,
    memory_nodes: dict[str, dict],
    now: str,
) -> dict:
    node_id = node_id_for_tag(tag)
    existing = memory_nodes.get(node_id)
    if existing is not None:
        return existing

    description = schema["relation_tags"][tag]
    node = {
        "id": node_id,
        "type": "pattern",
        "title": title_for_tag(tag, description),
        "summary": description,
        "relation_tags": [tag],
        "evidence_count": 0,
        "confidence": 0.5,
        "last_seen_at": None,
        "created_at": now,
        "updated_at": now,
    }
    memory_nodes[node_id] = node
    return node


def evidence_exists(memory_evidence: list[dict], node_id: str, memo_id: str) -> bool:
    return any(
        item.get("node_id") == node_id and item.get("memo_id") == memo_id
        for item in memory_evidence
    )


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
    if evidence_exists(memory_evidence, node_id, memo_id):
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

    node = memory_nodes[node_id]
    node["evidence_count"] = int(node.get("evidence_count", 0)) + 1
    node["last_seen_at"] = now
    node["updated_at"] = now
    node["confidence"] = min(0.95, round(0.5 + node["evidence_count"] * 0.05, 2))
    base_summary = str(node.get("summary", "")).split(" 현재 ")[0]
    if base_summary:
        node["summary"] = f"{base_summary} 현재 {node['evidence_count']}개 메모에서 반복 관찰됨."
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


def count_dense_node_hits(dense_matches: list[dict], memory_evidence: list[dict]) -> dict[str, dict]:
    memo_to_node_ids: dict[str, list[str]] = {}
    for evidence in memory_evidence:
        memo_id = evidence.get("memo_id")
        node_id = evidence.get("node_id")
        if isinstance(memo_id, str) and isinstance(node_id, str):
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


def build_fixture_memory_graph(
    past_memos: list[dict],
    annotations: dict[str, dict],
    schema: dict,
) -> tuple[dict[str, dict], list[dict]]:
    now = utc_now()
    memory_nodes: dict[str, dict] = {}
    memory_evidence: list[dict] = []

    for memo in past_memos:
        relation_tags = annotations.get(memo["id"], {}).get("relation_tags", [])
        for tag in relation_tags:
            node = ensure_node_for_tag(tag, schema, memory_nodes, now)
            add_evidence(
                memory_nodes,
                memory_evidence,
                node["id"],
                memo["id"],
                schema["relation_tags"][tag],
                "fixture",
                0.8,
                memo.get("date", now),
            )

    return memory_nodes, memory_evidence


def score_node_candidates(
    query_tags: list[str],
    dense_matches: list[dict],
    memory_evidence: list[dict],
    tag_weight: float,
    dense_weight: float,
) -> dict[str, dict]:
    candidates: dict[str, dict] = {}

    for tag in query_tags:
        node_id = node_id_for_tag(tag)
        candidate = candidates.setdefault(
            node_id,
            {"score": 0.0, "tag_hits": [], "dense_memo_ids": [], "dense_hits": 0},
        )
        candidate["score"] += tag_weight
        candidate["tag_hits"].append(tag)

    memo_to_node_ids: dict[str, list[str]] = {}
    for evidence in memory_evidence:
        memo_id = evidence.get("memo_id")
        node_id = evidence.get("node_id")
        if isinstance(memo_id, str) and isinstance(node_id, str):
            memo_to_node_ids.setdefault(memo_id, []).append(node_id)

    for rank, match in enumerate(dense_matches, start=1):
        rank_score = dense_weight / rank
        for node_id in memo_to_node_ids.get(match["id"], []):
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


def ingest(args: argparse.Namespace) -> None:
    schema = recall.load_tag_schema(args.tag_schema)
    memos, memo_annotations, memory_nodes, memory_evidence = load_state()
    now = utc_now()

    memo_id = args.memo_id or next_memo_id(memos)
    if any(memo["id"] == memo_id for memo in memos):
        raise SystemExit(f"memo already exists: {memo_id}")

    relation_tags = classify_relation_tags(args, schema)
    memo = {
        "id": memo_id,
        "date": args.date or now[:10],
        "text": args.memo.strip(),
        "created_at": now,
    }
    memos.append(memo)
    memo_annotations[memo_id] = {
        "relation_tags": relation_tags,
        "short_summary": args.memo.strip()[:80],
    }

    linked_nodes = []
    connection_decisions = []
    for tag in relation_tags:
        node_was_existing = node_id_for_tag(tag) in memory_nodes
        node = ensure_node_for_tag(tag, schema, memory_nodes, now)
        added = add_evidence(
            memory_nodes,
            memory_evidence,
            node["id"],
            memo_id,
            schema["relation_tags"][tag],
            "relation_tag",
            0.8,
            now,
        )
        if added:
            linked_nodes.append(node["id"])
        connection_decisions.append(
            {
                "node_id": node["id"],
                "source": "relation_tag",
                "decision": "attach" if node_was_existing else "create",
                "reason": f"새 메모가 relation tag `{tag}`로 분류됨",
                "weight": 0.8,
            }
        )

    dense_matches = dense_search(
        memo,
        memos,
        memo_annotations,
        schema,
        args.model,
        args.dense_top_k,
    )
    dense_node_hits = count_dense_node_hits(dense_matches, memory_evidence)

    for node_id, hit in dense_node_hits.items():
        if node_id in linked_nodes:
            continue
        if hit["count"] < args.attach_dense_node_min_hits:
            connection_decisions.append(
                {
                    "node_id": node_id,
                    "source": "dense",
                    "decision": "ignore",
                    "reason": f"dense evidence hit {hit['count']}개로 threshold {args.attach_dense_node_min_hits} 미만",
                    "weight": 0.0,
                }
            )
            continue
        if node_id not in memory_nodes:
            continue
        weight = min(0.75, round(0.45 + hit["count"] * 0.1, 2))
        added = add_evidence(
            memory_nodes,
            memory_evidence,
            node_id,
            memo_id,
            f"dense 결과에서 이 node의 evidence가 {hit['count']}개 발견됨",
            "dense",
            weight,
            now,
        )
        if added:
            linked_nodes.append(node_id)
        connection_decisions.append(
            {
                "node_id": node_id,
                "source": "dense",
                "decision": "attach",
                "reason": f"dense 결과의 과거 메모 {hit['count']}개가 이미 이 node에 연결되어 있음",
                "weight": weight,
                "dense_memo_ids": hit["memo_ids"],
            }
        )

    save_state(memos, memo_annotations, memory_nodes, memory_evidence)

    result = {
        "memo": memo,
        "annotation": memo_annotations[memo_id],
        "connection_decisions": connection_decisions,
        "linked_nodes": [memory_nodes[node_id] for node_id in linked_nodes],
        "dense_matches": [
            {
                "id": match["id"],
                "score": round(float(match["score"]), 4),
                "date": match.get("date"),
                "relation_tags": match.get("relation_tags", []),
                "text": match["text"],
            }
            for match in dense_matches
        ],
        "dense_node_candidates": dense_node_hits,
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


def parse_day(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None


def build_memo_node_map(memory_evidence: list[dict]) -> dict[str, set[str]]:
    memo_node_ids: dict[str, set[str]] = defaultdict(set)
    for evidence in memory_evidence:
        memo_id = evidence.get("memo_id")
        node_id = evidence.get("node_id")
        if isinstance(memo_id, str) and isinstance(node_id, str):
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
    memory_evidence: list[dict],
    min_support: int,
    window_days: int,
    include_temporal: bool,
) -> list[dict]:
    memo_by_id = {memo["id"]: memo for memo in memos}
    memo_node_ids = build_memo_node_map(memory_evidence)
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


def evaluate(args: argparse.Namespace) -> None:
    schema = recall.load_tag_schema(args.tag_schema)
    past_memos = recall.load_past_memos(args.data)
    cases = recall.load_eval_cases(args.eval_data)
    past_tags = recall.load_tag_annotations(args.past_tags, schema)
    case_tags = recall.load_tag_annotations(args.eval_case_tags, schema)
    memory_nodes, memory_evidence = build_fixture_memory_graph(past_memos, past_tags, schema)
    memo_by_id = {memo["id"]: memo for memo in past_memos}

    indexed_memos = recall.build_past_memo_index(
        past_memos,
        args.model,
        past_tags,
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
            memory_evidence,
            args.tag_weight,
            args.dense_weight,
        )
        memory_results = rank_memory_evidence_memos(
            node_candidates,
            memory_evidence,
            memo_by_id,
            past_tags,
            dense_matches[: args.dense_top_k],
            query_tags,
            args.node_top_k,
            args.top_k,
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
    print(f"memory evidence: {len(memory_evidence)}")
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
    ingest_parser.add_argument("--generate-timeout", type=int, default=recall.GENERATE_TIMEOUT_SECONDS)

    show_parser = subparsers.add_parser("nodes", help="print memory nodes")
    show_parser.add_argument("--limit", type=int, default=20)

    links_parser = subparsers.add_parser("links", help="print memory links")
    links_parser.add_argument("--limit", type=int, default=20)

    build_links_parser = subparsers.add_parser("build-links", help="build memory links from accumulated evidence")
    build_links_parser.add_argument("--min-support", type=int, default=DEFAULT_LINK_MIN_SUPPORT)
    build_links_parser.add_argument("--window-days", type=int, default=DEFAULT_LINK_WINDOW_DAYS)
    build_links_parser.add_argument("--include-temporal", action="store_true")
    build_links_parser.add_argument("--limit", type=int, default=50)
    build_links_parser.add_argument("--min-confidence", type=float, default=0.0)
    build_links_parser.add_argument("--use-llm", action="store_true")
    build_links_parser.add_argument("--link-model", default=recall.DEFAULT_TAGGING_MODEL)
    build_links_parser.add_argument("--generate-timeout", type=int, default=recall.GENERATE_TIMEOUT_SECONDS)

    seed_parser = subparsers.add_parser("seed-eval", help="replace data state with eval past memos")
    seed_parser.add_argument("--force", action="store_true", help="replace existing data files")
    seed_parser.add_argument("--data", type=Path, default=recall.DATA_PATH)
    seed_parser.add_argument("--tag-schema", type=Path, default=recall.TAG_SCHEMA_PATH)
    seed_parser.add_argument("--past-tags", type=Path, default=recall.PAST_TAGS_PATH)

    eval_parser = subparsers.add_parser("eval", help="evaluate fixture-backed memory graph retrieval")
    eval_parser.add_argument("--model", default=recall.DEFAULT_MODEL)
    eval_parser.add_argument("--top-k", type=int, default=recall.DEFAULT_TOP_K)
    eval_parser.add_argument("--dense-top-k", type=int, default=DEFAULT_EVAL_DENSE_TOP_K)
    eval_parser.add_argument("--node-top-k", type=int, default=DEFAULT_NODE_TOP_K)
    eval_parser.add_argument("--tag-weight", type=float, default=2.0)
    eval_parser.add_argument("--dense-weight", type=float, default=1.0)
    eval_parser.add_argument("--summary-only", action="store_true")
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
    elif args.command == "build-links":
        build_links(args)
    elif args.command == "seed-eval":
        seed_from_eval(args)
    elif args.command == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
