#!/usr/bin/env python3
"""Validate and write a KEO recall eval dataset from a combined JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON file containing `past_memos` and `eval_cases`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "eval",
        help="Directory where past_memos.json and eval_cases.json will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Append the input dataset to existing eval files in the output directory.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")
    if "past_memos" not in data or "eval_cases" not in data:
        raise ValueError("JSON must include `past_memos` and `eval_cases`.")
    if not isinstance(data["past_memos"], list) or not isinstance(data["eval_cases"], list):
        raise ValueError("`past_memos` and `eval_cases` must both be arrays.")
    return data


def load_existing_array(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return data


def validate_memos(past_memos: list[dict]) -> list[str]:
    memo_ids: list[str] = []
    for index, memo in enumerate(past_memos, start=1):
        if not isinstance(memo, dict):
            raise ValueError(f"Memo #{index} must be an object.")
        for key in ("id", "date", "text"):
            if key not in memo or not isinstance(memo[key], str) or not memo[key].strip():
                raise ValueError(f"Memo #{index} must include non-empty string `{key}`.")
        memo_ids.append(memo["id"])
    if len(memo_ids) != len(set(memo_ids)):
        raise ValueError("Duplicate memo ids found.")
    return memo_ids


def validate_cases(eval_cases: list[dict], memo_ids: set[str]) -> None:
    case_ids: list[str] = []
    for index, case in enumerate(eval_cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{index} must be an object.")
        if "id" not in case or not isinstance(case["id"], str) or not case["id"].strip():
            raise ValueError(f"Case #{index} must include non-empty string `id`.")
        if (
            "current_memo" not in case
            or not isinstance(case["current_memo"], str)
            or not case["current_memo"].strip()
        ):
            raise ValueError(f"Case #{index} must include non-empty string `current_memo`.")
        if "expected_ids" not in case or not isinstance(case["expected_ids"], list) or not case["expected_ids"]:
            raise ValueError(f"Case #{index} must include non-empty array `expected_ids`.")
        for expected_id in case["expected_ids"]:
            if not isinstance(expected_id, str) or expected_id not in memo_ids:
                raise ValueError(
                    f"Case #{index} has unknown expected_id `{expected_id}`."
                )
        case_ids.append(case["id"])
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case ids found.")


def write_json(path: Path, payload: list[dict], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Re-run with --force to overwrite.")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input.resolve())
    past_memos = payload["past_memos"]
    eval_cases = payload["eval_cases"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    past_memos_path = output_dir / "past_memos.json"
    eval_cases_path = output_dir / "eval_cases.json"

    if args.merge:
        existing_past_memos = (
            load_existing_array(past_memos_path) if past_memos_path.exists() else []
        )
        existing_eval_cases = (
            load_existing_array(eval_cases_path) if eval_cases_path.exists() else []
        )

        existing_memo_ids = validate_memos(existing_past_memos)
        validate_cases(existing_eval_cases, set(existing_memo_ids))

        combined_past_memos = existing_past_memos + past_memos
        combined_eval_cases = existing_eval_cases + eval_cases

        combined_memo_ids = validate_memos(combined_past_memos)
        validate_cases(combined_eval_cases, set(combined_memo_ids))

        write_json(past_memos_path, combined_past_memos, True)
        write_json(eval_cases_path, combined_eval_cases, True)

        print(
            f"Merged {len(past_memos)} memos into {past_memos_path} "
            f"for a total of {len(combined_past_memos)}"
        )
        print(
            f"Merged {len(eval_cases)} eval cases into {eval_cases_path} "
            f"for a total of {len(combined_eval_cases)}"
        )
        return

    memo_ids = validate_memos(past_memos)
    validate_cases(eval_cases, set(memo_ids))

    write_json(past_memos_path, past_memos, args.force)
    write_json(eval_cases_path, eval_cases, args.force)

    print(f"Wrote {len(past_memos)} memos to {past_memos_path}")
    print(f"Wrote {len(eval_cases)} eval cases to {eval_cases_path}")


if __name__ == "__main__":
    main()
