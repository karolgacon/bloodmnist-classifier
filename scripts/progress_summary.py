#!/usr/bin/env python3
"""Print task summary from docs/PROGRESS.yaml (no PyYAML required)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "docs" / "PROGRESS.yaml"


def parse_tasks(text: str) -> list[dict]:
    tasks = []
    blocks = re.split(r"\n  - id: ", text)
    for block in blocks[1:]:
        tid = block.split("\n", 1)[0].strip()
        status = re.search(r"status: (\w+)", block)
        title = re.search(r"title: (.+)", block)
        owner = re.search(r"owner: (.+)", block)
        tasks.append(
            {
                "id": tid,
                "status": status.group(1) if status else "?",
                "owner": owner.group(1).strip() if owner else "?",
                "title": title.group(1).strip() if title else "?",
            }
        )
    return tasks


def main() -> None:
    if not PROGRESS.exists():
        print(f"Missing {PROGRESS}", file=sys.stderr)
        sys.exit(1)
    text = PROGRESS.read_text(encoding="utf-8")
    tasks = parse_tasks(text)
    counts = {}
    for t in tasks:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    print("BloodMNIST Classifier — postęp\n")
    print("Podsumowanie statusów:", dict(sorted(counts.items())))
    print()
    print(f"{'ID':<8} {'Status':<12} {'Owner':<6} Title")
    print("-" * 72)
    for t in tasks:
        print(f"{t['id']:<8} {t['status']:<12} {t['owner']:<6} {t['title'][:45]}")


if __name__ == "__main__":
    main()
