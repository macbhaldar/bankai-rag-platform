"""Print a summary of the provided dataset (documents, QA splits, benchmarks) """

from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.pipelines.chunking import DOC_ID_RE, family_of  # noqa: E402


def main() -> int:
    settings = get_settings()
    corpus = settings.paths.corpus_dir
    files = sorted(corpus.rglob("*.md"))
    print(f"corpus: {corpus}")
    print(f"documents: {len(files)}")

    domains = Counter(p.parent.name for p in files)
    print("per domain:", dict(sorted(domains.items())))

    families: Counter = Counter()
    versions: Counter = Counter()
    for path in files:
        match = DOC_ID_RE.match(path.stem)
        if match:
            families[match.group("family")] += 1
            versions[match.group("version").split(".")[0]] += 1
    print(f"document families: {len(families)}")
    for family, count in families.most_common(12):
        print(f"  {family}: {count} versions")
    print(f"total files: {sum(families.values())}")

    qa_dir = settings.paths.qa_dir
    for split in ("train", "validation", "test", "all"):
        path = qa_dir / f"qa_{split}.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            sections = Counter(row.get("section", "?") for row in rows)
            print(f"qa_{split}: {len(rows)} cases; sections: {dict(sections.most_common(6))}")

    retrieval = qa_dir.parent / "retrieval"
    for name in ("queries.jsonl", "relevance_judgments.jsonl", "hard_negatives.jsonl"):
        path = retrieval / name
        if path.exists():
            n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"retrieval/{name}: {n} rows")

    structured = settings.paths.structured_dir
    if structured.exists():
        for csv_path in sorted(structured.glob("*.csv")):
            n = sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1
            print(f"structured/{csv_path.name}: {n} rows")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())