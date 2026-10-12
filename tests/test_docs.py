import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("JOB_STORIES.md", "INFORMATION_ARCHITECTURE.md", "README.md", "QA_LOG.md", "CLAUDE.md")

# Чотири літери російської абетки, яких немає в українській. Задані кодами
# символів, щоб цей файл сам не потрапляв під перевірку.
RUSSIAN_ONLY = re.compile(
    "[" + "".join(map(chr, (0x44B, 0x44D, 0x44A, 0x451, 0x42B, 0x42D, 0x42A, 0x401))) + "]"
)


def _project_texts() -> dict[str, str]:
    paths = [ROOT / name for name in DOCS if (ROOT / name).exists()]
    paths += sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in paths}


def test_references_point_to_defined_criteria_and_decisions():
    # AC-10.2: кожен згаданий AC-x.y і D-x визначено в JOB_STORIES.md
    job_stories = (ROOT / "JOB_STORIES.md").read_text(encoding="utf-8")
    defined_ac = set(re.findall(r"\*\*(AC-\d+\.\d+)\*\*", job_stories))
    defined_d = set(re.findall(r"^\| (D-\d+) \|", job_stories, re.MULTILINE))
    broken = []
    for name, text in _project_texts().items():
        broken += [(name, ref) for ref in set(re.findall(r"AC-\d+\.\d+", text)) - defined_ac]
        broken += [(name, ref) for ref in set(re.findall(r"\bD-\d+\b", text)) - defined_d]
    assert not broken


def test_no_russian_text_in_docs_and_code():
    # D-11, AC-10.7: документація й код лише українською
    hits = [
        (name, text[max(0, m.start() - 20) : m.end() + 20])
        for name, text in _project_texts().items()
        for m in RUSSIAN_ONLY.finditer(text)
    ]
    assert not hits
