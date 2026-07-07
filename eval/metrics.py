"""Scoring for ALQAC 2026: 4-label accuracy + law-F1 + API efficiency penalty."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "B_WIN", "PARTIAL_B_WIN"}


def _strip_diacritics(s: str) -> str:
    """NFD normalize then drop combining marks. Keeps đ → d."""
    s = unicodedata.normalize("NFD", s)
    out = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return out.replace("đ", "d").replace("Đ", "d")


def _canonical_key(raw: str) -> str:
    """Aggressive normalization: strip diacritics, lowercase, drop year suffixes, collapse spaces."""
    s = _strip_diacritics(raw).lower()
    # Drop "nam YYYY" / "YYYY" year suffixes
    s = re.sub(r"\bnam\s+\d{4}\b", "", s)
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)
    # Drop punctuation
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Canonical-keyed lookup → corpus law_id. Built once from corpus + extended with manual entries.
CANONICAL_LAW_MAP = {
    "bo luat dan su": "91/2015/QH13",
    "bo luat to tung dan su": "92/2015/QH13",
    "bo luat hinh su": "100/2015/QH13",
    "bo luat hon nhan va gia dinh": "52/2014/QH13",
    "luat dat dai": "45/2013/QH13",
    "luat kinh doanh bat dong san": "66/2014/QH13",
    "luat to chuc tin dung": "47/2010/QH12",
    "luat thi hanh an dan su": "26/2008/QH12",
    "luat ho tich": "60/2014/QH13",
    "luat nuoi con nuoi": "52/2010/QH12",
    "nghi dinh kinh doanh vang": "24/2012/NĐ-CP",
    "nghi dinh hop dong xay dung": "37/2015/NĐ-CP",
    "luat khieu nai": "02/2011/QH13",
    "luat to tung hanh chinh": "93/2015/QH13",
    "luat nguoi cao tuoi": "39/2009/QH12",
    "nghi quyet an phi le phi toa an": "326/2016/UBTVQH14",
    "phap lenh an phi le phi toa an": "10/2009/UBTVQH12",
}


LAW_NAME_TO_ID_OVERRIDES = CANONICAL_LAW_MAP


@dataclass
class CaseResult:
    case_id: str
    gold_verdict: str
    pred_verdict: str
    gold_law_refs: set[tuple[str, int]] = field(default_factory=set)
    pred_law_refs: set[tuple[str, int]] = field(default_factory=set)
    n_unique_chunks: int = 0
    gold_segments_estimate: int = 5


def _norm_name(raw: str) -> str:
    return _canonical_key(raw)


def _guess_law_id(raw_name: str, fallback_map: dict[str, str] | None = None) -> str:
    n = _canonical_key(raw_name)
    if n in LAW_NAME_TO_ID_OVERRIDES:
        return LAW_NAME_TO_ID_OVERRIDES[n]
    if fallback_map and n in fallback_map:
        return fallback_map[n]
    # Substring fallbacks for short / partial citations
    if "to tung" in n:
        return "92/2015/QH13"
    if "to chuc tin dung" in n:
        return "47/2010/QH12"
    if "hon nhan" in n or "hon nhan va gia dinh" in n:
        return "52/2014/QH13"
    if "dat dai" in n:
        return "45/2013/QH13"
    if "kinh doanh bat dong san" in n:
        return "66/2014/QH13"
    if "thi hanh an" in n:
        return "26/2008/QH12"
    if "xay dung" in n:
        # Luật Xây dựng 2014 - not in our corpus; keep original
        return raw_name.strip()
    if "dan su" in n and len(n) < 30:
        return "91/2015/QH13"
    if "326/2016" in raw_name:
        return "326/2016/UBTVQH14"
    if "10/2009" in raw_name or "an phi le phi" in n:
        return "10/2009/UBTVQH12"
    if "37/2015" in raw_name and "nđ-cp" in n:
        return "37/2015/NĐ-CP"
    return raw_name.strip()


def build_law_name_map(corpus_path: Path | str) -> dict[str, str]:
    """Build canonical→law_id map from corpus article 1 first sentence (heuristic)."""
    corpus = json.loads(Path(corpus_path).read_text())
    out: dict[str, str] = {}
    for law in corpus:
        first = law["content"][0]["content_Article"]
        key = _canonical_key(first[:120])
        # Match against canonical substrings
        if "bo luat" in key and "to tung" in key:
            out.setdefault("bo luat to tung dan su", law["law_id"])
        elif "bo luat" in key and "hinh su" in key:
            out.setdefault("bo luat hinh su", law["law_id"])
        elif "bo luat" in key and "hon nhan" in key:
            out.setdefault("bo luat hon nhan va gia dinh", law["law_id"])
        elif "bo luat" in key:
            out.setdefault("bo luat dan su", law["law_id"])
        elif "dat dai" in key:
            out.setdefault("luat dat dai", law["law_id"])
        elif "to chuc tin dung" in key:
            out.setdefault("luat to chuc tin dung", law["law_id"])
        elif "kinh doanh bat dong san" in key:
            out.setdefault("luat kinh doanh bat dong san", law["law_id"])
    return out


def parse_gold_law_refs(related_law_provisions: str, fallback_map: dict[str, str] | None = None) -> set[tuple[str, int]]:
    """Extract (law_id, corpus_aid) tuples from gold `related_law_provisions`.

    The raw text stores article numbers (e.g. "Điều 116"). We look up the
    corresponding corpus aid via retrieval.law_refs_registry so gold and pred
    live in the same id-space.
    """
    from retrieval.law_refs_registry import article_aid  # local import to avoid cycle at module load

    refs: set[tuple[str, int]] = set()
    for line in (related_law_provisions or "").splitlines():
        if "|" not in line:
            continue
        name_part, art_part = line.split("|", 1)
        nums = re.findall(r"điều\s*(\d+)", art_part.strip().lower())
        if not nums:
            continue
        law_id = _guess_law_id(name_part.strip(), fallback_map)
        for n in nums:
            article_number = int(n)
            aid = article_aid(law_id, article_number)
            if aid is None:
                aid = article_number  # fallback if lookup fails
            refs.add((law_id, int(aid)))
    return refs


def parse_pred_law_refs(law_refs: list, case_id: str | None = None) -> set[tuple[str, int]]:
    """Accept either:
      - list[int]: ref_ids from search_law.py, looked up in runs/law_refs.db by case_id
      - list[dict]: legacy {law_id, article_number|aid} shape.
    Returns set of (law_id, aid) tuples where aid is the corpus-internal id.
    """
    out: set[tuple[str, int]] = set()
    if not law_refs:
        return out

    if all(isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit() for x in law_refs):
        try:
            from retrieval.law_refs_registry import LawRefsRegistry
        except Exception:
            return out
        if not case_id:
            return out
        reg = LawRefsRegistry()
        for x in law_refs:
            try:
                ref = reg.lookup(case_id, int(x))
            except (ValueError, TypeError):
                continue
            if ref and ref.aid is not None:
                out.add((ref.law_id, int(ref.aid)))
        return out

    # Legacy dict shape.
    for r in law_refs:
        if not isinstance(r, dict):
            continue
        try:
            num = r.get("article_number") or r.get("aid")
            out.add((str(r["law_id"]), int(num)))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def parse_submission_verdict(submission: str | dict) -> tuple[str, list[dict], list[str]]:
    if isinstance(submission, dict):
        data = submission
    else:
        m = re.search(r"\{[\s\S]*\}", submission or "")
        if not m:
            return "", [], []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return "", [], []
    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict not in VALID_LABELS:
        verdict = ""
    law_refs = data.get("law_refs", []) or []
    chunk_ids = data.get("evidence_chunk_ids", []) or []
    if isinstance(chunk_ids, list):
        chunk_ids = [str(c) for c in chunk_ids]
    return verdict, law_refs, chunk_ids


def score_case(result: CaseResult) -> dict:
    verdict_correct = int(result.gold_verdict == result.pred_verdict and result.pred_verdict != "")
    gold = result.gold_law_refs
    pred = result.pred_law_refs
    precision = len(gold & pred) / len(pred) if pred else 0.0
    recall = len(gold & pred) / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    n = result.n_unique_chunks
    g = max(1, result.gold_segments_estimate)
    if n <= 2 * g:
        penalty = 0.0
    elif n >= 5 * g:
        penalty = 1.0
    else:
        penalty = (n - 2 * g) / (3 * g)
    return {
        "case_id": result.case_id,
        "gold_verdict": result.gold_verdict,
        "pred_verdict": result.pred_verdict,
        "verdict_correct": verdict_correct,
        "law_precision": precision,
        "law_recall": recall,
        "law_f1": f1,
        "n_unique_chunks": n,
        "api_penalty": penalty,
    }


def aggregate(per_case: list[dict]) -> dict:
    """Official ALQAC 2026 formula:
        FinalScore = 0.70·OutcomeAccuracy + 0.20·PenalizedCaseRecall + 0.10·LawF1_micro

    LawF1_micro is a single F1 over the union of (case_id, law_id, aid) TP/FP/FN
    across all cases (not the mean of per-case F1).

    PenalizedCaseRecall requires gold case-evidence segments per case, which are
    NOT included in the public test JSON. Local score uses a placeholder — the
    real value comes from the official server.
    """
    n = len(per_case) or 1
    acc = sum(r["verdict_correct"] for r in per_case) / n

    tp = fp = fn = 0
    for r in per_case:
        gold = {(x["law_id"], int(x["aid"])) for x in r.get("gold_law_refs", []) or []}
        pred = {(x["law_id"], int(x["aid"])) for x in r.get("pred_law_refs", []) or []}
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    law_f1_micro = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    penalty = sum(r["api_penalty"] for r in per_case) / n
    penalized_case_recall_placeholder = 1.0 - penalty

    combined = (
        0.70 * acc
        + 0.20 * penalized_case_recall_placeholder
        + 0.10 * law_f1_micro
    )

    return {
        "n_cases": len(per_case),
        "accuracy_4class": acc,
        "law_f1_micro": law_f1_micro,
        "law_precision_micro": prec,
        "law_recall_micro": rec,
        "avg_api_penalty": penalty,
        "penalized_case_recall_placeholder": penalized_case_recall_placeholder,
        "combined_score": combined,
    }
