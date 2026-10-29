"""
backend/services/theme_matcher.py — Real matching service against candidate research gaps,
biomedical concepts, and topics.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger("ThemeMatcher")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class ThemeMatcher:
    def __init__(self):
        self._gaps: List[Dict[str, Any]] = []
        self._concepts: List[Dict[str, Any]] = []
        self._topics: List[Dict[str, Any]] = []
        self._loaded = False
        self._load_data()

    def _load_data(self):
        try:
            # 1. Candidate research gaps
            gaps_path = OUTPUTS_DIR / "candidate_research_gaps.csv"
            if gaps_path.exists():
                gaps_df = pd.read_csv(gaps_path, encoding="utf-8-sig")
                # Normalize column names
                if "\ufeffconcept_a" in gaps_df.columns:
                    gaps_df.rename(columns={"\ufeffconcept_a": "concept_a"}, inplace=True)
                
                # Check for gap_id
                for idx, row in gaps_df.iterrows():
                    ca = str(row.get("concept_a", "")).strip()
                    cb = str(row.get("concept_b", "")).strip()
                    gap_id = f"GAP-{idx+1:03d}"
                    self._gaps.append({
                        "gap_id": gap_id,
                        "concept_a": ca,
                        "concept_b": cb,
                        "gap_score": float(row.get("gap_score", 0.0)),
                        "semantic_similarity": float(row.get("semantic_similarity", 0.0)),
                        "cooccurrence_score": float(row.get("cooccurrence_score", 0.0)),
                        "temporal_growth_avg": float(row.get("temporal_growth_avg", 0.0)),
                        "papers_concept_a": int(row.get("papers_concept_a", 0)),
                        "papers_concept_b": int(row.get("papers_concept_b", 0)),
                        "papers_joint": int(row.get("papers_joint", 0)),
                        "supporting_topic_ids": str(row.get("supporting_topic_ids", "")).split(";"),
                        "description": f"Underexplored intersection between {ca} and {cb} in mHealth research.",
                        "is_precomputed": True
                    })
                logger.info("Loaded %d candidate research gaps for matching.", len(self._gaps))

            # 2. Biomedical Concepts
            concepts_path = OUTPUTS_DIR / "concept_summary.csv"
            if concepts_path.exists():
                c_df = pd.read_csv(concepts_path, encoding="utf-8-sig")
                if "\ufeffconcept" in c_df.columns:
                    c_df.rename(columns={"\ufeffconcept": "concept"}, inplace=True)
                for _, row in c_df.iterrows():
                    c_name = str(row.get("concept", "")).strip()
                    if c_name and len(c_name) > 2:
                        self._concepts.append({
                            "concept": c_name,
                            "document_count": int(row.get("document_count", 0)),
                            "total_mentions": int(row.get("total_mentions", 0)),
                        })
                logger.info("Loaded %d biomedical concepts.", len(self._concepts))

            # 3. Topic Summary
            topics_path = OUTPUTS_DIR / "topic_summary.csv"
            if topics_path.exists():
                t_df = pd.read_csv(topics_path, encoding="utf-8-sig")
                if "\ufefftopic_id" in t_df.columns:
                    t_df.rename(columns={"\ufefftopic_id": "topic_id"}, inplace=True)
                for _, row in t_df.iterrows():
                    tid = int(row.get("topic_id", -1))
                    if tid >= 0:  # ignore outlier -1
                        self._topics.append({
                            "topic_id": tid,
                            "top_terms": str(row.get("top_terms", "")),
                            "count": int(row.get("count", 0)),
                            "percentage": float(row.get("percentage", 0.0)),
                        })
                logger.info("Loaded %d topics.", len(self._topics))

            self._loaded = True
        except Exception as e:
            logger.error("Error loading theme matcher data: %s", e, exc_info=True)

    def get_live_suggestions(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Fast real suggestion search across candidate gaps, MeSH concepts, and topics.
        """
        if not query or len(query.strip()) < 2:
            return []

        q = query.strip().lower()
        tokens = [t for t in re.split(r"\W+", q) if len(t) >= 2]
        suggestions: List[Dict[str, Any]] = []

        # 1. Check Gaps
        for g in self._gaps:
            ca = g["concept_a"].lower()
            cb = g["concept_b"].lower()
            score = 0.0

            if q in ca or q in cb:
                score = 1.0 if ca.startswith(q) or cb.startswith(q) else 0.85
            elif tokens and all(t in ca or t in cb for t in tokens):
                score = 0.75
            elif tokens and any(t in ca or t in cb for t in tokens):
                score = 0.55

            if score > 0:
                suggestions.append({
                    "text": f"{g['concept_a']} ↔ {g['concept_b']}",
                    "type": "gap",
                    "category": "Research Gap",
                    "details": f"GapScore: {g['gap_score']:.2f} · Joint Papers: {g['papers_joint']}",
                    "gap_id": g["gap_id"],
                    "score": score + (g["gap_score"] * 0.05),
                })

        # 2. Check Biomedical Concepts
        for c in self._concepts:
            c_name = c["concept"].lower()
            score = 0.0
            if c_name == q:
                score = 0.95
            elif c_name.startswith(q):
                score = 0.85
            elif q in c_name:
                score = 0.70
            elif tokens and all(t in c_name for t in tokens):
                score = 0.60

            if score > 0:
                suggestions.append({
                    "text": c["concept"],
                    "type": "concept",
                    "category": "Biomedical Concept",
                    "details": f"{c['document_count']} papers in corpus",
                    "gap_id": None,
                    "score": score + min(0.1, c["document_count"] / 10000.0),
                })

        # 3. Check Topics
        for t in self._topics:
            terms = t["top_terms"].lower()
            if q in terms or (tokens and any(tk in terms for tk in tokens)):
                suggestions.append({
                    "text": f"Topic #{t['topic_id']}: {t['top_terms']}",
                    "type": "topic",
                    "category": "Research Topic",
                    "details": f"{t['count']} papers ({t['percentage']:.1f}%)",
                    "gap_id": None,
                    "score": 0.65,
                })

        # Sort by score descending and deduplicate by text
        seen = set()
        unique_suggestions = []
        for s in sorted(suggestions, key=lambda x: x["score"], reverse=True):
            key = s["text"].lower()
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)
            if len(unique_suggestions) >= limit:
                break

        return unique_suggestions

    def match_theme_to_gaps(self, theme: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Matches a user's natural language research theme to the 50 candidate gaps.
        Returns top_k matched gaps with matching score and explanation.
        """
        if not self._gaps:
            return []

        theme_clean = theme.lower().strip()
        tokens = [t for t in re.split(r"\W+", theme_clean) if len(t) >= 3]

        scored_gaps = []
        for g in self._gaps:
            ca = g["concept_a"].lower()
            cb = g["concept_b"].lower()

            match_points = 0.0
            reasons = []

            # Exact concept occurrences
            if ca in theme_clean:
                match_points += 3.0
                reasons.append(f"Explicitly mentions concept '{g['concept_a']}'")
            if cb in theme_clean:
                match_points += 3.0
                reasons.append(f"Explicitly mentions concept '{g['concept_b']}'")

            # Token overlap
            ca_tokens = [t for t in re.split(r"\W+", ca) if len(t) >= 3]
            cb_tokens = [t for t in re.split(r"\W+", cb) if len(t) >= 3]

            overlap_a = [t for t in tokens if t in ca_tokens]
            overlap_b = [t for t in tokens if t in cb_tokens]

            if overlap_a and ca not in theme_clean:
                match_points += 1.0 * len(overlap_a)
                reasons.append(f"Shares keywords with '{g['concept_a']}' ({', '.join(overlap_a)})")
            if overlap_b and cb not in theme_clean:
                match_points += 1.0 * len(overlap_b)
                reasons.append(f"Shares keywords with '{g['concept_b']}' ({', '.join(overlap_b)})")

            # Priority and gap score baseline
            baseline_score = g["gap_score"] * 0.5
            total_matching_score = match_points + baseline_score

            if not reasons:
                explanation = f"Top-ranked mHealth research gap based on semantic divergence and temporal growth (GapScore: {g['gap_score']:.2f})."
            else:
                explanation = f"Theme alignment: {'; '.join(reasons)}. Bridges isolated concepts in mHealth."

            matched_dict = dict(g)
            matched_dict["matching_score"] = round(total_matching_score, 4)
            matched_dict["matching_explanation"] = explanation
            scored_gaps.append(matched_dict)

        # Sort by matching score descending
        scored_gaps.sort(key=lambda x: (x["matching_score"], x["gap_score"]), reverse=True)
        return scored_gaps[:top_k]


# Singleton instance
_matcher_instance: Optional[ThemeMatcher] = None


def get_theme_matcher() -> ThemeMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = ThemeMatcher()
    return _matcher_instance
