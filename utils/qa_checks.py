# utils/qa_checks.py
"""
Upgraded TranslationQAEngine
- Multilingual sentence embeddings (SentenceTransformers)
- Embedding caching and batch encode
- Many rule-based checks (numbers, dates, punctuation, tags, glossary, length ratio, etc.)
- Dynamic weighting for scoring
- Single-segment analyze_segment() and batch_analyze()
"""

import re
import html
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Optional: language detection (install langdetect if you want)
try:
    from langdetect import detect as lang_detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger("TranslationQAEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

#sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 we can use this for faster download
from sentence_transformers import SentenceTransformer
import torch
import logging

logger = logging.getLogger(__name__)

class TranslationQAEngine:
    def __init__(
        self,
        model_name: str = r"E:\Industry_Projects\QA assistant\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
        cache_size: int = 16384,
    ):
        """
        model_name: a multilingual SentenceTransformers model recommended for translation QA.
        device: "cpu" or "cuda" or None (auto-detect)
        cache_size: number of cached embeddings in LRU cache
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cache_size = cache_size

        logger.info(f"Loading model '{model_name}' on device '{self.device}'")

        # ✅ Proper model loading
        self.model = SentenceTransformer(model_name, device=self.device)

        # small warm-up (optional)
        try:
            self.model.max_seq_length = 512
        except Exception:
            pass

        # Embedding cache (simple LRU)
        self._get_embedding = lru_cache(maxsize=cache_size)(self._get_embedding_uncached)

        # Glossary: src_term -> tgt_term (lowercased)
        self.glossary: Dict[str, str] = {}

        # Default error weights (tunable)
        self.error_weights = {
            "Low Semantic Similarity": 30,
            "Number Mismatch": 20,
            "Glossary Violation": 20,
            "Date Mismatch": 12,
            "Symbol Mismatch": 8,
            "Punctuation Mismatch": 8,
            "Tag Mismatch": 25,
            "Length Ratio Issue": 6,
            "Identical Segment": 18,
            "Case Mismatch": 4,
            "Model Error": 40,
            "Untranslated Segment": 22,
        }

        # Scoring config: how much semantic similarity contributes (0-1)
        self.semantic_weight = 0.7
        self.rule_weight = 1 - self.semantic_weight

    def load_glossary(self, glossary_dict: Dict[str, str]):
        """Load glossary (source_term -> target_term)"""
        self.glossary = {k.lower(): v.lower() for k, v in (glossary_dict or {}).items()}

    # -------------------------
    # Embedding helpers
    # -------------------------
    def _get_embedding_uncached(self, text: str) -> Tuple[np.ndarray, bool]:
        """Uncached embedding function wrapped by lru_cache decorator.
           Returns (embedding_numpy, success_flag)"""
        try:
            emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            emb = emb.detach().cpu().numpy()
            return emb, True
        except Exception as e:
            logger.exception("Embedding error")
            # Return zero vector on error (and flag false)
            return np.zeros(self.model.get_sentence_embedding_dimension(), dtype=float), False

    def get_embedding(self, text: str) -> Tuple[np.ndarray, bool]:
        """Public wrapper for cached embedding retrieval."""
        if text is None:
            text = ""
        # sanitize text to keep cache stable
        key = text.strip()
        return self._get_embedding(key)

    def batch_encode(self, texts: List[str], batch_size: int = 64) -> Tuple[np.ndarray, List[bool]]:
        """Efficient batch encoding with fallback per-sentence.
           Returns embeddings shape (n, dim) and list of success flags."""
        cleaned = [("" if t is None else t) for t in texts]
        embeddings = []
        success_flags: List[bool] = []

        # attempt to encode in batches via model (faster)
        try:
            tensor_embs = self.model.encode(cleaned, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False)
            arr = tensor_embs.detach().cpu().numpy()
            return arr, [True] * len(cleaned)
        except Exception:
            # fallback to per-item (and cache will help)
            for t in cleaned:
                emb, ok = self.get_embedding(t)
                embeddings.append(emb)
                success_flags.append(ok)
            return np.vstack(embeddings) if embeddings else np.zeros((0, self.model.get_sentence_embedding_dimension())), success_flags

    # -------------------------
    # Rule checks
    # -------------------------
    def check_numbers(self, source: str, translation: str) -> List[Dict[str, Any]]:
        src_nums = re.findall(r'\b\d+(?:[.,]\d+)*\b', source or "")
        tgt_nums = re.findall(r'\b\d+(?:[.,]\d+)*\b', translation or "")
        src_norm = [re.sub(r'[.,]', '', n) for n in src_nums]
        tgt_norm = [re.sub(r'[.,]', '', n) for n in tgt_nums]
        if sorted(src_norm) != sorted(tgt_norm):
            return [self._err("Number Mismatch", "High",
                              f"Numbers differ. Source: {src_nums}, Translation: {tgt_nums}",
                              category="Numbers",
                              suggestion="Ensure numeric values are preserved exactly.")]
        return []

    def check_dates(self, source: str, translation: str) -> List[Dict[str, Any]]:
        patterns = [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}-\d{2}-\d{2}']
        src_dates = []
        tgt_dates = []
        for p in patterns:
            src_dates.extend(re.findall(p, source or ""))
            tgt_dates.extend(re.findall(p, translation or ""))
        if len(src_dates) != len(tgt_dates):
            return [self._err("Date Mismatch", "Medium",
                              f"Source dates: {src_dates}, Translation dates: {tgt_dates}",
                              category="Dates",
                              suggestion="Check date formats and presence.")]
        return []

    def check_symbols(self, source: str, translation: str) -> List[Dict[str, Any]]:
        special_chars = r'[@#$%&*(){}[\]<>|\\~`]'
        src = re.findall(special_chars, source or "")
        tgt = re.findall(special_chars, translation or "")
        if Counter(src) != Counter(tgt):
            return [self._err("Symbol Mismatch", "Medium",
                              f"Symbols differ. Source: {src}, Translation: {tgt}",
                              category="Symbols",
                              suggestion="Preserve special characters.")]
        return []

    def check_punctuation(self, source: str, translation: str) -> List[Dict[str, Any]]:
        # Punctuation excluding alphanumeric and whitespace
        src = re.findall(r"[^\w\s]", source or "")
        tgt = re.findall(r"[^\w\s]", translation or "")
        if Counter(src) != Counter(tgt):
            return [self._err("Punctuation Mismatch", "Low",
                              "Punctuation differs between source and translation.",
                              category="Punctuation",
                              suggestion="Check commas, periods, colons etc.")]
        return []

    def check_tags(self, source: str, translation: str) -> List[Dict[str, Any]]:
        # Extract HTML/XML tags like <b>, </b>, <br />, {{placeholders}}, <0> ... </0>
        tag_pattern = r'(<[^>]+>)|(\{\{.*?\}\})|(%\w+%)|(<\d+>)|(<\/\d+>)'
        src_tags = re.findall(tag_pattern, source or "")
        tgt_tags = re.findall(tag_pattern, translation or "")
        # flatten tuples returned by findall
        src_tags = [next(filter(None, t)) for t in src_tags]
        tgt_tags = [next(filter(None, t)) for t in tgt_tags]
        if Counter(src_tags) != Counter(tgt_tags):
            return [self._err("Tag Mismatch", "High",
                              f"Tags/placeholders mismatch. Source tags: {src_tags}, Translation tags: {tgt_tags}",
                              category="Tags",
                              suggestion="Ensure tags and placeholders are preserved and placed correctly.")]
        return []

    def check_length_ratio(self, source: str, translation: str, min_ratio: float = 0.5, max_ratio: float = 2.0) -> List[Dict[str, Any]]:
        s_len = max(len(source or ""), 1)
        t_len = len(translation or "")
        ratio = t_len / s_len
        if ratio < min_ratio or ratio > max_ratio:
            return [self._err("Length Ratio Issue", "Low",
                              f"Translation length ratio {ratio:.2f} (source {s_len} -> target {t_len})",
                              category="Length",
                              suggestion="Check for missing content or verbosity.")]
        return []

    def check_identical_or_untranslated(self, source: str, translation: str) -> List[Dict[str, Any]]:
        s = (source or "").strip()
        t = (translation or "").strip()
        if not t:
            return [self._err("Untranslated Segment", "High", "Translation is empty.", category="Content", suggestion="Provide translation.")]
        # identical after basic normalization -> likely untranslated
        def norm(x): return re.sub(r'\s+', ' ', x.lower()).strip()
        if norm(s) == norm(t):
            return [self._err("Identical Segment", "High", "Source and translation are identical (may be untranslated).", category="Content", suggestion="Translate the segment properly.")]
        return []

    def check_case_issues(self, source: str, translation: str) -> List[Dict[str, Any]]:
        # Check for uppercase/lowercase mismatches for proper nouns or starts of sentences
        src_caps = re.findall(r'\b[A-Z][a-z]+', source or "")
        tgt_caps = re.findall(r'\b[A-Z][a-z]+', translation or "")
        # if many source ProperNouns not found in target -> flag (low severity)
        missing = [w for w in src_caps if w.lower() not in (t.lower() for t in tgt_caps)]
        if missing:
            return [self._err("Case Mismatch", "Low", f"Proper nouns/case differences: {missing}", category="Case", suggestion="Verify proper nouns and capitalization.")]
        return []

    def check_glossary(self, source: str, translation: str) -> List[Dict[str, Any]]:
        errs = []
        if not self.glossary:
            return errs
        s_lower = (source or "").lower()
        t_lower = (translation or "").lower()
        for src_term, tgt_term in self.glossary.items():
            if src_term in s_lower and tgt_term not in t_lower:
                errs.append(self._err("Glossary Violation", "High",
                                      f"Term '{src_term}' expected as '{tgt_term}' but not found in translation.",
                                      category="Terminology",
                                      suggestion=f"Use glossary translation for '{src_term}' -> '{tgt_term}'"))
        return errs

    # -------------------------
    # Semantic checks
    # -------------------------
    def semantic_similarity(self, source: str, translation: str) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compute cosine similarity between source and translation using embeddings.
        Returns (similarity_score_float, errors_list)
        """
        try:
            src_emb, ok1 = self.get_embedding(source)
            tgt_emb, ok2 = self.get_embedding(translation)
            if not (ok1 and ok2):
                return 0.0, [self._err("Model Error", "High", "Failed to encode text with model.", category="Model")]
            # cosine similarity using numpy
            sim = float(np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-12))
            return sim, []
        except Exception as e:
            logger.exception("semantic_similarity error")
            return 0.0, [self._err("Model Error", "High", str(e), category="Model")]

    # -------------------------
    # Helper for structured errors
    # -------------------------
    def _err(self, typ: str, severity: str, description: str, category: str = "General", suggestion: Optional[str] = None) -> Dict[str, Any]:
        return {
            "type": typ,
            "severity": severity,
            "description": description,
            "category": category,
            "suggestion": suggestion or ""
        }

    # -------------------------
    # Single-segment analyze
    # -------------------------
    def analyze_segment(self, source: str, translation: str, similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Run all checks on a single segment and compute a calibrated quality score.
        Returns dict with fields:
         - source, translation, semantic_similarity, quality_score, errors (list), error_count, explain (weights)
        """
        errors: List[Dict[str, Any]] = []

        # Basic rule checks
        errors.extend(self.check_numbers(source, translation))
        errors.extend(self.check_dates(source, translation))
        errors.extend(self.check_symbols(source, translation))
        errors.extend(self.check_punctuation(source, translation))
        errors.extend(self.check_tags(source, translation))
        errors.extend(self.check_length_ratio(source, translation))
        errors.extend(self.check_identical_or_untranslated(source, translation))
        errors.extend(self.check_case_issues(source, translation))
        errors.extend(self.check_glossary(source, translation))

        # Semantic
        sim, sem_errs = self.semantic_similarity(source, translation)
        errors.extend(sem_errs)
        if sim < similarity_threshold:
            errors.append(self._err("Low Semantic Similarity", "High",
                                    f"Similarity {sim:.3f} below threshold {similarity_threshold:.3f}",
                                    category="Semantics",
                                    suggestion="Review translation meaning / re-translate."))

        # Score computation
        # semantic component scaled to 0-100
        semantic_score = sim * 100
        # rule-based penalty: sum weights of unique error types
        penalty = 0
        error_types = set()
        for e in errors:
            et = e.get("type", "")
            if et in error_types:
                # don't double-penalize same exact error type repeatedly (configurable)
                continue
            error_types.add(et)
            penalty += self.error_weights.get(et, 5)

        # Combine: weighted average between semantic_score and (100 - penalty)
        rule_component = max(0, 100 - penalty)
        final_score = (semantic_score * self.semantic_weight) + (rule_component * self.rule_weight)
        final_score = max(0.0, min(100.0, final_score))

        # Explainability / confidence
        explain = {
            "semantic_score": round(semantic_score, 2),
            "rule_penalty": penalty,
            "final_score": round(final_score, 2),
            "num_errors": len(errors),
            "error_types": list(error_types)
        }

        return {
            "source": source,
            "translation": translation,
            "semantic_similarity": float(sim),
            "quality_score": final_score,
            "errors": errors,
            "error_count": len(errors),
            "explain": explain
        }

    # -------------------------
    # Batch analyze (vectorized semantic)
    # -------------------------
    def batch_analyze(self, sources: List[str], translations: List[str], similarity_threshold: float = 0.6, batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Analyze lists of sources and translations efficiently.
        Returns list of per-segment analysis dicts (same shape/order as inputs).
        """
        if len(sources) != len(translations):
            raise ValueError("sources and translations must have same length")

        # Precompute embeddings in batches for speed
        src_embs, src_flags = self.batch_encode(sources, batch_size=batch_size)
        tgt_embs, tgt_flags = self.batch_encode(translations, batch_size=batch_size)

        # cosine similarities using sentence-transformers util if arrays valid
        sims = []
        for i in range(len(sources)):
            try:
                if src_flags[i] and tgt_flags[i]:
                    a = src_embs[i]
                    b = tgt_embs[i]
                    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                else:
                    sim = 0.0
                sims.append(sim)
            except Exception:
                sims.append(0.0)

        results = []
        for i, (s, t) in enumerate(zip(sources, translations)):
            # run lighter rule checks (we avoid re-encoding inside those)
            errs = []
            errs.extend(self.check_numbers(s, t))
            errs.extend(self.check_dates(s, t))
            errs.extend(self.check_symbols(s, t))
            errs.extend(self.check_punctuation(s, t))
            errs.extend(self.check_tags(s, t))
            errs.extend(self.check_length_ratio(s, t))
            errs.extend(self.check_identical_or_untranslated(s, t))
            errs.extend(self.check_case_issues(s, t))
            errs.extend(self.check_glossary(s, t))

            sim = sims[i]
            if sim < similarity_threshold:
                errs.append(self._err("Low Semantic Similarity", "High",
                                      f"Similarity {sim:.3f} below threshold {similarity_threshold:.3f}",
                                      category="Semantics"))

            # scoring (reuse same logic)
            semantic_score = sim * 100
            penalty = 0
            error_types = set()
            for e in errs:
                et = e.get("type", "")
                if et in error_types:
                    continue
                error_types.add(et)
                penalty += self.error_weights.get(et, 5)
            rule_component = max(0, 100 - penalty)
            final_score = (semantic_score * self.semantic_weight) + (rule_component * self.rule_weight)
            final_score = max(0.0, min(100.0, final_score))

            explain = {
                "semantic_score": round(semantic_score, 2),
                "rule_penalty": penalty,
                "final_score": round(final_score, 2),
                "num_errors": len(errs),
                "error_types": list(error_types)
            }

            results.append({
                "segment_id": i + 1,
                "source": s,
                "translation": t,
                "semantic_similarity": float(sim),
                "quality_score": final_score,
                "errors": errs,
                "error_count": len(errs),
                "explain": explain
            })

        return results
