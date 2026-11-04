"""
scripts/patch_nli.py — Optimize NLIValidator with offline-first loading, inference_mode, caching, and batch evaluation.
"""

from pathlib import Path

NLI_PY = Path("src/phase2/nli_validation.py")
content = NLI_PY.read_text(encoding="utf-8")

# 1. Update __init__ to include pair cache
old_init_tail = """        self.model = None
        self.available = False
        self.id2label: Dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.label2id: Dict[str, int] = {"contradiction": 0, "entailment": 1, "neutral": 2}

        self._initialize_model()"""

new_init_tail = """        self.model = None
        self.available = False
        self.id2label: Dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}
        self.label2id: Dict[str, int] = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self._pair_cache: Dict[Tuple[str, str], Tuple[str, Dict[str, float]]] = {}

        self._initialize_model()"""

content = content.replace(old_init_tail.replace("\r\n", "\n"), new_init_tail.replace("\r\n", "\n"))

# 2. Update _initialize_model to use local_files_only=True first
old_cross_enc = """            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )"""

new_cross_enc = """            try:
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512,
                    local_files_only=True,
                )
            except Exception:
                self.model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=512,
                )"""

content = content.replace(old_cross_enc.replace("\r\n", "\n"), new_cross_enc.replace("\r\n", "\n"))

# 3. Update predict_batch_nli to use torch.inference_mode() and caching
old_predict_batch = '''    def predict_batch_nli(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Predict NLI labels for multiple premise-hypothesis pairs in efficient mini-batches.

        Parameters
        ----------
        pairs : list of tuple of (premise, hypothesis)
            List of text pairs.

        Returns
        -------
        list of tuple of (str, dict)
            List of (best_label, scores_dict) per pair.
        """
        if not pairs:
            return []

        if not self.is_available():
            raise RuntimeError("NLI model is not available for batch prediction.")

        raw_logits = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        probs = scipy.special.softmax(raw_logits, axis=-1)
        results = []

        for row_probs in probs:
            scores_dict: Dict[str, float] = {}
            for idx, prob in enumerate(row_probs):
                lbl = self.id2label.get(idx, f"label_{idx}")
                scores_dict[lbl] = float(prob)

            entail_score = scores_dict.get("entailment", 0.0)
            contra_score = scores_dict.get("contradiction", 0.0)
            neutral_score = scores_dict.get("neutral", 0.0)

            if entail_score >= self.support_threshold and entail_score > contra_score:
                best_label = "entailment"
            elif contra_score >= self.contradiction_threshold and contra_score > entail_score:
                best_label = "contradiction"
            else:
                argmax_idx = int(np.argmax(row_probs))
                best_label = self.id2label.get(argmax_idx, "neutral")

            results.append(
                (
                    best_label,
                    {
                        "entailment": round(entail_score, 4),
                        "contradiction": round(contra_score, 4),
                        "neutral": round(neutral_score, 4),
                    },
                )
            )

        return results'''

new_predict_batch = '''    def predict_batch_nli(
        self, pairs: List[Tuple[str, str]]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Predict NLI labels for multiple premise-hypothesis pairs in efficient mini-batches with caching.

        Parameters
        ----------
        pairs : list of tuple of (premise, hypothesis)
            List of text pairs.

        Returns
        -------
        list of tuple of (str, dict)
            List of (best_label, scores_dict) per pair.
        """
        if not pairs:
            return []

        if not self.is_available():
            raise RuntimeError("NLI model is not available for batch prediction.")

        uncached_indices = []
        uncached_pairs = []
        cached_results: Dict[int, Tuple[str, Dict[str, float]]] = {}

        if not hasattr(self, "_pair_cache"):
            self._pair_cache = {}

        for idx, (p, h) in enumerate(pairs):
            pair_key = (str(p).strip(), str(h).strip())
            if pair_key in self._pair_cache:
                cached_results[idx] = self._pair_cache[pair_key]
            else:
                uncached_indices.append(idx)
                uncached_pairs.append((pair_key[0], pair_key[1]))

        if uncached_pairs:
            inference_ctx = torch.inference_mode() if (TORCH_AVAILABLE and torch is not None) else nullcontext()
            with inference_ctx:
                raw_logits = self.model.predict(
                    uncached_pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )

            probs = scipy.special.softmax(raw_logits, axis=-1)

            for u_idx, (p_text, h_text), row_probs in zip(uncached_indices, uncached_pairs, probs):
                scores_dict: Dict[str, float] = {}
                for idx, prob in enumerate(row_probs):
                    lbl = self.id2label.get(idx, f"label_{idx}")
                    scores_dict[lbl] = float(prob)

                entail_score = scores_dict.get("entailment", 0.0)
                contra_score = scores_dict.get("contradiction", 0.0)
                neutral_score = scores_dict.get("neutral", 0.0)

                if entail_score >= self.support_threshold and entail_score > contra_score:
                    best_label = "entailment"
                elif contra_score >= self.contradiction_threshold and contra_score > entail_score:
                    best_label = "contradiction"
                else:
                    argmax_idx = int(np.argmax(row_probs))
                    best_label = self.id2label.get(argmax_idx, "neutral")

                res_tuple = (
                    best_label,
                    {
                        "entailment": round(entail_score, 4),
                        "contradiction": round(contra_score, 4),
                        "neutral": round(neutral_score, 4),
                    },
                )
                self._pair_cache[(p_text, h_text)] = res_tuple
                cached_results[u_idx] = res_tuple

        return [cached_results[i] for i in range(len(pairs))]

    def validate_hypotheses_batch(
        self,
        hypotheses: List[Any],
        evidence_passages: List[Any],
    ) -> Dict[str, NLIResult]:
        """Validate directional entailment for multiple hypotheses in a single batch pass.

        Parameters
        ----------
        hypotheses : list of hypothesis entities
            Candidate hypotheses to validate.
        evidence_passages : list of evidence passages
            Retrieved evidence pool.

        Returns
        -------
        dict of str -> NLIResult
            Mapping from hypothesis_id to aggregated NLI verification result.
        """
        if not hypotheses:
            return {}

        results: Dict[str, NLIResult] = {}
        if not self.is_available() or not evidence_passages:
            for h in hypotheses:
                hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id", "unknown_hyp")
                results[hid] = NLIResult(
                    hypothesis_id=hid,
                    passage_results=[],
                    support_count=0,
                    contradiction_count=0,
                    neutral_count=0,
                    overall_alignment="neutral",
                    alignment_score=0.5,
                    model_used=self.model_name if self.is_available() else f"{self.model_name} (unavailable)",
                )
            return results

        # Flatten all (passage, hypothesis) pairs
        all_pairs: List[Tuple[str, str]] = []
        pair_to_hyp: List[Tuple[str, str]] = []  # (hid, chunk_id)

        for h in hypotheses:
            hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id", "unknown_hyp")
            htext = h.hypothesis if hasattr(h, "hypothesis") else h.get("hypothesis", "")
            if not htext:
                continue

            for ev in evidence_passages:
                p_text = ev.text_content if hasattr(ev, "text_content") else ev.get("text_content", ev.get("text", ""))
                cid = ev.chunk_id if hasattr(ev, "chunk_id") else ev.get("chunk_id", "chunk_unknown")
                if p_text:
                    all_pairs.append((str(p_text), str(htext)))
                    pair_to_hyp.append((hid, str(cid)))

        if not all_pairs:
            for h in hypotheses:
                hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id", "unknown_hyp")
                results[hid] = NLIResult(
                    hypothesis_id=hid,
                    passage_results=[],
                    support_count=0,
                    contradiction_count=0,
                    neutral_count=0,
                    overall_alignment="neutral",
                    alignment_score=0.5,
                    model_used=self.model_name,
                )
            return results

        # Single batched inference
        batch_predictions = self.predict_batch_nli(all_pairs)

        # Group results by hypothesis_id
        grouped_passages: Dict[str, List[PassageNLI]] = {
            (h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id", "unknown_hyp")): []
            for h in hypotheses
        }

        for (hid, cid), (label, scores) in zip(pair_to_hyp, batch_predictions):
            p_nli = PassageNLI(
                evidence_id=cid,
                label=label,
                entailment_score=scores["entailment"],
                contradiction_score=scores["contradiction"],
                neutral_score=scores["neutral"],
            )
            if hid in grouped_passages:
                grouped_passages[hid].append(p_nli)

        for hid, p_list in grouped_passages.items():
            sup, contra, neut, align, score = self.aggregate_results(p_list)
            results[hid] = NLIResult(
                hypothesis_id=hid,
                passage_results=p_list,
                support_count=sup,
                contradiction_count=contra,
                neutral_count=neut,
                overall_alignment=align,
                alignment_score=score,
                model_used=self.model_name,
            )

        return results'''

content = content.replace(old_predict_batch.replace("\r\n", "\n"), new_predict_batch.replace("\r\n", "\n"))

# Ensure nullcontext import if needed
if "from contextlib import nullcontext" not in content:
    content = "from contextlib import nullcontext\n" + content

NLI_PY.write_text(content, encoding="utf-8")
print("nli_validation.py patched successfully!")
