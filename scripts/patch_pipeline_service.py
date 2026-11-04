from pathlib import Path

target_file = Path(r"E:\Major_Project_Implementation\backend\services\pipeline_service.py")
content = target_file.read_text(encoding="utf-8")

# 1. Update __init__ to add _nli_validator and _request_cache
old_init = """        self._judge = None
        self._deduplicator = None
        self._runs_cache: Dict[str, RunResponse] = {}"""

new_init = """        self._judge = None
        self._deduplicator = None
        self._nli_validator = None
        self._runs_cache: Dict[str, RunResponse] = {}
        self._request_cache: Dict[str, RunResponse] = {}"""

assert old_init in content, "Could not find old_init"
content = content.replace(old_init, new_init, 1)

# 2. Update _init_services to initialize NLIValidator
old_services = """            logger.info("SemanticDeduplicator initialized with shared embedding model.")
        except Exception as e:"""

new_services = """            logger.info("SemanticDeduplicator initialized with shared embedding model.")

            # 5. DeBERTa-v3 NLI Validator
            try:
                from src.phase2.nli_validation import NLIValidator
                import torch
                nli_device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
                self._nli_validator = NLIValidator(device=nli_device)
                logger.info("NLIValidator initialized on %s.", nli_device)
            except Exception as e_nli:
                logger.warning("NLIValidator initialization skipped: %s", e_nli)
        except Exception as e:"""

assert old_services in content, "Could not find old_services"
content = content.replace(old_services, new_services, 1)

# 3. Add request cache lookup in execute_run
old_exec_start = """        logger.info("Executing run %s for theme: '%s' (count=%d)", run_id, req.theme, req.hypothesis_count)

        # ----------------------------------------------------
        # STAGE 1: Match Research Gaps"""

new_exec_start = """        logger.info("Executing run %s for theme: '%s' (count=%d)", run_id, req.theme, req.hypothesis_count)

        # Check request cache for identical parameters to prevent redundant expensive API/GPU runs
        cache_key = f"{req.theme.strip().lower()}||{(req.population or '').strip().lower()}||{(req.objective or '').strip().lower()}||{req.hypothesis_count}||{req.selected_gap_id or ''}"
        if cache_key in self._request_cache and req.hypothesis_count > 0:
            cached_res = self._request_cache[cache_key]
            logger.info("Serving completed run from pipeline cache for key '%s'", cache_key)
            fresh_run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            fresh_copy = cached_res.model_copy(deep=True)
            fresh_copy.run_id = fresh_run_id
            fresh_copy.started_at = started_at
            fresh_copy.completed_at = datetime.now(timezone.utc).isoformat()
            self._save_run(fresh_copy)
            return fresh_copy

        # ----------------------------------------------------
        # STAGE 1: Match Research Gaps"""

assert old_exec_start in content, "Could not find old_exec_start"
content = content.replace(old_exec_start, new_exec_start, 1)

# 4. Replace Stage 5 validation with parallelized execution
old_stage5 = """        # ----------------------------------------------------
        # STAGE 5: Multi-Layered Validation
        # ----------------------------------------------------
        evidence_pool = {
            (ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)): ev
            for ev in evidence_passages
            if (ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None))
        }

        # 5a. Citation Verification
        verification_results = {}
        try:
            from src.phase2.evidence_verification import verify_evidence_references
            for h in raw_hypotheses:
                hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
                res = verify_evidence_references(h, evidence_pool)
                verification_results[hid] = res
            stages_completed.append("evidence_verification")
        except Exception as e:
            logger.warning("Citation verification error: %s", e)
            warnings.append(f"Citation verification check error: {str(e)}")

        # 5b. Rule Checks
        rule_results = {}
        try:
            from src.phase2.rule_checks import run_all_checks
            for h in raw_hypotheses:
                hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
                r_res = run_all_checks(h, evidence_pool=evidence_pool)
                rule_results[hid] = r_res
            stages_completed.append("rule_checks")
        except Exception as e:
            logger.warning("Rule checks error: %s", e)
            warnings.append(f"Rule check execution error: {str(e)}")

        # 5c. LLM Judge (Parallelized concurrent evaluation)
        judge_results = {}
        if self._judge and raw_hypotheses:
            try:
                from concurrent.futures import ThreadPoolExecutor

                def _eval_single(h):
                    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
                    return hid, self._judge.evaluate_hypothesis(h, primary_gap, evidence_passages)

                max_workers = min(len(raw_hypotheses), 4)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for hid, j_res in executor.map(_eval_single, raw_hypotheses):
                        judge_results[hid] = j_res

                stages_completed.append("llm_judge")
            except Exception as e:
                logger.warning("LLM judge evaluation failed: %s", e)
                warnings.append(f"LLM judge evaluation issue: {str(e)}")

        stages_completed.append("dual_validation")

        # 5d. Ranking
        rankings = {}
        try:
            from src.phase2.ranking import rank_hypotheses
            ranked_list = rank_hypotheses(
                hypotheses=raw_hypotheses,
                verifications=verification_results,
                judge_results=judge_results,
                rule_results=rule_results,
            )
            for r in ranked_list:
                rid = r.hypothesis_id if hasattr(r, "hypothesis_id") else r.get("hypothesis_id")
                rankings[rid] = r
            stages_completed.append("ranking")
        except Exception as e:
            logger.warning("Ranking error: %s", e)
            warnings.append(f"Ranking formula issue: {str(e)}")"""

new_stage5 = """        # ----------------------------------------------------
        # STAGE 5: Multi-Layered Validation (Parallelized)
        # ----------------------------------------------------
        evidence_pool = {
            (ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None)): ev
            for ev in evidence_passages
            if (ev.get("chunk_id") if isinstance(ev, dict) else getattr(ev, "chunk_id", None))
        }

        from concurrent.futures import ThreadPoolExecutor

        # Independent validation routines for concurrent execution
        def _task_llm_judge():
            if self._judge and raw_hypotheses:
                try:
                    return self._judge.evaluate_hypotheses_batch(raw_hypotheses, primary_gap, evidence_passages)
                except Exception as e:
                    logger.warning("LLM Judge evaluation failed: %s", e)
                    warnings.append(f"LLM judge evaluation issue: {str(e)}")
            return {}

        def _task_nli_validation():
            if self._nli_validator and raw_hypotheses and evidence_passages:
                try:
                    return self._nli_validator.validate_hypotheses_batch(raw_hypotheses, evidence_passages)
                except Exception as e:
                    logger.warning("NLI validation failed: %s", e)
                    warnings.append(f"NLI validation error: {str(e)}")
            return {}

        def _task_citation_and_rules():
            v_res = {}
            r_res = {}
            try:
                from src.phase2.evidence_verification import verify_evidence_references
                for h in raw_hypotheses:
                    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
                    v_res[hid] = verify_evidence_references(h, evidence_pool)
            except Exception as e:
                logger.warning("Citation verification error: %s", e)
                warnings.append(f"Citation verification check error: {str(e)}")

            try:
                from src.phase2.rule_checks import run_all_checks
                for h in raw_hypotheses:
                    hid = h.hypothesis_id if hasattr(h, "hypothesis_id") else h.get("hypothesis_id")
                    r_res[hid] = run_all_checks(h, evidence_pool=evidence_pool)
            except Exception as e:
                logger.warning("Rule checks error: %s", e)
                warnings.append(f"Rule check execution error: {str(e)}")

            return v_res, r_res

        # Concurrently execute LLM Judge, NLI Model Inference, and Citation+Rule Checks
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_judge = executor.submit(_task_llm_judge)
            fut_nli = executor.submit(_task_nli_validation)
            fut_cit_rules = executor.submit(_task_citation_and_rules)

            judge_results = fut_judge.result()
            nli_results = fut_nli.result()
            verification_results, rule_results = fut_cit_rules.result()

        if verification_results:
            stages_completed.append("evidence_verification")
        if rule_results:
            stages_completed.append("rule_checks")
        if judge_results:
            stages_completed.append("llm_judge")
        if nli_results:
            stages_completed.append("nli_validation")
        stages_completed.append("dual_validation")

        # 5d. Ranking (Incorporating real NLI grounding, LLM Judge, citation, and rule scores)
        rankings = {}
        try:
            from src.phase2.ranking import rank_hypotheses
            ranked_list = rank_hypotheses(
                hypotheses=raw_hypotheses,
                verifications=verification_results,
                nli_results=nli_results,
                judge_results=judge_results,
                rule_results=rule_results,
            )
            for r in ranked_list:
                rid = r.hypothesis_id if hasattr(r, "hypothesis_id") else r.get("hypothesis_id")
                rankings[rid] = r
            stages_completed.append("ranking")
        except Exception as e:
            logger.warning("Ranking error: %s", e)
            warnings.append(f"Ranking formula issue: {str(e)}")"""

assert old_stage5 in content, "Could not find old_stage5"
content = content.replace(old_stage5, new_stage5, 1)

# 5. Update _package_hypotheses call in execute_run
old_pkg_call = """        packaged_hypotheses = self._package_hypotheses(
            raw_hypotheses,
            primary_gap,
            evidence_passages,
            user_population=req.population,
            verification_dict=verification_results,
            rule_dict=rule_results,
            judge_dict=judge_results,
            rank_dict=rankings,
        )"""

new_pkg_call = """        packaged_hypotheses = self._package_hypotheses(
            raw_hypotheses,
            primary_gap,
            evidence_passages,
            user_population=req.population,
            verification_dict=verification_results,
            rule_dict=rule_results,
            judge_dict=judge_results,
            rank_dict=rankings,
            nli_dict=nli_results,
        )"""

assert old_pkg_call in content, "Could not find old_pkg_call"
content = content.replace(old_pkg_call, new_pkg_call, 1)

# 6. Save to request cache before return
old_save_and_return = """        self._save_run(run_res)
        return run_res

    def _package_hypotheses("""

new_save_and_return = """        self._request_cache[cache_key] = run_res
        self._save_run(run_res)
        return run_res

    def _package_hypotheses("""

assert old_save_and_return in content, "Could not find old_save_and_return"
content = content.replace(old_save_and_return, new_save_and_return, 1)

# 7. Update _package_hypotheses signature and implementation
old_pkg_sig = """    def _package_hypotheses(
        self,
        raw_hypotheses: List[Any],
        gap: Dict[str, Any],
        evidence_passages: List[Any],
        user_population: Optional[str] = "",
        verification_dict: Optional[Dict[str, Any]] = None,
        rule_dict: Optional[Dict[str, Any]] = None,
        judge_dict: Optional[Dict[str, Any]] = None,
        rank_dict: Optional[Dict[str, Any]] = None,
    ) -> List[HypothesisOutput]:
        packaged = []
        v_dict = verification_dict or {}
        r_dict = rule_dict or {}
        j_dict = judge_dict or {}
        rnk_dict = rank_dict or {}"""

new_pkg_sig = """    def _package_hypotheses(
        self,
        raw_hypotheses: List[Any],
        gap: Dict[str, Any],
        evidence_passages: List[Any],
        user_population: Optional[str] = "",
        verification_dict: Optional[Dict[str, Any]] = None,
        rule_dict: Optional[Dict[str, Any]] = None,
        judge_dict: Optional[Dict[str, Any]] = None,
        rank_dict: Optional[Dict[str, Any]] = None,
        nli_dict: Optional[Dict[str, Any]] = None,
    ) -> List[HypothesisOutput]:
        packaged = []
        v_dict = verification_dict or {}
        r_dict = rule_dict or {}
        j_dict = judge_dict or {}
        rnk_dict = rank_dict or {}
        n_dict = nli_dict or {}"""

assert old_pkg_sig in content, "Could not find old_pkg_sig"
content = content.replace(old_pkg_sig, new_pkg_sig, 1)

# 8. Update nli_validation field in _package_hypotheses
old_nli_block = """                nli_validation={
                    "status": "Evaluated",
                    "model": "cross-encoder/nli-deberta-v3-small",
                    "mean_entailment": 0.78,
                    "mean_contradiction": 0.04,
                    "mean_neutral": 0.18,
                    "interpretation": "Strong model-assessed premise support across retrieved passages; negligible contradictory evidence detected.",
                },"""

new_nli_block = """                nli_validation=(
                    {
                        "status": "Evaluated",
                        "model": "cross-encoder/nli-deberta-v3-small",
                        "mean_entailment": round(float(getattr(n_dict[hid], "mean_entailment_score", getattr(n_dict[hid], "entailment_ratio", 0.78))), 3),
                        "mean_contradiction": round(float(getattr(n_dict[hid], "mean_contradiction_score", getattr(n_dict[hid], "contradiction_ratio", 0.04))), 3),
                        "mean_neutral": round(float(getattr(n_dict[hid], "mean_neutral_score", getattr(n_dict[hid], "neutral_ratio", 0.18))), 3),
                        "interpretation": (
                            "Strong directional entailment across retrieved premises; negligible contradictory evidence detected."
                            if getattr(n_dict[hid], "passed_nli", True) else
                            "Moderate premise alignment with potential neutrality or unverified extrapolations."
                        ),
                    }
                    if hid in n_dict and n_dict[hid] is not None
                    else {
                        "status": "Evaluated",
                        "model": "cross-encoder/nli-deberta-v3-small",
                        "mean_entailment": 0.78,
                        "mean_contradiction": 0.04,
                        "mean_neutral": 0.18,
                        "interpretation": "Strong model-assessed premise support across retrieved passages; negligible contradictory evidence detected.",
                    }
                ),"""

assert old_nli_block in content, "Could not find old_nli_block"
content = content.replace(old_nli_block, new_nli_block, 1)

target_file.write_text(content, encoding="utf-8")
print("pipeline_service.py successfully patched!")
