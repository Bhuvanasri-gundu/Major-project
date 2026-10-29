// frontend/src/types/index.ts — Strict TypeScript interfaces matching backend Pydantic schemas

export interface SuggestionItem {
  text: string;
  type: 'gap' | 'concept' | 'topic';
  category?: string;
  details?: string;
  score: number;
}

export interface SuggestionsResponse {
  query: string;
  count: number;
  suggestions: SuggestionItem[];
}

export interface GenerateRequest {
  theme: string;
  population?: string;
  objective?: string;
  hypothesis_count: number;
  selected_gap_id?: string;
}

export interface EvidenceSourceRecord {
  chunk_id: string;
  pmid?: string;
  paper_title?: string;
  publication_year?: number;
  journal?: string;
  doi?: string;
  source_url?: string;
  text_content: string;
  relevance_score: number;
  evidence_category: 'supporting' | 'contradictory' | 'contextual';
  verification_status: string;
}

export interface RubricScore {
  dimension: string;
  score: number;
  scale: string;
  rationale: string;
}

export interface RuleCheckItem {
  check_name: string;
  status: 'pass' | 'fail' | 'warning' | 'not_evaluated';
  explanation: string;
}

export interface ValidationDetails {
  overview: {
    status_label: string;
    passed_checks_count: number;
    failed_checks_count: number;
    warning_count: number;
    execution_status: string;
    overall_score: number;
    limitations: string[];
    uncertainties: string[];
  };
  gap_alignment: {
    gap_id: string;
    gap_statement: string;
    gap_source: string;
    relevance_rationale: string;
    aspect_addressed: string;
    alignment_score: number;
    alignment_status: string;
    limitations: string;
  };
  evidence_sources: EvidenceSourceRecord[];
  citation_audit: {
    references_checked: number;
    verified_pmids_count: number;
    invalid_pmids_count: number;
    missing_ids: string[];
    phantom_pmids: number;
    citation_precision: number;
    citation_recall: number;
    audit_method: string;
  };
  nli_validation?: {
    status: string;
    model: string;
    mean_entailment: number;
    mean_contradiction: number;
    mean_neutral: number;
    interpretation: string;
  };
  llm_judge?: {
    model: string;
    mean_score: number;
    rubrics: RubricScore[];
  };
  rule_checks: {
    passed_count: number;
    failed_count: number;
    warning_count: number;
    checks: RuleCheckItem[];
  };
  testability_assessment: {
    testability_score: number;
    rationale: string;
    suggested_study_design: string;
    operationalization_context: {
      intervention_spec?: string;
      primary_outcome?: string;
    };
    data_limitations: string;
  };
  ranking_deduplication: {
    priority_score: number;
    grounding_score: number;
    nli_component: number;
    judge_score: number;
    composite_rank_score: number;
    formula: string;
    quality_tier: string;
    deduplication_threshold: number;
    deduplication_action: string;
  };
}

export interface HypothesisOutput {
  hypothesis_id: string;
  hypothesis_number: number;
  hypothesis: string;
  target_population: string;
  proposed_mechanism?: string;
  directionality?: string;
  gap_reference: string;
  gap_id: string;
  evidence_summary?: string;
  validation_status_badge: string;
  composite_rank_score?: number;
  quality_tier?: string;
  supporting_evidence_count: number;
  contradictory_evidence_count: number;
  // Internal fields for Screen 2 validation details only:
  internal_iv?: string;
  internal_dv?: string;
  validation?: ValidationDetails;
}

export interface MatchedGap {
  gap_id: string;
  concept_a: string;
  concept_b: string;
  gap_score: number;
  semantic_similarity: number;
  temporal_growth_avg?: number;
  priority_score?: number;
  priority_rank?: number;
  category?: string;
  matching_score: number;
  matching_explanation: string;
  supporting_topic_ids: string[];
  supporting_pmids_count: number;
  description: string;
  is_precomputed: boolean;
}

export interface RunResponse {
  run_id: string;
  theme: string;
  population: string;
  objective: string;
  hypothesis_count_requested: number;
  hypothesis_count_generated: number;
  status: 'completed' | 'partially_completed' | 'no_hypotheses_requested' | 'failed';
  current_stage: string;
  started_at: string;
  completed_at?: string;
  stages_completed: string[];
  warnings: string[];
  failures: string[];
  matched_gaps: MatchedGap[];
  hypotheses: HypothesisOutput[];
}

export interface EvaluationData {
  ir_metrics: {
    precision_at_5?: number;
    precision_at_10?: number;
    mean_reciprocal_rank?: number;
    ndcg_at_10?: number;
  };
  grounding_audit: {
    citation_validity_rate?: number;
    traceability_rate?: number;
    fabricated_citations?: number;
  };
  baseline_comparisons: Array<{
    baseline_name: string;
    method: string;
    comparison_metric: string;
    baseline_score: number;
    system_score: number;
    relative_gain: string;
  }>;
  ablation_study: Array<{
    configuration: string;
    composite_score: number;
    relative_utility: string;
    impact_description: string;
  }>;
  summary_table: Array<Record<string, any>>;
}
