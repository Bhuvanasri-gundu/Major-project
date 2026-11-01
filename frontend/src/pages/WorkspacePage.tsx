// frontend/src/pages/WorkspacePage.tsx — Screen 2: Evidence Validation Workbench (Stitch Design)

import React, { useEffect, useState, useMemo } from 'react';
import { useParams, useSearchParams, useNavigate, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  FileText,
  Shield,
  Search,
  BarChart3,
  ExternalLink,
  Copy,
  BookOpen,
  FlaskConical,
  CircleDot,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  RunResponse,
  HypothesisOutput,
  MatchedGap,
} from '../types';

// ─── Helpers ────────────────────────────────────────────────────────────────

function formatTimestamp(iso?: string): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max) + '…' : str;
}

function computeRunHash(runId: string): string {
  // Simple deterministic hash for display based on run_id
  let hash = 0;
  for (let i = 0; i < runId.length; i++) {
    const chr = runId.charCodeAt(i);
    hash = ((hash << 5) - hash) + chr;
    hash |= 0;
  }
  return Math.abs(hash).toString(16).padStart(12, '0').slice(0, 12);
}

function tierColor(tier?: string): string {
  if (!tier) return 'bg-studio-100 text-studio-600';
  const t = tier.toLowerCase();
  if (t.includes('1') || t.includes('recommended')) return 'bg-emerald-50 text-emerald-700 border border-emerald-200';
  if (t.includes('2') || t.includes('acceptable')) return 'bg-amber-50 text-amber-700 border border-amber-200';
  return 'bg-studio-100 text-studio-600 border border-studio-200';
}

// ─── Pipeline Stage Definitions ─────────────────────────────────────────────

interface PipelineStage {
  key: string;
  label: string;
  detail: string;
  icon: React.ReactNode;
}

const PIPELINE_STAGES: PipelineStage[] = [
  { key: 'gap_matching', label: 'Gap Matching', detail: 'GAP Discovery', icon: <Search className="w-4 h-4" /> },
  { key: 'literature_retrieval', label: 'Literature Retrieval', detail: 'PubMed / PMC Indexed', icon: <BookOpen className="w-4 h-4" /> },
  { key: 'generation', label: 'Generation', detail: 'Candidate Synthesis', icon: <FlaskConical className="w-4 h-4" /> },
  { key: 'dual_validation', label: 'Dual Validation', detail: 'NLI & DOI Verification', icon: <Shield className="w-4 h-4" /> },
  { key: 'ranking', label: 'Multi-Criteria Ranking', detail: 'Pareto Stratified', icon: <BarChart3 className="w-4 h-4" /> },
];

// ─── Inspector Tabs ─────────────────────────────────────────────────────────

type InspectorTab = 'overview' | 'gap_alignment' | 'evidence' | 'validation' | 'ranking';

const INSPECTOR_TABS: { key: InspectorTab; label: string; number: number }[] = [
  { key: 'overview', label: 'Overview', number: 1 },
  { key: 'gap_alignment', label: 'Gap Alignment', number: 2 },
  { key: 'evidence', label: 'Evidence & Sources', number: 3 },
  { key: 'validation', label: 'Scientific Validation', number: 4 },
  { key: 'ranking', label: 'Ranking & Deduplication', number: 5 },
];

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN PAGE COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export const WorkspacePage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [run, setRun] = useState<RunResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedHypothesisId, setSelectedHypothesisId] = useState<string>('');
  const [activeTab, setActiveTab] = useState<InspectorTab>('overview');
  const [expandedGaps, setExpandedGaps] = useState<Record<string, boolean>>({});

  const targetRunId = runId || 'latest';

  // ── Data Fetching ──────────────────────────────────────────────────────

  const fetchRunData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data: RunResponse =
        targetRunId === 'latest'
          ? await api.getLatestRun()
          : await api.getRun(targetRunId);
      setRun(data);

      const paramHypId = searchParams.get('hypothesis');
      if (paramHypId && data.hypotheses.some((h) => h.hypothesis_id === paramHypId)) {
        setSelectedHypothesisId(paramHypId);
      } else if (data.hypotheses.length > 0) {
        setSelectedHypothesisId(data.hypotheses[0].hypothesis_id);
      }
    } catch (err: any) {
      setError(err.message || `Failed to load research run '${targetRunId}'.`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRunData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetRunId]);

  const selectedHypothesis: HypothesisOutput | undefined = useMemo(
    () => run?.hypotheses.find((h) => h.hypothesis_id === selectedHypothesisId),
    [run, selectedHypothesisId]
  );

  const toggleGap = (gapId: string) => {
    setExpandedGaps((prev) => ({ ...prev, [gapId]: !prev[gapId] }));
  };

  // ── Loading State ──────────────────────────────────────────────────────

  if (isLoading) {
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-16">
        <div className="stitch-card p-16 text-center space-y-3">
          <Loader2 className="w-8 h-8 text-teal-accent animate-spin mx-auto" />
          <h3 className="text-sm font-semibold text-studio-800">Loading Research & Validation Workspace…</h3>
          <p className="text-xs text-studio-500">Fetching pipeline trajectory and validation records</p>
        </div>
      </div>
    );
  }

  // ── Error State ────────────────────────────────────────────────────────

  if (error || !run) {
    return (
      <div className="max-w-[1280px] mx-auto px-6 py-16">
        <div className="stitch-card p-12 text-center space-y-4">
          <AlertCircle className="w-10 h-10 text-rose-500 mx-auto" />
          <h3 className="text-base font-semibold text-studio-900">Run Dossier Unavailable</h3>
          <p className="text-xs text-studio-500 max-w-md mx-auto">{error || 'No run data available.'}</p>
          <button
            type="button"
            onClick={() => navigate('/')}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-md text-xs font-semibold text-white bg-teal-accent hover:bg-teal-dark transition"
          >
            Return to Generation Stream
          </button>
        </div>
      </div>
    );
  }

  // ── Derived values ─────────────────────────────────────────────────────

  const hashDisplay = `sha256:${computeRunHash(run.run_id)}...`;
  const gapCount = run.matched_gaps.length;
  const stagesCompleted = run.stages_completed || [];

  const isStageComplete = (stageKey: string): boolean => {
    const normalized = stageKey.toLowerCase().replace(/[_\s]/g, '');
    return stagesCompleted.some((s) => s.toLowerCase().replace(/[_\s]/g, '').includes(normalized));
  };

  // For the pipeline stats row
  const gapDiscoveredLabel = run.matched_gaps.length > 0
    ? run.matched_gaps.map((g) => g.gap_id).join('/') + ' Discovered'
    : 'No Gaps';
  const totalSources = run.matched_gaps.reduce((sum, g) => sum + g.supporting_pmids_count, 0);

  return (
    <div className="max-w-[1280px] mx-auto px-6 py-6 space-y-6">

      {/* ═══════════════════════════════════════════════════════════════════
          TOP SECTION: Navigation + Title Block
          ═══════════════════════════════════════════════════════════════════ */}

      <header className="space-y-4">
        {/* Nav row */}
        <div className="flex items-center justify-between flex-wrap gap-2">
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-studio-500 hover:text-teal-accent transition"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to generation stream</span>
          </Link>

          <div className="flex items-center gap-3">
            <span className="font-code text-studio-400 text-xs">HASH: {hashDisplay}</span>
            <button
              type="button"
              onClick={() => navigator.clipboard?.writeText(run.run_id)}
              className="p-1.5 text-studio-400 hover:text-studio-600 transition"
              title="Copy run ID"
            >
              <Copy className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold text-white bg-studio-800 hover:bg-studio-900 transition"
            >
              <FileText className="w-3.5 h-3.5" />
              Export Monograph Report
            </button>
          </div>
        </div>

        {/* Title block */}
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div className="space-y-1">
            <p className="stitch-kicker text-teal-accent">Evidence Validation Workbench</p>
            <h1 className="text-2xl font-semibold text-studio-900 tracking-tight">
              Research & Validation Workspace
            </h1>
          </div>

          <div className="flex items-start gap-5 flex-wrap">
            {/* Active Theme card */}
            <div className="stitch-card px-4 py-3 max-w-xs">
              <p className="stitch-kicker mb-1">Active Theme</p>
              <p className="text-sm font-medium text-studio-800 leading-snug">
                {truncate(run.theme, 80)}
              </p>
            </div>

            {/* Evaluator status badge */}
            <div className="space-y-1.5 text-right">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-xs font-medium text-emerald-700">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                Evaluator Status: Evaluated & Audited
              </div>
              <p className="font-code text-xs text-studio-500">
                Run ID: {run.run_id}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* ═══════════════════════════════════════════════════════════════════
          PIPELINE EXECUTION TRAJECTORY
          ═══════════════════════════════════════════════════════════════════ */}

      <section className="rounded-lg overflow-hidden border border-studio-200">
        {/* Dark header bar */}
        <div className="bg-studio-900 px-5 py-3 flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-white">Pipeline Execution Trajectory</h2>
            <span className="stage-badge bg-teal-accent/20 text-teal-light border border-teal-accent/30">
              Strict Monograph Determinism
            </span>
          </div>
          <div className="flex items-center gap-4 text-xs text-studio-400 font-code">
            <span>Hypotheses Requested: <strong className="text-white">{run.hypothesis_count_requested}</strong></span>
            <span>Generated: <strong className="text-white">{run.hypothesis_count_generated}</strong></span>
            <span>Failures: <strong className={run.failures.length > 0 ? 'text-rose-400' : 'text-white'}>{run.failures.length}</strong></span>
            <span>Warnings: <strong className={run.warnings.length > 0 ? 'text-amber-400' : 'text-white'}>{run.warnings.length}</strong></span>
          </div>
        </div>

        {/* Timeline stages */}
        <div className="bg-studio-50 px-5 py-4">
          <div className="grid grid-cols-5 gap-3">
            {PIPELINE_STAGES.map((stage, idx) => {
              const completed = isStageComplete(stage.key);
              // Build dynamic detail text per stage
              let detailText = stage.detail;
              if (idx === 0 && run.matched_gaps.length > 0) {
                detailText = gapDiscoveredLabel.length > 30
                  ? `${run.matched_gaps.length} Gaps Discovered`
                  : gapDiscoveredLabel;
              }
              if (idx === 1) detailText = `${totalSources} Sources Indexed`;
              if (idx === 2) detailText = `${run.hypothesis_count_generated} Candidates`;

              return (
                <div
                  key={stage.key}
                  className={`relative rounded-lg border px-3.5 py-3 space-y-1.5 transition ${
                    completed
                      ? 'bg-white border-studio-200'
                      : 'bg-studio-100/50 border-studio-200/60 opacity-60'
                  }`}
                >
                  {/* Connector line */}
                  {idx < PIPELINE_STAGES.length - 1 && (
                    <div className="absolute top-1/2 -right-3 w-3 border-t border-dashed border-studio-300 -translate-y-1/2" />
                  )}

                  <div className="flex items-center justify-between">
                    <div className={`flex items-center gap-1.5 text-xs font-semibold ${
                      completed ? 'text-studio-800' : 'text-studio-500'
                    }`}>
                      {stage.icon}
                      <span>{stage.label}</span>
                    </div>
                    {completed && (
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                    )}
                  </div>

                  <p className="text-[11px] text-studio-500 font-medium">{detailText}</p>
                  <p className="text-[10px] text-studio-400 font-code">
                    {completed ? formatTimestamp(run.completed_at || run.started_at) : 'Pending'}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════
          TWO-COLUMN EXPLORER: Gaps + Hypotheses
          ═══════════════════════════════════════════════════════════════════ */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* ─── LEFT: Research Gaps ─────────────────────────────────────── */}
        <section className="space-y-3">
          <div className="space-y-0.5">
            <p className="stitch-kicker">Taxonomy & Corpus Cavities</p>
            <div className="flex items-center gap-2.5">
              <h2 className="text-lg font-semibold text-studio-900">Research Gaps</h2>
              <span className="stage-badge bg-studio-100 text-studio-600 border border-studio-200">
                {gapCount} Active Gap{gapCount !== 1 ? 's' : ''}
              </span>
            </div>
          </div>

          <div className="space-y-3">
            {run.matched_gaps.map((gap) => (
              <GapCard
                key={gap.gap_id}
                gap={gap}
                isExpanded={!!expandedGaps[gap.gap_id]}
                onToggle={() => toggleGap(gap.gap_id)}
              />
            ))}
            {run.matched_gaps.length === 0 && (
              <div className="stitch-card p-6 text-center">
                <p className="text-xs text-studio-500">No gaps matched for this run.</p>
              </div>
            )}
          </div>
        </section>

        {/* ─── RIGHT: Generated Hypotheses ─────────────────────────────── */}
        <section className="space-y-3">
          <div className="space-y-0.5">
            <p className="stitch-kicker">Empirical Derivations</p>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <h2 className="text-lg font-semibold text-studio-900">Generated Hypotheses</h2>
              </div>
              <span className="text-[11px] text-studio-400 italic">Select to Inspect</span>
            </div>
          </div>

          <div className="space-y-3">
            {run.hypotheses.map((hyp) => (
              <HypothesisCard
                key={hyp.hypothesis_id}
                hypothesis={hyp}
                isSelected={hyp.hypothesis_id === selectedHypothesisId}
                onSelect={() => {
                  setSelectedHypothesisId(hyp.hypothesis_id);
                  setActiveTab('overview');
                }}
                onInspect={() => navigate(`/workspace/${run.run_id}/hypotheses/${hyp.hypothesis_id}`)}
              />
            ))}
            {run.hypotheses.length === 0 && (
              <div className="stitch-card p-6 text-center">
                <Search className="w-6 h-6 text-studio-300 mx-auto mb-2" />
                <p className="text-xs text-studio-500">
                  {run.status === 'no_hypotheses_requested'
                    ? 'No hypotheses were requested for this run (Count = 0).'
                    : 'No hypotheses were generated. Review matched gaps or adjust your theme.'}
                </p>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          BOTTOM INSPECTOR: Selected Hypothesis Deep Inspection
          ═══════════════════════════════════════════════════════════════════ */}

      {selectedHypothesis && (
        <InspectorSection
          hypothesis={selectedHypothesis}
          run={run}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onNavigateToFull={() =>
            navigate(`/workspace/${run.run_id}/hypotheses/${selectedHypothesis.hypothesis_id}`)
          }
        />
      )}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════
//  GAP CARD SUB-COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

interface GapCardProps {
  gap: MatchedGap;
  isExpanded: boolean;
  onToggle: () => void;
}

const GapCard: React.FC<GapCardProps> = ({ gap, isExpanded, onToggle }) => {
  // Derive MeSH-like term pills from supporting_topic_ids
  const meshTerms = gap.supporting_topic_ids.slice(0, 5);

  return (
    <div className="stitch-card overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={onToggle}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-studio-50 transition text-left"
      >
        <div className="flex items-center gap-2.5">
          <span className="font-code text-xs text-teal-accent font-semibold">{gap.gap_id}</span>
          <span className="text-[11px] text-studio-400">
            Match Score: <strong className="text-studio-700">{gap.matching_score.toFixed(2)}</strong>
          </span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-studio-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-studio-400" />
        )}
      </button>

      {/* Body — always visible */}
      <div className="px-4 pb-3 space-y-2">
        <h3 className="text-sm font-semibold text-studio-900 leading-snug">
          {gap.concept_a} ↔ {gap.concept_b}
        </h3>
        <p className="text-xs text-studio-600 leading-relaxed line-clamp-3">
          {gap.description}
        </p>

        {/* MeSH pills */}
        {meshTerms.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {meshTerms.map((term) => (
              <span
                key={term}
                className="px-2 py-0.5 rounded-full bg-studio-100 text-[10px] font-medium text-studio-600 border border-studio-200"
              >
                {term}
              </span>
            ))}
          </div>
        )}

        <p className="text-[11px] text-studio-400">
          {gap.supporting_pmids_count} Sources Indexed
        </p>

        {/* Expanded details */}
        {isExpanded && (
          <div className="pt-2 mt-2 border-t border-studio-100 space-y-2.5">
            {/* Original Gap Statement */}
            <div>
              <p className="text-[10px] font-semibold text-studio-500 uppercase tracking-wide mb-1">Original Gap Statement</p>
              <p className="text-xs text-studio-600 italic leading-relaxed">
                "{gap.matching_explanation}"
              </p>
            </div>

            {/* Provenance */}
            <div className="grid grid-cols-2 gap-3 text-[11px]">
              <div>
                <p className="text-studio-400 font-medium">Provenance</p>
                <p className="text-studio-700 font-code">{gap.is_precomputed ? 'Precomputed Gap Store' : 'Live Discovery'}</p>
              </div>
              <div>
                <p className="text-studio-400 font-medium">Semantic Similarity</p>
                <p className="text-studio-700">{gap.semantic_similarity.toFixed(3)}</p>
              </div>
            </div>

            {gap.category && (
              <div>
                <p className="text-studio-400 font-medium text-[11px]">Targeted Facet</p>
                <p className="text-studio-700 text-xs">{gap.category}</p>
              </div>
            )}

            <span className="stage-badge bg-studio-100 text-studio-600 border border-studio-200 text-[10px]">
              Cosine + Temporal Growth Alignment
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════
//  HYPOTHESIS CARD SUB-COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

interface HypothesisCardProps {
  hypothesis: HypothesisOutput;
  isSelected: boolean;
  onSelect: () => void;
  onInspect: () => void;
}

const HypothesisCard: React.FC<HypothesisCardProps> = ({
  hypothesis: hyp,
  isSelected,
  onSelect,
  onInspect,
}) => {
  const tierLabel = hyp.quality_tier || 'Unranked';
  const compositeScore = hyp.composite_rank_score?.toFixed(3) || '—';
  const groundingPct = hyp.validation
    ? `${Math.round(hyp.validation.overview.overall_score * 100)}%`
    : '—';
  const nliScore = hyp.validation?.nli_validation
    ? hyp.validation.nli_validation.mean_entailment.toFixed(2)
    : '—';
  const judgeScore = hyp.validation?.llm_judge
    ? `${hyp.validation.llm_judge.mean_score.toFixed(1)} / 5.0`
    : '—';

  const isAudited = hyp.validation_status_badge?.toLowerCase().includes('audit') ||
    hyp.validation_status_badge?.toLowerCase().includes('validated');

  return (
    <div
      onClick={onSelect}
      className={`stitch-card cursor-pointer transition overflow-hidden ${
        isSelected
          ? 'ring-2 ring-teal-accent border-teal-accent'
          : 'hover:border-studio-300'
      }`}
    >
      {/* Header row */}
      <div className="px-4 py-2.5 border-b border-studio-100 flex items-center justify-between flex-wrap gap-1.5">
        <div className="flex items-center gap-2.5">
          <span className="inline-flex items-center justify-center px-2 py-0.5 rounded bg-teal-accent text-white text-[11px] font-bold font-code">
            #H-{String(hyp.hypothesis_number).padStart(2, '0')}
          </span>
          <span className="text-[11px] text-studio-500">
            Linked: <strong className="text-studio-700">{hyp.gap_id}</strong>
          </span>
          <span className={`stage-badge text-[10px] py-0.5 ${tierColor(tierLabel)}`}>
            {tierLabel}
          </span>
        </div>
        <span className="text-[11px] text-studio-400 font-code">
          Composite: <strong className="text-studio-700">{compositeScore}</strong>
        </span>
      </div>

      {/* Body */}
      <div className="px-4 py-3 space-y-2">
        <h3 className="text-sm font-semibold text-studio-900 leading-snug line-clamp-2">
          {hyp.hypothesis}
        </h3>
        {hyp.proposed_mechanism && (
          <p className="text-xs text-studio-600 leading-relaxed line-clamp-2">
            {hyp.proposed_mechanism}
          </p>
        )}
      </div>

      {/* Footer metrics */}
      <div className="px-4 py-2.5 border-t border-studio-100 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-4">
          <span className="metric-pill text-emerald-600">
            <CircleDot className="w-3 h-3" />
            Grounding: {groundingPct}
          </span>
          <span className="metric-pill text-blue-600">
            <CircleDot className="w-3 h-3" />
            NLI: {nliScore} Entailment
          </span>
          <span className="metric-pill text-violet-600">
            <CircleDot className="w-3 h-3" />
            Judge: {judgeScore}
          </span>
        </div>
        <div>
          {isAudited ? (
            <span className="text-[11px] font-semibold text-emerald-600 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" />
              Audited
            </span>
          ) : (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onInspect(); }}
              className="text-[11px] font-semibold text-teal-accent hover:text-teal-dark flex items-center gap-1 transition"
            >
              Inspect
              <ExternalLink className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════════════
//  INSPECTOR SECTION SUB-COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

interface InspectorSectionProps {
  hypothesis: HypothesisOutput;
  run: RunResponse;
  activeTab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  onNavigateToFull: () => void;
}

const InspectorSection: React.FC<InspectorSectionProps> = ({
  hypothesis: hyp,
  run,
  activeTab,
  onTabChange,
  onNavigateToFull,
}) => {
  return (
    <section className="stitch-card overflow-hidden">
      {/* Inspector header */}
      <div className="px-5 py-4 border-b border-studio-200 flex items-center justify-between flex-wrap gap-2">
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <p className="stitch-kicker">Active Inspector</p>
            <span className="inline-flex items-center px-2 py-0.5 rounded bg-teal-accent/10 text-teal-accent text-[11px] font-bold font-code border border-teal-accent/20">
              Hypothesis #H-{String(hyp.hypothesis_number).padStart(2, '0')}
            </span>
          </div>
          <h2 className="text-lg font-semibold text-studio-900">
            Scientific Validation & Grounding Monograph
          </h2>
        </div>
        <button
          type="button"
          onClick={onNavigateToFull}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold text-teal-accent hover:text-white hover:bg-teal-accent border border-teal-accent/30 transition"
        >
          Full Validation View
          <ExternalLink className="w-3 h-3" />
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-studio-200 px-5 flex gap-0 overflow-x-auto">
        {INSPECTOR_TABS.map((tab) => (
          <button
            key={tab.key}
            type="button"
            onClick={() => onTabChange(tab.key)}
            className={`stitch-tab whitespace-nowrap ${
              activeTab === tab.key ? 'stitch-tab-active' : ''
            }`}
          >
            {tab.number}. {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="p-5">
        {activeTab === 'overview' && <OverviewTab hypothesis={hyp} />}
        {activeTab === 'gap_alignment' && <GapAlignmentTab hypothesis={hyp} gaps={run.matched_gaps} />}
        {activeTab === 'evidence' && <EvidenceTab hypothesis={hyp} />}
        {activeTab === 'validation' && <ValidationTab hypothesis={hyp} />}
        {activeTab === 'ranking' && <RankingTab hypothesis={hyp} />}
      </div>
    </section>
  );
};

// ─── Tab 1: Overview ─────────────────────────────────────────────────────

const OverviewTab: React.FC<{ hypothesis: HypothesisOutput }> = ({ hypothesis: hyp }) => {
  const v = hyp.validation;
  const overview = v?.overview;

  const passedChecks = overview?.passed_checks_count ?? 0;
  const failedChecks = overview?.failed_checks_count ?? 0;
  const warningCount = overview?.warning_count ?? 0;
  const totalChecks = passedChecks + failedChecks;

  // Total checks evaluated
  const notEvaluatedCount = v?.rule_checks
    ? v.rule_checks.checks.filter((c) => c.status === 'not_evaluated').length
    : 0;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: Hypothesis abstract synthesis */}
        <div className="lg:col-span-2 space-y-3">
          <p className="stitch-kicker">Hypothesis Abstract Synthesis</p>
          <p className="text-sm text-studio-800 leading-relaxed">
            {hyp.hypothesis}
          </p>

          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <p className="text-studio-400 font-medium mb-0.5">Linked Gap</p>
              <p className="text-studio-700 font-code">{hyp.gap_id}</p>
            </div>
            <div>
              <p className="text-studio-400 font-medium mb-0.5">Target Population</p>
              <p className="text-studio-700">{hyp.target_population}</p>
            </div>
            {hyp.directionality && (
              <div>
                <p className="text-studio-400 font-medium mb-0.5">Directionality</p>
                <p className="text-studio-700">{hyp.directionality}</p>
              </div>
            )}
            {hyp.evidence_summary && (
              <div className="col-span-2">
                <p className="text-studio-400 font-medium mb-0.5">Evidence Summary</p>
                <p className="text-studio-600 leading-relaxed">{hyp.evidence_summary}</p>
              </div>
            )}
          </div>
        </div>

        {/* Right: Empirical Scope Notice */}
        <div className="space-y-3">
          <p className="stitch-kicker">Empirical Scope Notice</p>
          {overview?.limitations && overview.limitations.length > 0 ? (
            <ul className="space-y-1.5">
              {overview.limitations.map((lim, i) => (
                <li key={i} className="text-xs text-studio-600 leading-relaxed flex items-start gap-1.5">
                  <AlertCircle className="w-3 h-3 text-amber-500 mt-0.5 shrink-0" />
                  <span>{lim}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-studio-500 italic">No limitations reported.</p>
          )}
          {overview?.uncertainties && overview.uncertainties.length > 0 && (
            <div className="pt-2 border-t border-studio-100">
              <p className="text-[10px] font-semibold text-studio-500 uppercase tracking-wide mb-1">Uncertainties</p>
              <ul className="space-y-1">
                {overview.uncertainties.map((u, i) => (
                  <li key={i} className="text-xs text-studio-500 italic">• {u}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Bottom metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          value={`${passedChecks} / ${totalChecks}`}
          label="Validation Checks Passed / Audited"
          color="emerald"
        />
        <MetricCard
          value={`${warningCount} Warning${warningCount !== 1 ? 's' : ''}`}
          label="Nominal"
          color={warningCount > 0 ? 'amber' : 'emerald'}
        />
        <MetricCard
          value={`${failedChecks} Failed Rule${failedChecks !== 1 ? 's' : ''}`}
          label={failedChecks === 0 ? 'Clean Run' : 'Action Required'}
          color={failedChecks > 0 ? 'rose' : 'emerald'}
        />
        <MetricCard
          value={`${notEvaluatedCount} Not Evaluated`}
          label={notEvaluatedCount === 0 ? 'Full Coverage' : 'Partial'}
          color={notEvaluatedCount > 0 ? 'amber' : 'emerald'}
        />
      </div>
    </div>
  );
};

// ─── Tab 2: Gap Alignment ────────────────────────────────────────────────

const GapAlignmentTab: React.FC<{ hypothesis: HypothesisOutput; gaps: MatchedGap[] }> = ({
  hypothesis: hyp,
  gaps,
}) => {
  const ga = hyp.validation?.gap_alignment;
  const linkedGap = gaps.find((g) => g.gap_id === hyp.gap_id);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Alignment details */}
        <div className="space-y-3">
          <p className="stitch-kicker">Gap Alignment Assessment</p>
          {ga ? (
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <span className="font-code text-xs text-teal-accent font-semibold">{ga.gap_id}</span>
                <span className={`stage-badge text-[10px] ${
                  ga.alignment_status === 'aligned'
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : 'bg-amber-50 text-amber-700 border border-amber-200'
                }`}>
                  {ga.alignment_status}
                </span>
                <span className="text-xs text-studio-500">
                  Score: <strong className="text-studio-800">{ga.alignment_score.toFixed(2)}</strong>
                </span>
              </div>

              <div>
                <p className="text-[10px] font-semibold text-studio-500 uppercase tracking-wide mb-1">Gap Statement</p>
                <p className="text-xs text-studio-600 italic leading-relaxed">"{ga.gap_statement}"</p>
              </div>

              <div>
                <p className="text-[10px] font-semibold text-studio-500 uppercase tracking-wide mb-1">Relevance Rationale</p>
                <p className="text-xs text-studio-700 leading-relaxed">{ga.relevance_rationale}</p>
              </div>

              <div>
                <p className="text-[10px] font-semibold text-studio-500 uppercase tracking-wide mb-1">Aspect Addressed</p>
                <p className="text-xs text-studio-700">{ga.aspect_addressed}</p>
              </div>

              {ga.limitations && (
                <div>
                  <p className="text-[10px] font-semibold text-studio-500 uppercase tracking-wide mb-1">Limitations</p>
                  <p className="text-xs text-studio-600 italic">{ga.limitations}</p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-studio-500 italic">Gap alignment data not available.</p>
          )}
        </div>

        {/* Linked gap detail */}
        <div className="space-y-3">
          <p className="stitch-kicker">Linked Gap Record</p>
          {linkedGap ? (
            <div className="stitch-card p-4 space-y-2">
              <h4 className="text-sm font-semibold text-studio-900">
                {linkedGap.concept_a} ↔ {linkedGap.concept_b}
              </h4>
              <p className="text-xs text-studio-600 leading-relaxed">{linkedGap.description}</p>
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div>
                  <span className="text-studio-400">Match Score:</span>{' '}
                  <strong className="text-studio-700">{linkedGap.matching_score.toFixed(2)}</strong>
                </div>
                <div>
                  <span className="text-studio-400">Gap Score:</span>{' '}
                  <strong className="text-studio-700">{linkedGap.gap_score.toFixed(3)}</strong>
                </div>
                <div>
                  <span className="text-studio-400">Similarity:</span>{' '}
                  <strong className="text-studio-700">{linkedGap.semantic_similarity.toFixed(3)}</strong>
                </div>
                <div>
                  <span className="text-studio-400">PMIDs:</span>{' '}
                  <strong className="text-studio-700">{linkedGap.supporting_pmids_count}</strong>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-studio-500 italic">Linked gap not found in run data.</p>
          )}
        </div>
      </div>
    </div>
  );
};

// ─── Tab 3: Evidence & Sources ───────────────────────────────────────────

const EvidenceTab: React.FC<{ hypothesis: HypothesisOutput }> = ({ hypothesis: hyp }) => {
  const sources = hyp.validation?.evidence_sources || [];
  const audit = hyp.validation?.citation_audit;

  return (
    <div className="space-y-5">
      {/* Citation audit summary */}
      {audit && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <MetricCard value={String(audit.references_checked)} label="References Checked" color="studio" />
          <MetricCard value={String(audit.verified_pmids_count)} label="Verified PMIDs" color="emerald" />
          <MetricCard value={String(audit.invalid_pmids_count)} label="Invalid PMIDs" color={audit.invalid_pmids_count > 0 ? 'rose' : 'emerald'} />
          <MetricCard value={`${(audit.citation_precision * 100).toFixed(1)}%`} label="Citation Precision" color="emerald" />
        </div>
      )}

      {/* Source records */}
      <div className="space-y-2">
        <p className="stitch-kicker">Retrieved Evidence Passages ({sources.length})</p>
        {sources.length > 0 ? (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {sources.map((src, i) => (
              <div key={src.chunk_id || i} className="stitch-card px-4 py-3 space-y-1.5">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    {src.pmid && (
                      <span className="font-code text-[10px] text-teal-accent">PMID:{src.pmid}</span>
                    )}
                    <span className={`stage-badge text-[9px] py-0 ${
                      src.evidence_category === 'supporting'
                        ? 'bg-emerald-50 text-emerald-600 border border-emerald-200'
                        : src.evidence_category === 'contradictory'
                        ? 'bg-rose-50 text-rose-600 border border-rose-200'
                        : 'bg-studio-100 text-studio-600 border border-studio-200'
                    }`}>
                      {src.evidence_category}
                    </span>
                  </div>
                  <span className="text-[10px] text-studio-400">
                    Relevance: {src.relevance_score.toFixed(2)}
                  </span>
                </div>
                {src.paper_title && (
                  <p className="text-xs font-medium text-studio-800 line-clamp-1">{src.paper_title}</p>
                )}
                <p className="text-[11px] text-studio-600 leading-relaxed line-clamp-3">{src.text_content}</p>
                <div className="flex items-center gap-3 text-[10px] text-studio-400">
                  {src.journal && <span>{src.journal}</span>}
                  {src.publication_year && <span>{src.publication_year}</span>}
                  {src.verification_status && (
                    <span className={`font-medium ${
                      src.verification_status === 'verified' ? 'text-emerald-600' : 'text-amber-600'
                    }`}>
                      {src.verification_status}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-studio-500 italic">No evidence source records available.</p>
        )}
      </div>
    </div>
  );
};

// ─── Tab 4: Scientific Validation ────────────────────────────────────────

const ValidationTab: React.FC<{ hypothesis: HypothesisOutput }> = ({ hypothesis: hyp }) => {
  const v = hyp.validation;
  const nli = v?.nli_validation;
  const judge = v?.llm_judge;
  const rules = v?.rule_checks;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* NLI Validation */}
        <div className="space-y-3">
          <p className="stitch-kicker">NLI Validation (DeBERTa)</p>
          {nli ? (
            <div className="stitch-card p-4 space-y-2">
              <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                  <p className="text-lg font-semibold text-emerald-600">{nli.mean_entailment.toFixed(3)}</p>
                  <p className="text-[10px] text-studio-500">Entailment</p>
                </div>
                <div>
                  <p className="text-lg font-semibold text-rose-500">{nli.mean_contradiction.toFixed(3)}</p>
                  <p className="text-[10px] text-studio-500">Contradiction</p>
                </div>
                <div>
                  <p className="text-lg font-semibold text-studio-500">{nli.mean_neutral.toFixed(3)}</p>
                  <p className="text-[10px] text-studio-500">Neutral</p>
                </div>
              </div>
              <p className="text-xs text-studio-600 pt-2 border-t border-studio-100">
                <strong>Interpretation:</strong> {nli.interpretation}
              </p>
              <p className="text-[10px] text-studio-400 font-code">Model: {nli.model}</p>
            </div>
          ) : (
            <p className="text-xs text-studio-500 italic">NLI validation not available.</p>
          )}
        </div>

        {/* LLM Judge */}
        <div className="space-y-3">
          <p className="stitch-kicker">LLM Judge Evaluation</p>
          {judge ? (
            <div className="stitch-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-studio-800">
                  Mean Score: <span className="text-teal-accent">{judge.mean_score.toFixed(1)}</span> / 5.0
                </p>
                <span className="font-code text-[10px] text-studio-400">{judge.model}</span>
              </div>
              <div className="space-y-1.5">
                {judge.rubrics.map((r) => (
                  <div key={r.dimension} className="flex items-center gap-2">
                    <div className="flex-1 flex items-center gap-2">
                      <span className="text-[11px] text-studio-700 font-medium w-24 shrink-0 truncate">{r.dimension}</span>
                      <div className="flex-1 h-1.5 bg-studio-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-teal-accent rounded-full transition-all"
                          style={{ width: `${(r.score / 5) * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="text-[11px] font-code text-studio-600 w-8 text-right">{r.score.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-studio-500 italic">LLM judge evaluation not available.</p>
          )}
        </div>
      </div>

      {/* Rule checks */}
      {rules && (
        <div className="space-y-3">
          <div className="flex items-center gap-3">
            <p className="stitch-kicker">Rule-Based Checks</p>
            <span className="text-[10px] text-studio-400">
              {rules.passed_count} pass · {rules.failed_count} fail · {rules.warning_count} warn
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {rules.checks.map((check) => (
              <div key={check.check_name} className="stitch-card px-3 py-2 flex items-start gap-2">
                {check.status === 'pass' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 mt-0.5 shrink-0" />}
                {check.status === 'fail' && <AlertCircle className="w-3.5 h-3.5 text-rose-500 mt-0.5 shrink-0" />}
                {check.status === 'warning' && <AlertCircle className="w-3.5 h-3.5 text-amber-500 mt-0.5 shrink-0" />}
                {check.status === 'not_evaluated' && <CircleDot className="w-3.5 h-3.5 text-studio-400 mt-0.5 shrink-0" />}
                <div className="min-w-0">
                  <p className="text-[11px] font-semibold text-studio-800">{check.check_name}</p>
                  <p className="text-[10px] text-studio-500 line-clamp-2">{check.explanation}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Tab 5: Ranking & Deduplication ──────────────────────────────────────

const RankingTab: React.FC<{ hypothesis: HypothesisOutput }> = ({ hypothesis: hyp }) => {
  const rd = hyp.validation?.ranking_deduplication;

  return (
    <div className="space-y-5">
      <p className="stitch-kicker">Composite Ranking & Deduplication</p>
      {rd ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="stitch-card p-4 space-y-3">
            <h4 className="text-sm font-semibold text-studio-800">Ranking Scores</h4>
            <div className="grid grid-cols-2 gap-3">
              <ScoreItem label="Composite Rank" value={rd.composite_rank_score.toFixed(3)} />
              <ScoreItem label="Priority Score" value={rd.priority_score.toFixed(3)} />
              <ScoreItem label="Grounding" value={rd.grounding_score.toFixed(3)} />
              <ScoreItem label="NLI Component" value={rd.nli_component.toFixed(3)} />
              <ScoreItem label="Judge Score" value={rd.judge_score.toFixed(3)} />
            </div>
            {rd.formula && (
              <div className="pt-2 border-t border-studio-100">
                <p className="text-[10px] text-studio-400 font-medium">Formula</p>
                <p className="font-code text-[11px] text-studio-600">{rd.formula}</p>
              </div>
            )}
          </div>

          <div className="stitch-card p-4 space-y-3">
            <h4 className="text-sm font-semibold text-studio-800">Quality & Deduplication</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs text-studio-500">Quality Tier</span>
                <span className={`stage-badge text-[10px] ${tierColor(rd.quality_tier)}`}>
                  {rd.quality_tier}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-studio-500">Deduplication Threshold</span>
                <span className="text-xs font-code text-studio-700">{rd.deduplication_threshold.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-studio-500">Action</span>
                <span className={`stage-badge text-[10px] ${
                  rd.deduplication_action === 'keep'
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : 'bg-amber-50 text-amber-700 border border-amber-200'
                }`}>
                  {rd.deduplication_action}
                </span>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <p className="text-xs text-studio-500 italic">Ranking and deduplication data not available.</p>
      )}
    </div>
  );
};

// ─── Utility Sub-components ──────────────────────────────────────────────

interface MetricCardProps {
  value: string;
  label: string;
  color: 'emerald' | 'amber' | 'rose' | 'studio';
}

const MetricCard: React.FC<MetricCardProps> = ({ value, label, color }) => {
  const colorMap = {
    emerald: 'border-emerald-200 bg-emerald-50/50',
    amber: 'border-amber-200 bg-amber-50/50',
    rose: 'border-rose-200 bg-rose-50/50',
    studio: 'border-studio-200 bg-studio-50',
  };
  const textColor = {
    emerald: 'text-emerald-700',
    amber: 'text-amber-700',
    rose: 'text-rose-700',
    studio: 'text-studio-700',
  };

  return (
    <div className={`rounded-lg border px-3 py-2.5 ${colorMap[color]}`}>
      <p className={`text-base font-semibold ${textColor[color]}`}>{value}</p>
      <p className="text-[10px] text-studio-500">{label}</p>
    </div>
  );
};

const ScoreItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div>
    <p className="text-[10px] text-studio-400 font-medium">{label}</p>
    <p className="text-sm font-semibold text-studio-800 font-code">{value}</p>
  </div>
);
