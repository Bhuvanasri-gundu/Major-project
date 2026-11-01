// frontend/src/components/HypothesisCard.tsx — Hypothesis card for Screen 1 (Stitch design with metric pills)

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Copy,
  Check,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  Clock,
  GitBranch,
} from 'lucide-react';
import type { HypothesisOutput } from '../types';

interface HypothesisCardProps {
  hypothesis: HypothesisOutput;
  runId: string;
}

export const HypothesisCard: React.FC<HypothesisCardProps> = ({ hypothesis, runId }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(hypothesis.hypothesis);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isPassed = hypothesis.validation_status_badge === 'Checks passed';
  const isReview = hypothesis.validation_status_badge === 'Review required';

  const tierColor = (tier?: string) => {
    switch (tier) {
      case 'high':
        return 'bg-teal-light text-teal-dark';
      case 'medium':
        return 'bg-studio-100 text-studio-700';
      default:
        return 'bg-studio-100 text-studio-600';
    }
  };

  const statusColor = isPassed
    ? 'bg-teal-light text-teal-dark'
    : isReview
    ? 'bg-amber-50 text-amber-700'
    : 'bg-rose-50 text-rose-700';

  const statusIcon = isPassed ? (
    <ShieldCheck className="w-3 h-3" />
  ) : isReview ? (
    <Clock className="w-3 h-3" />
  ) : (
    <AlertCircle className="w-3 h-3" />
  );

  // Extract scores for metric pills
  const grounding = hypothesis.validation?.ranking_deduplication?.grounding_score;
  const nli = hypothesis.validation?.ranking_deduplication?.nli_component;
  const judge = hypothesis.validation?.ranking_deduplication?.judge_score;
  const composite = hypothesis.composite_rank_score;

  return (
    <div className="stitch-card p-4 space-y-3 hover:shadow-sm transition-shadow">
      {/* Header row: ID + Gap ref + Tier badge + Score */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-code text-[11px] text-studio-500 bg-studio-50 px-1.5 py-0.5 rounded">
            {hypothesis.hypothesis_id}
          </span>
          <span className="text-[11px] text-studio-400">·</span>
          <span className="text-[11px] text-studio-500">
            Linked gap: <span className="font-medium text-studio-700">{hypothesis.gap_id}</span>
          </span>
          {hypothesis.quality_tier && (
            <span className={`stage-badge text-[10px] capitalize ${tierColor(hypothesis.quality_tier)}`}>
              {hypothesis.quality_tier} tier
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {composite !== undefined && (
            <span className="font-code text-xs text-studio-600 bg-studio-50 px-2 py-0.5 rounded border border-studio-200">
              {composite.toFixed(3)}
            </span>
          )}
          <span className={`stage-badge text-[10px] ${statusColor}`}>
            {statusIcon}
            {hypothesis.validation_status_badge}
          </span>
        </div>
      </div>

      {/* Main hypothesis statement */}
      <p className="text-sm font-medium text-studio-900 leading-relaxed">
        "{hypothesis.hypothesis}"
      </p>

      {/* Mechanism & population row */}
      {hypothesis.proposed_mechanism && (
        <div className="flex items-start gap-2 text-xs text-studio-600 bg-studio-50 rounded px-3 py-2 border border-studio-100">
          <GitBranch className="w-3.5 h-3.5 text-studio-400 mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold text-studio-700">Mechanism: </span>
            <span>{hypothesis.proposed_mechanism}</span>
            {hypothesis.directionality && (
              <span className="ml-1.5 font-code text-[10px] text-studio-500 bg-studio-100 px-1.5 py-0.5 rounded capitalize">
                {hypothesis.directionality}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Bottom row: Metric pills + actions */}
      <div className="flex items-center justify-between pt-2 border-t border-studio-100">
        <div className="flex items-center gap-3 flex-wrap">
          {grounding !== undefined && (
            <span className="metric-pill text-teal-dark">
              <span className="w-1.5 h-1.5 rounded-full bg-teal-accent inline-block mr-1" />
              Grounding {(grounding * 100).toFixed(0)}%
            </span>
          )}
          {nli !== undefined && (
            <span className="metric-pill text-studio-600">
              <span className="w-1.5 h-1.5 rounded-full bg-studio-400 inline-block mr-1" />
              NLI {nli.toFixed(2)}
            </span>
          )}
          {judge !== undefined && (
            <span className="metric-pill text-studio-600">
              <span className="w-1.5 h-1.5 rounded-full bg-studio-500 inline-block mr-1" />
              Judge {judge.toFixed(2)}
            </span>
          )}
          {grounding === undefined && nli === undefined && judge === undefined && (
            <span className="text-[11px] text-studio-400">
              {hypothesis.supporting_evidence_count} supporting · {hypothesis.contradictory_evidence_count} contradictory passages
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium text-studio-600 hover:text-studio-900 bg-white hover:bg-studio-50 border border-studio-200 rounded transition"
            title="Copy hypothesis statement"
          >
            {copied ? (
              <>
                <Check className="w-3 h-3 text-teal-accent" />
                <span className="text-teal-dark">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3 h-3" />
                <span>Copy</span>
              </>
            )}
          </button>

          <Link
            to={`/workspace/${runId}?hypothesis=${hypothesis.hypothesis_id}`}
            className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-semibold text-white bg-studio-900 hover:bg-studio-800 rounded transition"
          >
            <span>Validate</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </div>
    </div>
  );
};
