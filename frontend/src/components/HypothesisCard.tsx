// frontend/src/components/HypothesisCard.tsx — Card for Screen 1 (strictly NO IV/DV fields)

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Copy,
  Check,
  ArrowRight,
  ShieldCheck,
  AlertCircle,
  Clock,
  Layers,
  Users,
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

  // Badge colors
  const isPassed = hypothesis.validation_status_badge === 'Checks passed';
  const isReview = hypothesis.validation_status_badge === 'Review required';

  return (
    <div className="bg-white rounded-xl border border-slate-200/90 hover:border-slate-300 shadow-xs hover:shadow-sm transition p-5 space-y-4">
      {/* Top Header: ID, Tier, Status Badge */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center space-x-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-slate-100 text-slate-700 font-mono">
            #{hypothesis.hypothesis_number} · {hypothesis.hypothesis_id}
          </span>
          {hypothesis.quality_tier && (
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold capitalize ${
                hypothesis.quality_tier === 'high'
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : hypothesis.quality_tier === 'medium'
                  ? 'bg-blue-50 text-blue-700 border border-blue-200'
                  : 'bg-slate-100 text-slate-600'
              }`}
            >
              {hypothesis.quality_tier} Tier
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          <span
            className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${
              isPassed
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : isReview
                ? 'bg-amber-50 text-amber-700 border border-amber-200'
                : 'bg-rose-50 text-rose-700 border border-rose-200'
            }`}
          >
            {isPassed ? (
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            ) : isReview ? (
              <Clock className="w-3.5 h-3.5 text-amber-600" />
            ) : (
              <AlertCircle className="w-3.5 h-3.5 text-rose-600" />
            )}
            <span>{hypothesis.validation_status_badge}</span>
          </span>
        </div>
      </div>

      {/* Main Hypothesis Statement */}
      <div>
        <p className="text-base font-semibold text-slate-900 leading-snug">
          "{hypothesis.hypothesis}"
        </p>
        <p className="text-[11px] text-slate-400 mt-1">
          AI-generated research hypothesis grounded in retrieved PubMed scientific evidence.
        </p>
      </div>

      {/* Structured Details (Target Population, Mechanism, Directionality, Gap Ref) */}
      {/* NOTE: Independent Variables (IV) and Dependent Variables (DV) are strictly NOT displayed here per prompt requirements */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs bg-slate-50/70 p-3 rounded-lg border border-slate-100">
        <div className="flex items-start space-x-2">
          <Users className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold text-slate-600">Target Population: </span>
            <span className="text-slate-800 font-medium">{hypothesis.target_population}</span>
          </div>
        </div>

        <div className="flex items-start space-x-2">
          <Layers className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold text-slate-600">Bridged Gap: </span>
            <span className="text-slate-800 font-medium">{hypothesis.gap_reference}</span>
          </div>
        </div>

        {hypothesis.proposed_mechanism && (
          <div className="flex items-start space-x-2 md:col-span-2">
            <GitBranch className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-slate-600">Proposed Mechanism: </span>
              <span className="text-slate-800">{hypothesis.proposed_mechanism}</span>
              {hypothesis.directionality && (
                <span className="ml-2 inline-flex items-center px-1.5 py-0.2 rounded text-[10px] font-medium bg-slate-200/80 text-slate-700 capitalize">
                  {hypothesis.directionality} direction
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Brief Evidence & Validation Summary */}
      <div className="flex items-center justify-between text-xs text-slate-500 pt-1 border-t border-slate-100">
        <div>
          <span>{hypothesis.evidence_summary || `${hypothesis.supporting_evidence_count} evidence passages`}</span>
          {hypothesis.composite_rank_score !== undefined && (
            <span className="ml-2 font-mono text-[11px] text-slate-600 bg-slate-100 px-1.5 py-0.5 rounded">
              Score: {hypothesis.composite_rank_score.toFixed(3)}
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center space-x-1 px-2.5 py-1 text-xs font-medium text-slate-600 hover:text-slate-900 bg-white hover:bg-slate-50 border border-slate-200 rounded-md transition shadow-2xs"
            title="Copy hypothesis statement"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>

          <Link
            to={`/workspace/${runId}?hypothesis=${hypothesis.hypothesis_id}`}
            className="inline-flex items-center space-x-1 px-2.5 py-1 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-md transition shadow-2xs"
          >
            <span>View Complete Validation</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </div>
    </div>
  );
};
