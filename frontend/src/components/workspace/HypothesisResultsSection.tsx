// frontend/src/components/workspace/HypothesisResultsSection.tsx — Section 3: Hypothesis Results

import React from 'react';
import {
  FileText,
  ShieldCheck,
  AlertCircle,
  Clock,
  Layers,
  Users,
  CheckCircle2,
} from 'lucide-react';
import type { HypothesisOutput } from '../../types';

interface HypothesisResultsSectionProps {
  hypotheses: HypothesisOutput[];
  selectedHypothesisId: string;
  onSelectHypothesis: (id: string) => void;
  runStatus: string;
}

export const HypothesisResultsSection: React.FC<HypothesisResultsSectionProps> = ({
  hypotheses,
  selectedHypothesisId,
  onSelectHypothesis,
  runStatus,
}) => {
  if (hypotheses.length === 0) {
    return (
      <section className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-3">
        <h3 className="text-base font-bold text-slate-900 flex items-center space-x-2">
          <FileText className="w-4 h-4 text-indigo-600" />
          <span>Synthesized Hypothesis Results</span>
        </h3>
        <div className="p-6 bg-slate-50 rounded-xl border border-slate-100 text-center text-xs text-slate-500">
          {runStatus === 'no_hypotheses_requested'
            ? 'No hypotheses were requested for this run (count = 0). Research gaps and literature passages are available for exploration above.'
            : 'No hypotheses are available for this run.'}
        </div>
      </section>
    );
  }

  return (
    <section className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-5">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <FileText className="w-4 h-4 text-indigo-600" />
            <span>Generated Hypothesis Results ({hypotheses.length})</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Select any hypothesis below to inspect its complete scientific validation dossier in Section 4.
          </p>
        </div>
        <span className="text-xs text-slate-400 font-mono">
          Click card to inspect validation details
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {hypotheses.map((hyp) => {
          const isSelected = hyp.hypothesis_id === selectedHypothesisId;
          const isPassed = hyp.validation_status_badge === 'Checks passed';
          const isReview = hyp.validation_status_badge === 'Review required';

          return (
            <button
              key={hyp.hypothesis_id}
              type="button"
              onClick={() => onSelectHypothesis(hyp.hypothesis_id)}
              className={`w-full text-left rounded-xl border p-4 transition space-y-2.5 relative ${
                isSelected
                  ? 'border-indigo-600 bg-indigo-50/25 ring-2 ring-indigo-200 shadow-xs'
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              {isSelected && (
                <div className="absolute top-4 right-4">
                  <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-600 text-white shadow-2xs">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>Active Selection</span>
                  </span>
                </div>
              )}

              {/* ID & Badges */}
              <div className="flex items-center space-x-2 pr-28 flex-wrap gap-1">
                <span className="font-mono text-xs font-bold px-2 py-0.5 bg-slate-100 text-slate-700 rounded">
                  #{hyp.hypothesis_number} · {hyp.hypothesis_id}
                </span>

                {hyp.quality_tier && (
                  <span
                    className={`text-[10px] font-semibold px-2 py-0.5 rounded capitalize ${
                      hyp.quality_tier === 'high'
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                        : hyp.quality_tier === 'medium'
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    {hyp.quality_tier} Tier
                  </span>
                )}

                <span
                  className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-medium ${
                    isPassed
                      ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                      : isReview
                      ? 'bg-amber-50 text-amber-700 border border-amber-200'
                      : 'bg-rose-50 text-rose-700 border border-rose-200'
                  }`}
                >
                  {isPassed ? (
                    <ShieldCheck className="w-3 h-3 text-emerald-600" />
                  ) : isReview ? (
                    <Clock className="w-3 h-3 text-amber-600" />
                  ) : (
                    <AlertCircle className="w-3 h-3 text-rose-600" />
                  )}
                  <span>{hyp.validation_status_badge}</span>
                </span>
              </div>

              {/* Statement */}
              <p className="text-sm font-semibold text-slate-900 leading-snug">
                "{hyp.hypothesis}"
              </p>

              {/* Target Population & Evidence counts */}
              <div className="flex items-center justify-between text-xs text-slate-500 pt-1 border-t border-slate-100 flex-wrap gap-2">
                <div className="flex items-center space-x-3">
                  <span className="flex items-center space-x-1">
                    <Users className="w-3 h-3 text-slate-400" />
                    <span>{hyp.target_population}</span>
                  </span>
                  <span>·</span>
                  <span className="flex items-center space-x-1">
                    <Layers className="w-3 h-3 text-slate-400" />
                    <span>{hyp.gap_reference}</span>
                  </span>
                </div>

                <div className="flex items-center space-x-2 font-mono text-[11px]">
                  <span>Evidence: {hyp.supporting_evidence_count} sup / {hyp.contradictory_evidence_count} con</span>
                  {hyp.composite_rank_score !== undefined && (
                    <span className="bg-slate-100 px-1.5 py-0.5 rounded font-bold text-slate-700">
                      Score: {hyp.composite_rank_score.toFixed(3)}
                    </span>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
};
