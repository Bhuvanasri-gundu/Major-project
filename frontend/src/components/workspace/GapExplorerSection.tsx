// frontend/src/components/workspace/GapExplorerSection.tsx — Section 2: Research Gap Explorer

import React, { useState } from 'react';
import { Layers, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import type { MatchedGap, HypothesisOutput } from '../../types';

interface GapExplorerSectionProps {
  matchedGaps: MatchedGap[];
  hypotheses?: HypothesisOutput[];
  selectedHypothesis?: HypothesisOutput;
}

export const GapExplorerSection: React.FC<GapExplorerSectionProps> = ({
  matchedGaps,
  selectedHypothesis,
}) => {
  const [expandedGapId, setExpandedGapId] = useState<string | null>(
    matchedGaps.length > 0 ? matchedGaps[0].gap_id : null
  );

  const activeGapId = selectedHypothesis?.gap_id || (matchedGaps.length > 0 ? matchedGaps[0].gap_id : '');

  return (
    <section className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-6">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900 flex items-center space-x-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <span>Research Gap Explorer & Conceptual Lineage</span>
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Precomputed candidate literature gaps identified from 12,642 PubMed abstracts via MeSH term co-occurrence and SPECTER2 semantic divergence.
          </p>
        </div>
        <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
          Precomputed Literature Gaps (Phase 1)
        </span>
      </div>

      {/* Lineage Flow Visualization */}
      <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 text-xs flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-slate-600 uppercase text-[10px] tracking-wide">Research Lineage:</span>
          <span className="bg-white px-2 py-0.5 rounded border border-slate-200 text-slate-800 font-medium">
            User Research Theme
          </span>
          <span className="text-slate-400 font-bold">→</span>
          <span className="bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200 text-indigo-800 font-bold">
            Matched Literature Gap ({activeGapId})
          </span>
          <span className="text-slate-400 font-bold">→</span>
          <span className="bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 text-emerald-800 font-bold">
            Grounding Passages (FAISS)
          </span>
          <span className="text-slate-400 font-bold">→</span>
          <span className="bg-purple-50 px-2 py-0.5 rounded border border-purple-200 text-purple-800 font-bold">
            Synthesized Hypothesis
          </span>
        </div>
      </div>

      {/* Matched Gaps Cards */}
      <div className="space-y-3">
        <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">
          Matched Candidate Gaps for Active Run ({matchedGaps.length})
        </h4>

        <div className="grid grid-cols-1 gap-3">
          {matchedGaps.map((gap) => {
            const isFocal = gap.gap_id === activeGapId;
            const isExpanded = expandedGapId === gap.gap_id;

            return (
              <div
                key={gap.gap_id}
                className={`rounded-xl border transition p-4 space-y-3 ${
                  isFocal
                    ? 'border-indigo-500 bg-indigo-50/20 shadow-xs ring-1 ring-indigo-200'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                {/* Header Row */}
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-0.5 rounded text-xs font-bold bg-slate-100 text-slate-700 font-mono">
                      {gap.gap_id}
                    </span>
                    <span className="text-sm font-bold text-slate-900">
                      {gap.concept_a} <span className="text-indigo-600 font-black">↔</span> {gap.concept_b}
                    </span>
                    {isFocal && (
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-600 text-white shadow-2xs">
                        Primary Gap Used
                      </span>
                    )}
                  </div>

                  <div className="flex items-center space-x-3 text-xs">
                    <span className="font-mono text-slate-600 bg-slate-100 px-2 py-0.5 rounded">
                      GapScore: {gap.gap_score.toFixed(3)}
                    </span>
                    <button
                      type="button"
                      onClick={() => setExpandedGapId(isExpanded ? null : gap.gap_id)}
                      className="text-slate-400 hover:text-slate-600 p-1"
                    >
                      {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                {/* Matching Explanation */}
                <p className="text-xs text-slate-600 leading-relaxed">
                  <strong className="text-slate-800">Theme Alignment: </strong>
                  {gap.matching_explanation}
                </p>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="pt-3 border-t border-slate-100 text-xs space-y-3 animate-in fade-in duration-150">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                      <div>
                        <span className="text-[10px] text-slate-400 uppercase font-semibold">Semantic Distance</span>
                        <p className="font-mono text-slate-800 font-bold">{gap.semantic_similarity.toFixed(4)}</p>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-400 uppercase font-semibold">Temporal Velocity</span>
                        <p className="font-mono text-slate-800 font-bold">
                          {gap.temporal_growth_avg ? `+${(gap.temporal_growth_avg * 100).toFixed(1)}%` : 'N/A'}
                        </p>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-400 uppercase font-semibold">Joint Literature</span>
                        <p className="font-mono text-slate-800 font-bold">{gap.supporting_pmids_count} papers</p>
                      </div>
                      <div>
                        <span className="text-[10px] text-slate-400 uppercase font-semibold">Matching Score</span>
                        <p className="font-mono text-indigo-700 font-bold">{gap.matching_score.toFixed(3)}</p>
                      </div>
                    </div>

                    {/* Gap Alignment Breakdown */}
                    {selectedHypothesis && selectedHypothesis.validation?.gap_alignment && (
                      <div className="p-3 bg-indigo-50/50 rounded-lg border border-indigo-100 space-y-1.5">
                        <span className="font-bold text-indigo-950 uppercase text-[10px] tracking-wide flex items-center space-x-1">
                          <Sparkles className="w-3 h-3 text-indigo-600" />
                          <span>Hypothesis Gap Alignment Assessment</span>
                        </span>
                        <p className="text-slate-700">
                          <strong>Relevance Rationale: </strong>
                          {selectedHypothesis.validation.gap_alignment.relevance_rationale}
                        </p>
                        <p className="text-slate-700">
                          <strong>Aspect Addressed: </strong>
                          {selectedHypothesis.validation.gap_alignment.aspect_addressed}
                        </p>
                        <div className="text-[11px] text-slate-500 pt-1 border-t border-indigo-100/60 flex items-center justify-between">
                          <span>
                            Methodology: {selectedHypothesis.validation.gap_alignment.limitations}
                          </span>
                          <span className="font-semibold text-indigo-700">
                            Status: {selectedHypothesis.validation.gap_alignment.alignment_status}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
