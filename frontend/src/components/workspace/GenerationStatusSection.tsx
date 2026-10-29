// frontend/src/components/workspace/GenerationStatusSection.tsx — Section 1: Generation Status

import React from 'react';
import { Activity, Clock, CheckCircle2, AlertTriangle, XCircle, ShieldCheck } from 'lucide-react';
import type { RunResponse } from '../../types';

interface GenerationStatusSectionProps {
  run: RunResponse;
}

export const GenerationStatusSection: React.FC<GenerationStatusSectionProps> = ({ run }) => {
  const isCompleted = run.status === 'completed' || run.status === 'no_hypotheses_requested';
  const isFailed = run.status === 'failed';

  return (
    <section className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3 border-b border-slate-100 pb-4">
        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-700 font-bold">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900 leading-tight">Generation Run Status</h3>
            <p className="text-xs text-slate-500 font-mono">Run ID: {run.run_id}</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <span
            className={`inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-semibold capitalize ${
              isCompleted
                ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                : isFailed
                ? 'bg-rose-50 text-rose-700 border border-rose-200'
                : 'bg-amber-50 text-amber-700 border border-amber-200'
            }`}
          >
            {isCompleted ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            ) : isFailed ? (
              <XCircle className="w-3.5 h-3.5 text-rose-600" />
            ) : (
              <Clock className="w-3.5 h-3.5 text-amber-600 animate-spin" />
            )}
            <span>{run.status.replace(/_/g, ' ')}</span>
          </span>
        </div>
      </div>

      {/* Run Parameters Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
        <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 space-y-1">
          <span className="font-semibold text-slate-500 uppercase text-[10px] tracking-wide">Research Theme</span>
          <p className="font-medium text-slate-900 line-clamp-2">{run.theme}</p>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 space-y-1">
          <span className="font-semibold text-slate-500 uppercase text-[10px] tracking-wide">Target Population</span>
          <p className="font-medium text-slate-900">{run.population || 'Unspecified by user'}</p>
        </div>

        <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 space-y-1">
          <span className="font-semibold text-slate-500 uppercase text-[10px] tracking-wide">Hypothesis Density</span>
          <p className="font-medium text-slate-900">
            {run.hypothesis_count_generated} synthesized / {run.hypothesis_count_requested} requested
          </p>
        </div>
      </div>

      {/* Execution Stages Tracked */}
      <div className="space-y-2 pt-1">
        <div className="flex items-center justify-between text-xs font-semibold text-slate-700">
          <span className="flex items-center space-x-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-indigo-600" />
            <span>Verified Pipeline Execution Stages</span>
          </span>
          <span className="text-[11px] text-slate-400 font-mono">
            Started: {run.started_at ? new Date(run.started_at).toLocaleTimeString() : 'N/A'} · Completed:{' '}
            {run.completed_at ? new Date(run.completed_at).toLocaleTimeString() : 'In Progress'}
          </span>
        </div>

        <div className="flex items-center flex-wrap gap-1.5">
          {[
            { id: 'gap_matching', label: '1. Gap Matching' },
            { id: 'evidence_retrieval', label: '2. FAISS Retrieval' },
            { id: 'hypothesis_generation', label: '3. Gemini Synthesis' },
            { id: 'evidence_verification', label: '4. Citation Audit' },
            { id: 'rule_checks', label: '5. Rule Checks' },
            { id: 'llm_judge', label: '6. LLM Rubric Judge' },
            { id: 'ranking', label: '7. Composite Ranking' },
            { id: 'deduplication', label: '8. Semantic Dedup' },
          ].map((stage) => {
            const isDone = run.stages_completed.includes(stage.id);
            return (
              <span
                key={stage.id}
                className={`inline-flex items-center space-x-1 px-2.5 py-1 rounded-md text-xs font-medium ${
                  isDone
                    ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                    : 'bg-slate-100 text-slate-400 border border-slate-200/60'
                }`}
              >
                {isDone ? (
                  <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                ) : (
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                )}
                <span>{stage.label}</span>
              </span>
            );
          })}
        </div>
      </div>

      {/* Warnings & Failures */}
      {run.warnings.length > 0 && (
        <div className="p-3 bg-amber-50 rounded-xl border border-amber-200 text-xs text-amber-800 space-y-1">
          <div className="flex items-center space-x-1.5 font-bold text-amber-900">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
            <span>Pipeline Execution Warnings:</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-amber-700">
            {run.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {run.failures.length > 0 && (
        <div className="p-3 bg-rose-50 rounded-xl border border-rose-200 text-xs text-rose-800 space-y-1">
          <div className="flex items-center space-x-1.5 font-bold text-rose-900">
            <XCircle className="w-3.5 h-3.5 text-rose-600" />
            <span>Execution Failures:</span>
          </div>
          <ul className="list-disc list-inside space-y-0.5 text-rose-700">
            {run.failures.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
};
