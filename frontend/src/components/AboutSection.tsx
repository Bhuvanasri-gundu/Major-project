// frontend/src/components/AboutSection.tsx — Collapsible "About This Application"

import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Info, BookOpen, Search, Cpu, ShieldCheck, AlertTriangle } from 'lucide-react';

export const AboutSection: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden transition">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-slate-50 transition"
      >
        <div className="flex items-center space-x-2.5">
          <Info className="w-4 h-4 text-indigo-600" />
          <span className="text-sm font-semibold text-slate-900">About This Application</span>
          <span className="text-xs text-slate-400 font-normal">Methodological overview, evidence pipeline & limitations</span>
        </div>
        <div className="text-slate-400">
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isOpen && (
        <div className="px-5 pb-5 pt-2 text-xs text-slate-600 border-t border-slate-100 space-y-4 animate-in fade-in duration-150">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* System Purpose */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-50/70 border border-slate-100">
              <div className="flex items-center space-x-1.5 font-semibold text-slate-800">
                <BookOpen className="w-3.5 h-3.5 text-indigo-500" />
                <span>System Purpose</span>
              </div>
              <p className="leading-relaxed">
                An AI-assisted scientific research assistant that analyzes the peer-reviewed mHealth literature to discover underexplored conceptual intersections and formulate testable empirical hypotheses grounded in scientific evidence.
              </p>
            </div>

            {/* Gap Matching */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-50/70 border border-slate-100">
              <div className="flex items-center space-x-1.5 font-semibold text-slate-800">
                <Search className="w-3.5 h-3.5 text-sky-500" />
                <span>Ontology & Gap Matching</span>
              </div>
              <p className="leading-relaxed">
                User research themes are mapped against 50 candidate research gaps derived from 737 quality-filtered MeSH concepts and 33 BERTopic topics. Gaps represent high semantic proximity but historically low co-occurrence in literature.
              </p>
            </div>

            {/* Evidence Retrieval */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-50/70 border border-slate-100">
              <div className="flex items-center space-x-1.5 font-semibold text-slate-800">
                <Cpu className="w-3.5 h-3.5 text-emerald-500" />
                <span>Evidence Retrieval (FAISS RAG)</span>
              </div>
              <p className="leading-relaxed">
                Passages are retrieved from a persistent FAISS index of 12,776 passages across 12,642 screened PubMed abstracts (2015–2025) using MiniLM-L6-v2 embeddings with paper-diversity reranking to prevent source bias.
              </p>
            </div>

            {/* Scientific Validation */}
            <div className="space-y-1.5 p-3 rounded-lg bg-slate-50/70 border border-slate-100">
              <div className="flex items-center space-x-1.5 font-semibold text-slate-800">
                <ShieldCheck className="w-3.5 h-3.5 text-purple-500" />
                <span>Multi-Layered Validation</span>
              </div>
              <p className="leading-relaxed">
                Synthesized hypotheses undergo four independent checks: deterministic citation verification against corpus PMIDs, 9 rule and falsifiability checks, a 6-rubric Gemini evaluation, and cross-encoder Natural Language Inference.
              </p>
            </div>
          </div>

          {/* Limitations Warning */}
          <div className="p-3 bg-amber-50/80 rounded-lg border border-amber-200/80 text-amber-900 flex items-start space-x-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1 leading-relaxed">
              <span className="font-semibold text-amber-950">Important Scientific Limitations: </span>
              Generated hypotheses are computational suggestions intended to accelerate research exploration. The underlying corpus consists strictly of scientific abstracts (not full-text trial data). Hypotheses are not empirically proven and require formal human-subject experimental testing and institutional ethics approval prior to clinical application.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
