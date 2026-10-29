// frontend/src/pages/GeneratePage.tsx — Screen 1: Generate Hypotheses

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Sparkles,
  ArrowRight,
  Loader2,
  AlertCircle,
  Lightbulb,
  Layers,
  Database,
  FileQuestion,
} from 'lucide-react';
import { api } from '../services/api';
import { ThemeSuggestions } from '../components/ThemeSuggestions';
import { HypothesisCard } from '../components/HypothesisCard';
import { AboutSection } from '../components/AboutSection';
import type { RunResponse } from '../types';

export const GeneratePage: React.FC = () => {
  const navigate = useNavigate();

  // Form State
  const [theme, setTheme] = useState('');
  const [population, setPopulation] = useState('');
  const [objective, setObjective] = useState('');
  const [hypothesisCount, setHypothesisCount] = useState<number>(3);

  // Suggestions state
  const [showSuggestions, setShowSuggestions] = useState(false);

  // Execution & Output State
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentRun, setCurrentRun] = useState<RunResponse | null>(null);

  // Validation errors
  const [themeError, setThemeError] = useState<string | null>(null);

  // Load latest run on initial mount if available so output is displayed
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const latest = await api.getLatestRun();
        if (latest) {
          setCurrentRun(latest);
          if (!theme && latest.theme) {
            setTheme(latest.theme);
            setPopulation(latest.population === 'Unspecified' ? '' : latest.population);
            setObjective(latest.objective === 'Bridge literature gap' ? '' : latest.objective);
          }
        }
      } catch {
        // no prior run, start empty
      }
    };
    fetchInitial();
  }, []);

  const handleThemeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setTheme(val);
    if (themeError && val.trim().length >= 3) {
      setThemeError(null);
    }
    setShowSuggestions(val.trim().length >= 2);
  };

  const handleSelectSuggestion = (selectedText: string) => {
    setTheme(selectedText);
    setShowSuggestions(false);
    setThemeError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setShowSuggestions(false);

    // Validate theme
    const trimmedTheme = theme.trim();
    if (!trimmedTheme || trimmedTheme.length < 3) {
      setThemeError('Please enter a research theme or problem statement (minimum 3 characters).');
      return;
    }
    if (trimmedTheme.length > 1000) {
      setThemeError('Research theme cannot exceed 1,000 characters.');
      return;
    }
    setThemeError(null);
    setErrorMessage(null);
    setIsGenerating(true);

    try {
      const response = await api.generateHypotheses({
        theme: trimmedTheme,
        population: population.trim(),
        objective: objective.trim(),
        hypothesis_count: hypothesisCount,
      });
      setCurrentRun(response);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to complete hypothesis generation pipeline.');
    } finally {
      setIsGenerating(false);
    }
  };

  const activeRunId = currentRun?.run_id || 'latest';

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-10">
      {/* Introduction Card */}
      <div className="bg-gradient-to-r from-indigo-900 via-indigo-800 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-md relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-radial from-indigo-500/10 to-transparent pointer-events-none" />
        <div className="relative z-10 space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-xs text-xs font-medium text-indigo-200 border border-white/15">
            <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
            <span>Temporal-Semantic Literature Gap Analysis & Synthesis</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white leading-tight">
            Synthesize Evidence-Grounded mHealth Hypotheses
          </h2>
          <p className="text-sm text-indigo-100 max-w-2xl leading-relaxed">
            Enter your mobile health research theme. The system matches your query against 50 precomputed literature gaps, retrieves supporting PubMed abstract evidence from FAISS, and generates testable empirical hypotheses validated across 4 scientific dimensions.
          </p>
        </div>
      </div>

      {/* SECTION A: Research Input Form */}
      <section className="bg-white rounded-2xl border border-slate-200 shadow-xs p-6 sm:p-8 space-y-6">
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
              <Lightbulb className="w-5 h-5 text-indigo-600" />
              <span>Research Configuration</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Specify your domain focus, patient population, and requested output density.
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* 1. Research Theme — REQUIRED */}
          <div className="space-y-1.5 relative">
            <div className="flex items-center justify-between">
              <label htmlFor="theme" className="block text-xs font-bold text-slate-800 uppercase tracking-wide">
                Research Theme or Problem Statement <span className="text-rose-500">*</span>
              </label>
              <span className={`text-[11px] font-mono ${theme.length > 950 ? 'text-amber-600' : 'text-slate-400'}`}>
                {theme.length} / 1,000
              </span>
            </div>

            <div className="relative">
              <textarea
                id="theme"
                rows={3}
                value={theme}
                onChange={handleThemeChange}
                onFocus={() => theme.trim().length >= 2 && setShowSuggestions(true)}
                placeholder="Describe your mHealth research theme or the research problem you want to investigate (e.g. Asynchronous tele-triage to improve primary care accessibility in rural healthcare)..."
                className={`w-full rounded-xl border p-3.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 transition shadow-2xs resize-y ${
                  themeError
                    ? 'border-rose-300 focus:border-rose-500 focus:ring-rose-200'
                    : 'border-slate-200 focus:border-indigo-500 focus:ring-indigo-100'
                }`}
                disabled={isGenerating}
              />

              {/* Live Theme Suggestions Dropdown */}
              <ThemeSuggestions
                query={theme}
                isOpen={showSuggestions}
                onSelectSuggestion={handleSelectSuggestion}
                onClose={() => setShowSuggestions(false)}
              />
            </div>

            {themeError && (
              <p className="text-xs text-rose-600 font-medium flex items-center space-x-1 mt-1">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                <span>{themeError}</span>
              </p>
            )}
            <p className="text-[11px] text-slate-400">
              Tip: Start typing concepts like "rural health", "medication adherence", or "telemedicine" to view suggestions from candidate gaps and MeSH terms.
            </p>
          </div>

          {/* 2. Target Population & Research Objective */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Target Population — OPTIONAL */}
            <div className="space-y-1.5">
              <label htmlFor="population" className="block text-xs font-bold text-slate-800 uppercase tracking-wide">
                Target Population <span className="text-slate-400 font-normal lowercase">(optional)</span>
              </label>
              <input
                id="population"
                type="text"
                value={population}
                onChange={(e) => setPopulation(e.target.value)}
                placeholder="e.g. Adults with type 2 diabetes, rural older adults"
                className="w-full rounded-xl border border-slate-200 p-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition shadow-2xs"
                disabled={isGenerating}
                maxLength={200}
              />
              <p className="text-[11px] text-slate-400">
                Included in the generated hypothesis when provided; otherwise derived from literature evidence.
              </p>
            </div>

            {/* Research Objective — OPTIONAL */}
            <div className="space-y-1.5">
              <label htmlFor="objective" className="block text-xs font-bold text-slate-800 uppercase tracking-wide">
                Research Objective <span className="text-slate-400 font-normal lowercase">(optional)</span>
              </label>
              <input
                id="objective"
                type="text"
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                placeholder="e.g. Investigate factors associated with improved primary care utilization"
                className="w-full rounded-xl border border-slate-200 p-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 transition shadow-2xs"
                disabled={isGenerating}
                maxLength={300}
              />
              <p className="text-[11px] text-slate-400">
                Constrains the generation to specific clinical, behavioral, or health service goals.
              </p>
            </div>
          </div>

          {/* 3. Number of Hypotheses (0–3) */}
          <div className="space-y-2 pt-2 border-t border-slate-100">
            <label className="block text-xs font-bold text-slate-800 uppercase tracking-wide">
              Number of Hypotheses to Generate
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { val: 0, label: '0 Hypotheses', desc: 'Gap exploration & evidence retrieval only' },
                { val: 1, label: '1 Hypothesis', desc: 'Single focal hypothesis' },
                { val: 2, label: '2 Hypotheses', desc: 'Comparative mechanisms' },
                { val: 3, label: '3 Hypotheses', desc: 'Standard generation (Recommended)' },
              ].map((opt) => (
                <button
                  key={opt.val}
                  type="button"
                  onClick={() => setHypothesisCount(opt.val)}
                  className={`p-3 rounded-xl border text-left transition flex flex-col justify-between ${
                    hypothesisCount === opt.val
                      ? 'border-indigo-600 bg-indigo-50/70 text-indigo-950 ring-2 ring-indigo-200 shadow-xs'
                      : 'border-slate-200 hover:border-slate-300 bg-white text-slate-700'
                  }`}
                  disabled={isGenerating}
                >
                  <span className="font-bold text-xs">{opt.label}</span>
                  <span className="text-[11px] text-slate-500 mt-1">{opt.desc}</span>
                </button>
              ))}
            </div>
            {hypothesisCount === 0 && (
              <p className="text-xs text-amber-700 bg-amber-50 p-2.5 rounded-lg border border-amber-200">
                <strong>Zero hypotheses selected:</strong> The pipeline will match your theme to candidate gaps and retrieve literature evidence without making Gemini generation calls.
              </p>
            )}
          </div>

          {/* Submission Error Banner */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start space-x-2.5 animate-in fade-in">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-900">Pipeline Execution Error</p>
                <p className="mt-0.5">{errorMessage}</p>
              </div>
            </div>
          )}

          {/* Action Button */}
          <div className="flex items-center justify-between pt-2">
            <span className="text-xs text-slate-400">
              * Publication date filter removed per academic specification.
            </span>
            <button
              type="submit"
              disabled={isGenerating}
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-xl font-bold text-sm text-white bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Synthesizing Hypotheses (FAISS + Gemini)...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate Hypotheses</span>
                </>
              )}
            </button>
          </div>
        </form>
      </section>

      {/* SECTION B: Generated Hypotheses Output (SAME SCREEN) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div>
            <h3 className="text-xl font-bold text-slate-900 flex items-center space-x-2">
              <Layers className="w-5 h-5 text-indigo-600" />
              <span>Generated Research Hypotheses</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              {currentRun
                ? `Run ID: ${currentRun.run_id} · ${currentRun.hypotheses.length} hypotheses synthesized from literature evidence`
                : 'Generated hypotheses will appear below immediately upon synthesis.'}
            </p>
          </div>

          {currentRun && (
            <button
              type="button"
              onClick={() => navigate(`/workspace/${activeRunId}`)}
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 transition shadow-xs"
            >
              <span>View Complete Validation Workspace</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Empty / Zero Case State */}
        {currentRun && currentRun.hypotheses.length === 0 && (
          <div className="bg-white rounded-2xl border border-slate-200 p-8 text-center space-y-3">
            <FileQuestion className="w-10 h-10 text-slate-400 mx-auto" />
            <h4 className="text-base font-bold text-slate-800">
              {currentRun.status === 'no_hypotheses_requested'
                ? 'No Hypotheses Requested (Count = 0)'
                : 'No Hypotheses Generated'}
            </h4>
            <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
              {currentRun.status === 'no_hypotheses_requested'
                ? 'The pipeline successfully matched literature gaps and retrieved evidence passages without generating hypothesis statements.'
                : 'The research pipeline finished without producing valid hypotheses. You can inspect matched gaps or adjust your theme.'}
            </p>
            <div className="pt-2">
              <button
                type="button"
                onClick={() => navigate(`/workspace/${activeRunId}`)}
                className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-lg text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 transition"
              >
                <span>Inspect Matched Gaps in Workspace</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Hypothesis Cards */}
        {currentRun && currentRun.hypotheses.length > 0 && (
          <div className="space-y-4">
            {currentRun.hypotheses.map((hyp) => (
              <HypothesisCard
                key={hyp.hypothesis_id}
                hypothesis={hyp}
                runId={activeRunId}
              />
            ))}

            {/* Bottom Call to Action for Complete Workspace */}
            <div className="pt-4 flex items-center justify-between bg-indigo-50/70 border border-indigo-100 rounded-xl p-4">
              <div className="space-y-0.5">
                <h4 className="text-xs font-bold text-indigo-950 uppercase tracking-wide">
                  Deep Scientific Audit Available
                </h4>
                <p className="text-xs text-indigo-800">
                  Inspect passage-level DeBERTa NLI scores, 6-rubric Gemini evaluations, citation grounding, and rule checks.
                </p>
              </div>
              <button
                type="button"
                onClick={() => navigate(`/workspace/${activeRunId}`)}
                className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-700 transition shadow-xs shrink-0"
              >
                <span>View Complete Validation Workspace</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {!currentRun && !isGenerating && (
          <div className="bg-slate-50/70 border-2 border-dashed border-slate-200 rounded-2xl p-10 text-center space-y-2">
            <Database className="w-8 h-8 text-slate-300 mx-auto" />
            <p className="text-sm font-semibold text-slate-600">No Active Generation Run</p>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Enter a research theme above and click "Generate Hypotheses" to run the live retrieval and validation pipeline.
            </p>
          </div>
        )}
      </section>

      {/* SECTION C: Collapsible About This Application */}
      <AboutSection />
    </div>
  );
};
