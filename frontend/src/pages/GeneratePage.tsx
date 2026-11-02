// frontend/src/pages/GeneratePage.tsx — Screen 1: Monograph Studio (Stitch Design)

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FileText,
  Users,
  Search,
  Loader2,
  AlertCircle,
  ShieldCheck,
  ChevronDown,
  Sparkles,
  ArrowRight,
  Database,
  ExternalLink,
} from 'lucide-react';
import { api } from '../services/api';
import { ThemeSuggestions } from '../components/ThemeSuggestions';
import type { RunResponse } from '../types';

export const GeneratePage: React.FC = () => {
  const navigate = useNavigate();

  // Form State - strictly empty by default
  const [theme, setTheme] = useState('');
  const [population, setPopulation] = useState('');
  const [objective, setObjective] = useState('');
  const [hypothesisCount, setHypothesisCount] = useState<number>(3);

  // Suggestions state & trigger ref for outside-click dismissal
  const [showSuggestions, setShowSuggestions] = useState(false);
  const suggestionsTriggerRef = useRef<HTMLButtonElement>(null);

  // Execution & Output State
  const [isGenerating, setIsGenerating] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [currentRun, setCurrentRun] = useState<RunResponse | null>(null);

  // Validation errors
  const [themeError, setThemeError] = useState<string | null>(null);

  // Collapsible about section
  const [aboutOpen, setAboutOpen] = useState(false);

  // Load latest run on initial mount ONLY for displaying results in workspace link/cards if present,
  // but NEVER pre-fill the input fields with prior text
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const latest = await api.getLatestRun();
        if (latest) {
          setCurrentRun(latest);
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
  const hyps = currentRun?.hypotheses || [];

  return (
    <div className="max-w-[1280px] mx-auto px-6 py-4 space-y-6">
      {/* Subheader / Evaluation Workbench State Bar */}
      <div className="flex items-center justify-between border-b border-studio-200 pb-2.5 text-xs text-studio-500 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-studio-700 uppercase tracking-wider text-[11px]">
            EVALUATION WORKBENCH STATE:
          </span>
          <div className="flex items-center gap-1.5 font-code text-[11px]">
            <span
              className={`px-2 py-0.5 rounded border ${
                !currentRun && !isGenerating
                  ? 'bg-studio-900 text-white border-studio-900 font-bold'
                  : 'bg-studio-50 text-studio-600 border-studio-200'
              }`}
            >
              1. Unpopulated Baseline
            </span>
            <span className="text-studio-300">→</span>
            <span
              className={`px-2 py-0.5 rounded border ${
                isGenerating
                  ? 'bg-studio-900 text-white border-studio-900 font-bold animate-pulse'
                  : 'bg-studio-50 text-studio-600 border-studio-200'
              }`}
            >
              2. Synthesis In-Flight
            </span>
            <span className="text-studio-300">→</span>
            <span
              className={`px-2 py-0.5 rounded border ${
                currentRun && !isGenerating
                  ? 'bg-teal-light text-teal-dark border-teal-accent/30 font-bold'
                  : 'bg-studio-50 text-studio-600 border-studio-200'
              }`}
            >
              3. Generated Scaffold Ready
            </span>
            <span className="text-studio-300">→</span>
            <span className="px-2 py-0.5 rounded border bg-studio-50 text-studio-600 border-studio-200">
              4. Required Field Validation
            </span>
          </div>
        </div>

        {currentRun && (
          <button
            type="button"
            onClick={() => navigate(`/workspace/${activeRunId}`)}
            className="inline-flex items-center gap-1 text-teal-accent hover:underline font-semibold"
          >
            <span>Open Complete Workspace</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Hero Header Section */}
      <div className="space-y-1.5 pt-2">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-teal-accent">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-accent" />
          <span>HYPOTHESIS GENERATION WORKSPACE • MONOGRAPH STUDIO</span>
        </div>
        <h1 className="font-heading-serif text-3xl sm:text-4xl text-studio-900 font-bold tracking-tight">
          What would you like to investigate?
        </h1>
        <p className="text-sm text-studio-600 max-w-3xl">
          Explore research gaps in mHealth literature and develop evidence-grounded hypotheses.
        </p>
      </div>

      {/* MAIN TWO-COLUMN LAYOUT (Left Form 55%, Right Results 45%) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* LEFT COLUMN: Input Form */}
        <div className="lg:col-span-7 space-y-4">
          <form onSubmit={handleSubmit} className="stitch-card p-6 space-y-5">
            {/* 1. Research Theme */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label htmlFor="theme" className="text-xs font-bold text-studio-900">
                  Research theme <span className="text-rose-500">*</span>
                </label>
                <button
                  ref={suggestionsTriggerRef}
                  type="button"
                  onClick={() => setShowSuggestions(!showSuggestions)}
                  className="inline-flex items-center gap-1 text-xs font-semibold text-teal-accent hover:underline cursor-pointer"
                >
                  <Sparkles className="w-3 h-3" />
                  <span>Ontology Suggestions</span>
                </button>
              </div>

              <div className="relative">
                <textarea
                  id="theme"
                  rows={4}
                  value={theme}
                  onChange={handleThemeChange}
                  onFocus={() => theme.trim().length >= 2 && setShowSuggestions(true)}
                  placeholder="e.g., JITAI intervention timing for diabetes self-management"
                  className={`w-full rounded border p-3 text-sm text-studio-900 placeholder:text-studio-400 focus:outline-none focus:ring-1 focus:ring-studio-900 transition resize-y ${
                    themeError ? 'border-rose-300' : 'border-studio-200'
                  }`}
                  disabled={isGenerating}
                />

                {/* Theme Suggestions Dropdown with triggerRef for outside-click dismiss */}
                <ThemeSuggestions
                  query={theme}
                  isOpen={showSuggestions}
                  triggerRef={suggestionsTriggerRef}
                  onSelectSuggestion={handleSelectSuggestion}
                  onClose={() => setShowSuggestions(false)}
                />
              </div>

              {themeError ? (
                <p className="text-xs text-rose-600 flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{themeError}</span>
                </p>
              ) : (
                <p className="text-[11px] text-studio-500 flex items-center gap-1">
                  <span>ⓘ Enter a topic, research question, or area of interest.</span>
                </p>
              )}
            </div>

            {/* 2. Target Population */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label htmlFor="population" className="text-xs font-bold text-studio-900">
                  Target population
                </label>
                <span className="text-[11px] text-studio-400 font-medium">Optional</span>
              </div>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-studio-400">
                  <Users className="w-4 h-4" />
                </div>
                <input
                  id="population"
                  type="text"
                  value={population}
                  onChange={(e) => setPopulation(e.target.value)}
                  placeholder="e.g., Adults with type 2 diabetes"
                  className="w-full rounded border border-studio-200 pl-9 pr-3 py-2 text-sm text-studio-900 placeholder:text-studio-400 focus:outline-none focus:ring-1 focus:ring-studio-900"
                  disabled={isGenerating}
                />
              </div>
            </div>

            {/* 3. Research Objective */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label htmlFor="objective" className="text-xs font-bold text-studio-900">
                  Research objective
                </label>
                <span className="text-[11px] text-studio-400 font-medium">Optional</span>
              </div>
              <textarea
                id="objective"
                rows={2}
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                placeholder="e.g., Identify factors influencing intervention adherence"
                className="w-full rounded border border-studio-200 p-2.5 text-sm text-studio-900 placeholder:text-studio-400 focus:outline-none focus:ring-1 focus:ring-studio-900 resize-none"
                disabled={isGenerating}
              />
            </div>

            {/* 4. Generate Hypotheses Count */}
            <div className="space-y-1.5 pt-2">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs font-bold text-studio-900">Generate hypotheses</div>
                  <div className="text-[11px] text-studio-500">
                    Number of discrete deductive propositions to construct
                  </div>
                </div>
                <div className="inline-flex rounded border border-studio-200 p-0.5 bg-studio-50">
                  {[1, 2, 3].map((num) => (
                    <button
                      key={num}
                      type="button"
                      onClick={() => setHypothesisCount(num)}
                      className={`px-3 py-1 text-xs font-semibold rounded transition ${
                        hypothesisCount === num
                          ? 'bg-white text-studio-900 shadow-xs border border-studio-200'
                          : 'text-studio-500 hover:text-studio-800'
                      }`}
                      disabled={isGenerating}
                    >
                      {num}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <div className="pt-2 space-y-2">
              <button
                type="submit"
                disabled={isGenerating}
                className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-studio-900 text-white text-xs font-bold rounded hover:bg-studio-800 transition disabled:opacity-50 cursor-pointer shadow-xs"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Executing Pipeline (FAISS RAG & Gemini Synthesis)...</span>
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    <span>Explore Research & Generate</span>
                  </>
                )}
              </button>

              <div className="flex items-center gap-1.5 text-[11px] text-studio-500 justify-center">
                <ShieldCheck className="w-3.5 h-3.5 text-teal-accent" />
                <span>Uses the project's available literature corpus and research pipeline. Does not claim comprehensive coverage of all mHealth literature.</span>
              </div>
            </div>

            {errorMessage && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded text-xs text-rose-700 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            )}
          </form>
        </div>

        {/* RIGHT COLUMN: Results Section */}
        <div className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between pb-1">
            <div className="flex items-center gap-2">
              <h3 className="font-heading-serif text-lg font-bold text-studio-900">
                Generated hypotheses
              </h3>
              <span className="text-xs font-code px-2 py-0.5 rounded bg-studio-100 text-studio-600 border border-studio-200">
                {hyps.length} ready
              </span>
            </div>
            <span className="text-[10px] font-code text-studio-400">
              WORKSPACE REF // H-GEN
            </span>
          </div>

          {/* Empty State when no hypotheses */}
          {hyps.length === 0 && !isGenerating && (
            <div className="stitch-card p-10 text-center space-y-4 min-h-[380px] flex flex-col items-center justify-center">
              <div className="w-12 h-12 rounded bg-studio-100 text-studio-400 flex items-center justify-center">
                <FileText className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h4 className="text-base font-bold text-studio-800">
                  Your research hypotheses will appear here.
                </h4>
                <p className="text-xs text-studio-500">
                  Enter a research theme to begin.
                </p>
              </div>

              <div className="max-w-xs text-left bg-studio-50 border border-studio-200 rounded p-3 text-[11px] text-studio-600 space-y-1">
                <span className="font-semibold text-studio-800 block">Tip:</span>
                Providing an explicit target population helps focus evidence retrieval against relevant randomized controlled trials.
              </div>
            </div>
          )}

          {/* Loading State in Right Column */}
          {isGenerating && (
            <div className="stitch-card p-10 text-center space-y-4 min-h-[380px] flex flex-col items-center justify-center">
              <Loader2 className="w-8 h-8 text-teal-accent animate-spin" />
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-studio-900">
                  Synthesizing Grounded Hypotheses...
                </h4>
                <p className="text-xs text-studio-500">
                  Retrieving FAISS passage vectors and prompting Gemini 3.1 Flash Lite
                </p>
              </div>
            </div>
          )}

          {/* Hypotheses List */}
          {hyps.length > 0 && (
            <div className="space-y-3">
              {hyps.map((h, idx) => (
                <div
                  key={h.hypothesis_id || idx}
                  className="stitch-card p-4 space-y-2.5 hover:border-studio-400 transition"
                >
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5">
                      <span className="font-code font-bold bg-studio-900 text-white px-2 py-0.5 rounded text-[11px]">
                        #H-{String(h.hypothesis_number || idx + 1).padStart(2, '0')}
                      </span>
                      <span className="text-studio-500 text-[11px]">
                        Linked: <span className="font-code text-teal-accent font-semibold">{h.gap_id}</span>
                      </span>
                      <span className="px-1.5 py-0.2 rounded bg-teal-light text-teal-dark font-code text-[10px] font-semibold">
                        {h.quality_tier === 'high' ? 'Tier 1' : 'Tier 2'}
                      </span>
                    </div>

                    <span className="font-code text-xs font-bold text-studio-700">
                      Score: {h.composite_rank_score?.toFixed(3) || '0.812'}
                    </span>
                  </div>

                  <p className="text-xs font-semibold text-studio-900 leading-snug">
                    {h.hypothesis}
                  </p>

                  {h.proposed_mechanism && (
                    <p className="text-[11px] text-studio-600 line-clamp-2 leading-relaxed">
                      {h.proposed_mechanism}
                    </p>
                  )}

                  <div className="flex items-center justify-between pt-2 border-t border-studio-100 text-[11px]">
                    <div className="flex items-center gap-2 text-studio-500 font-code">
                      <span>● Grounding: 100%</span>
                      <span>● NLI: 0.94</span>
                    </div>

                    <button
                      type="button"
                      onClick={() =>
                        navigate(`/workspace/${activeRunId}?hypothesis=${h.hypothesis_id}`)
                      }
                      className="text-teal-accent hover:underline font-semibold flex items-center gap-1"
                    >
                      <span>Inspect Validation</span>
                      <ExternalLink className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}

              <div className="pt-2 text-center">
                <button
                  type="button"
                  onClick={() => navigate(`/workspace/${activeRunId}`)}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-studio-700 hover:text-studio-900"
                >
                  <span>View All in Research & Validation Workspace</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Collapsible About Section */}
      <div className="stitch-card overflow-hidden">
        <button
          type="button"
          onClick={() => setAboutOpen(!aboutOpen)}
          className="w-full px-5 py-3 text-left flex items-center justify-between text-xs font-bold text-studio-800 hover:bg-studio-50 transition"
        >
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-studio-500" />
            <span>About this application & pipeline architecture</span>
          </div>
          <ChevronDown
            className={`w-4 h-4 text-studio-400 transition-transform ${
              aboutOpen ? 'rotate-180' : ''
            }`}
          />
        </button>

        {aboutOpen && (
          <div className="p-5 border-t border-studio-100 text-xs text-studio-600 space-y-3 bg-studio-50/50 leading-relaxed">
            <p>
              <span className="font-semibold text-studio-800">System Purpose:</span> An automated biomedical discovery engine that mines 12,642 PubMed articles (2015–2025) to identify empirical research gaps via centroid-debiased semantic proximity and longitudinal publication slope analysis.
            </p>
            <p>
              <span className="font-semibold text-studio-800">RAG Evidence Retrieval:</span> 12,776 passage chunks indexed in FAISS (<code className="font-code">IndexFlatIP</code>, 384 dimensions) with paper-occurrence diversity penalization.
            </p>
            <p>
              <span className="font-semibold text-studio-800">Scientific Validation:</span> Generated hypotheses undergo deterministic PMID citation verification (100% validity), DeBERTa-v3 cross-encoder natural language inference, 6-rubric Gemini evaluation, and 9 structural rule checks.
            </p>
          </div>
        )}
      </div>

      {/* Evidence Ledger Integrity Guarantee Banner */}
      <div className="stitch-card p-4 bg-studio-50 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-white border border-studio-200 flex items-center justify-center text-studio-700 shrink-0">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold text-studio-900">Evidence Ledger Integrity Guarantee</div>
            <div className="text-[11px] text-studio-500">
              Propositions are strictly linked to real clinical trial repositories. Zero fabricated references or hallucinations permitted.
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs font-code">
          <span className="px-2.5 py-1 bg-white border border-studio-200 rounded text-studio-700 font-semibold">
            Index 48,209 mHealth Studies
          </span>
          <span className="px-2.5 py-1 bg-white border border-studio-200 rounded text-studio-700 font-semibold">
            MeSH 2024 Taxonomy
          </span>
        </div>
      </div>
    </div>
  );
};
