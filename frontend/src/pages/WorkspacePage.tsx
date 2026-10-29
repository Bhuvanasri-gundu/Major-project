// frontend/src/pages/WorkspacePage.tsx — Screen 2: Complete Research & Validation Workspace

import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import { api } from '../services/api';
import { GenerationStatusSection } from '../components/workspace/GenerationStatusSection';
import { GapExplorerSection } from '../components/workspace/GapExplorerSection';
import { HypothesisResultsSection } from '../components/workspace/HypothesisResultsSection';
import { ValidationWorkspaceSection } from '../components/workspace/ValidationWorkspaceSection';
import { ProjectEvaluationSection } from '../components/workspace/ProjectEvaluationSection';
import type { RunResponse, HypothesisOutput } from '../types';

export const WorkspacePage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [run, setRun] = useState<RunResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Active selected hypothesis ID within the workspace
  const [selectedHypothesisId, setSelectedHypothesisId] = useState<string>('');

  const targetRunId = runId || 'latest';

  const fetchRunData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      let data: RunResponse;
      if (targetRunId === 'latest') {
        data = await api.getLatestRun();
      } else {
        data = await api.getRun(targetRunId);
      }
      setRun(data);

      // Select hypothesis from query param if available, otherwise first hypothesis
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
  }, [targetRunId, searchParams]);

  // Find currently selected hypothesis object
  const selectedHypothesis: HypothesisOutput | undefined = run?.hypotheses.find(
    (h) => h.hypothesis_id === selectedHypothesisId
  );

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 space-y-8 animate-in fade-in duration-150">
      {/* Top Navigation & Back Action */}
      <div className="flex items-center justify-between border-b border-slate-200 pb-4 flex-wrap gap-2">
        <button
          type="button"
          onClick={() => navigate('/')}
          className="inline-flex items-center space-x-2 text-xs font-bold text-slate-700 hover:text-indigo-600 bg-white hover:bg-slate-50 border border-slate-200 px-3.5 py-2 rounded-xl transition shadow-2xs"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Hypothesis Generation (Screen 1)</span>
        </button>

        <div className="flex items-center space-x-3">
          <span className="text-xs text-slate-500 font-mono">
            Workspace: {run ? run.run_id : targetRunId}
          </span>
          <button
            type="button"
            onClick={fetchRunData}
            disabled={isLoading}
            className="p-2 text-slate-500 hover:text-slate-800 bg-white border border-slate-200 rounded-lg transition"
            title="Refresh run status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="bg-white rounded-2xl border border-slate-200 p-16 text-center space-y-3">
          <Loader2 className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
          <h3 className="text-sm font-bold text-slate-800">Loading Complete Research & Validation Dossier...</h3>
          <p className="text-xs text-slate-400">Fetching FAISS retrieval passages and scientific validation records</p>
        </div>
      )}

      {/* Error State */}
      {error && !isLoading && (
        <div className="bg-white rounded-2xl border border-rose-200 p-8 text-center space-y-4">
          <AlertCircle className="w-10 h-10 text-rose-500 mx-auto" />
          <div>
            <h3 className="text-base font-bold text-slate-900">Run Dossier Unavailable</h3>
            <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">{error}</p>
          </div>
          <button
            type="button"
            onClick={() => navigate('/')}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-700 transition"
          >
            <span>Return to Generation Screen</span>
          </button>
        </div>
      )}

      {/* Main Workspace Content — ALL IN ONE SCREEN */}
      {run && !isLoading && (
        <div className="space-y-8">
          {/* SECTION 1: Generation Status */}
          <GenerationStatusSection run={run} />

          {/* SECTION 2: Research Gap Explorer */}
          <GapExplorerSection
            matchedGaps={run.matched_gaps}
            hypotheses={run.hypotheses}
            selectedHypothesis={selectedHypothesis}
          />

          {/* SECTION 3: Hypothesis Results */}
          <HypothesisResultsSection
            hypotheses={run.hypotheses}
            selectedHypothesisId={selectedHypothesisId}
            onSelectHypothesis={(id) => setSelectedHypothesisId(id)}
            runStatus={run.status}
          />

          {/* SECTION 4: Complete Hypothesis Validation Workspace (for Selected Hypothesis) */}
          {selectedHypothesis && (
            <ValidationWorkspaceSection hypothesis={selectedHypothesis} />
          )}

          {/* SECTION 5: Project-Level Evaluation (Collapsible Subsection) */}
          <ProjectEvaluationSection />
        </div>
      )}
    </div>
  );
};
