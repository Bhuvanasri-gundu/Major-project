// frontend/src/components/Header.tsx — Minimal academic header

import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sparkles, ArrowRight, ShieldCheck, Database } from 'lucide-react';

interface HeaderProps {
  currentRunId?: string;
}

export const Header: React.FC<HeaderProps> = ({ currentRunId }) => {
  const location = useLocation();
  const isWorkspace = location.pathname.startsWith('/workspace');
  const targetRunId = currentRunId || 'latest';

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
        {/* Brand & Subtitle */}
        <div className="flex items-center space-x-3">
          <Link to="/" className="flex items-center space-x-2.5 group">
            <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-sm group-hover:bg-indigo-700 transition">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-semibold text-slate-900 tracking-tight leading-tight group-hover:text-indigo-600 transition">
                Ontology-Guided Hypothesis Generation
              </h1>
              <p className="text-xs text-slate-500 font-medium leading-none mt-0.5">
                Evidence-grounded research hypothesis generation for mHealth research
              </p>
            </div>
          </Link>
        </div>

        {/* System Badges & Primary Action */}
        <div className="flex items-center space-x-3">
          <div className="hidden md:flex items-center space-x-2 text-xs text-slate-500 bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
            <Database className="w-3.5 h-3.5 text-slate-400" />
            <span>12,642 PubMed Screened</span>
            <span className="text-slate-300">·</span>
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            <span className="text-emerald-700 font-medium">Gemini Grounded</span>
          </div>

          {!isWorkspace && (
            <Link
              to={`/workspace/${targetRunId}`}
              className="inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-indigo-700 bg-indigo-50 border border-indigo-200 hover:bg-indigo-100 transition shadow-xs"
            >
              <span>Complete Validation Workspace</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          )}
        </div>
      </div>
    </header>
  );
};
