// frontend/src/components/Header.tsx — Stitch design header with nav tabs & pipeline status

import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Download, User } from 'lucide-react';

interface HeaderProps {
  currentRunId?: string;
  pipelineActive?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ currentRunId, pipelineActive = true }) => {
  const location = useLocation();

  const isGenerate = location.pathname === '/';
  const isWorkspace = location.pathname.startsWith('/workspace') && !location.pathname.includes('/hypotheses/');
  const isValidation = location.pathname.includes('/hypotheses/');

  const targetRunId = currentRunId || 'latest';

  return (
    <header className="bg-white border-b border-studio-200 sticky top-0 z-50">
      <div className="max-w-[1280px] mx-auto px-6">
        {/* Top bar */}
        <div className="flex items-center justify-between h-14">
          {/* Left: Logo & branding */}
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-8 h-8 bg-studio-100 rounded flex items-center justify-center">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-studio-600">
                  <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
              </div>
              <div>
                <div className="text-sm font-semibold text-studio-900 leading-tight">
                  mHealth Research Studio
                </div>
                <div className="text-[10px] text-studio-500 leading-tight">
                  Evidence-grounded hypothesis generation & gap discovery
                </div>
              </div>
            </Link>

            <span className="text-[10px] font-code bg-studio-100 text-studio-600 px-2 py-0.5 rounded border border-studio-200">
              V2.4-MONOGRAPH
            </span>
          </div>

          {/* Center: Navigation tabs */}
          <nav className="flex items-center gap-0">
            <Link
              to="/"
              className={`px-4 py-4 text-sm font-medium border-b-2 transition-colors ${
                isGenerate
                  ? 'text-studio-900 border-studio-900 font-semibold'
                  : 'text-studio-500 border-transparent hover:text-studio-700'
              }`}
            >
              Generate Hypotheses
            </Link>
            <Link
              to={`/workspace/${targetRunId}`}
              className={`px-4 py-4 text-sm font-medium border-b-2 transition-colors ${
                isWorkspace
                  ? 'text-studio-900 border-studio-900 font-semibold'
                  : 'text-studio-500 border-transparent hover:text-studio-700'
              }`}
            >
              Research & Validation Workspace
            </Link>
            {isValidation && (
              <span className="px-4 py-4 text-sm font-semibold text-studio-900 border-b-2 border-studio-900">
                Scientific Validation
              </span>
            )}
          </nav>

          {/* Right: Status & actions */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs text-studio-600">
              <span className={`w-2 h-2 rounded-full ${pipelineActive ? 'bg-teal-accent' : 'bg-studio-400'}`} />
              <span className="font-medium">Synthesis Pipeline:</span>
              <span className={pipelineActive ? 'text-teal-accent font-semibold' : 'text-studio-400'}>
                {pipelineActive ? 'Active' : 'Idle'}
              </span>
            </div>

            <button className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-studio-700 border border-studio-200 rounded hover:bg-studio-50 transition">
              <Download className="w-3.5 h-3.5" />
              Export Report
            </button>

            <div className="w-8 h-8 bg-studio-100 rounded-full flex items-center justify-center">
              <User className="w-4 h-4 text-studio-500" />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
