// frontend/src/components/Header.tsx - Clean design header with nav tabs

import React from 'react';
import { Link, useLocation } from 'react-router-dom';

interface HeaderProps {
  currentRunId?: string;
  pipelineActive?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ currentRunId }) => {
  const location = useLocation();

  const isGenerate = location.pathname === '/';
  const isWorkspace = location.pathname.startsWith('/workspace');

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
                  mHealth Research
                </div>
                <div className="text-[10px] text-studio-500 leading-tight">
                  Evidence-grounded hypothesis generation & gap discovery
                </div>
              </div>
            </Link>
          </div>

          {/* Right: Navigation tabs */}
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
          </nav>
        </div>
      </div>
    </header>
  );
};
