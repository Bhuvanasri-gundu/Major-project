// frontend/src/App.tsx — Two-screen research platform with shared layout

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Header } from './components/Header';
import { GeneratePage } from './pages/GeneratePage';
import { WorkspacePage } from './pages/WorkspacePage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-white flex flex-col">
          <Header />
          <main className="flex-1">
            <Routes>
              {/* Screen 1: Generate Hypotheses — Monograph Studio */}
              <Route path="/" element={<GeneratePage />} />

              {/* Screen 2: Complete Research & Validation Workspace */}
              <Route path="/workspace/:runId" element={<WorkspacePage />} />
              <Route path="/workspace" element={<Navigate to="/workspace/latest" replace />} />

              {/* Fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          {/* Stitch footer */}
          <footer className="border-t border-studio-200 bg-white">
            <div className="max-w-[1280px] mx-auto px-6 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs text-studio-500">
                <span className="font-semibold text-studio-700">mHealth Research Studio</span>
                <span>•</span>
                <span className="italic">Biomedical Literature Index & Confidence Ledger</span>
              </div>
              <div className="flex items-center gap-3 text-xs text-studio-500">
                <span className="font-code">PMC / PubMed Core Linked</span>
                <span className="text-studio-300">|</span>
                <span className="font-semibold">Strict Academic Review Mode</span>
              </div>
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
