// frontend/src/App.tsx — Application Router & Providers

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
        <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-900">
          <Header />
          <main className="flex-1">
            <Routes>
              {/* SCREEN 1: Generate Hypotheses */}
              <Route path="/" element={<GeneratePage />} />

              {/* SCREEN 2: Complete Research & Validation Workspace */}
              <Route path="/workspace/:runId" element={<WorkspacePage />} />
              <Route path="/workspace" element={<Navigate to="/workspace/latest" replace />} />

              {/* Fallback to Screen 1 */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          <footer className="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-400">
            <div className="max-w-7xl mx-auto px-4 space-y-1">
              <p className="font-medium text-slate-600">
                Ontology-Guided Hypothesis Generation in mHealth Research
              </p>
              <p>
                B.Tech Computer Science & Engineering (AI & ML) Major Project · PubMed Screened Corpus (2015–2025)
              </p>
            </div>
          </footer>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
