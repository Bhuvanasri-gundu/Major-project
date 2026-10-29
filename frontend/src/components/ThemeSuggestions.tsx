// frontend/src/components/ThemeSuggestions.tsx — Debounced live suggestion dropdown

import React, { useEffect, useState, useRef } from 'react';
import { Sparkles, Layers, BookOpen, Loader2 } from 'lucide-react';
import { api } from '../services/api';
import type { SuggestionItem } from '../types';

interface ThemeSuggestionsProps {
  query: string;
  onSelectSuggestion: (suggestion: string) => void;
  onClose: () => void;
  isOpen: boolean;
}

export const ThemeSuggestions: React.FC<ThemeSuggestionsProps> = ({
  query,
  onSelectSuggestion,
  onClose,
  isOpen,
}) => {
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number>(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounced fetch
  useEffect(() => {
    if (!query || query.trim().length < 2) {
      setSuggestions([]);
      setIsLoading(false);
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    const timeoutId = setTimeout(async () => {
      try {
        const res = await api.getSuggestions(query.trim());
        setSuggestions(res.suggestions);
        setSelectedIndex(-1);
      } catch (err: any) {
        setError('Suggestions temporarily unavailable');
        setSuggestions([]);
      } finally {
        setIsLoading(false);
      }
    }, 250);

    return () => clearTimeout(timeoutId);
  }, [query]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev < suggestions.length - 1 ? prev + 1 : 0));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : suggestions.length - 1));
      } else if (e.key === 'Enter' && selectedIndex >= 0) {
        e.preventDefault();
        const chosen = suggestions[selectedIndex];
        if (chosen) {
          onSelectSuggestion(chosen.text);
        }
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, suggestions, selectedIndex, onSelectSuggestion, onClose]);

  if (!isOpen || (!isLoading && suggestions.length === 0 && !error && query.length < 2)) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className="absolute left-0 right-0 top-full mt-1.5 bg-white border border-slate-200 rounded-xl shadow-lg z-50 overflow-hidden divide-y divide-slate-100 animate-in fade-in slide-in-from-top-1 duration-150"
    >
      <div className="px-3.5 py-2 bg-slate-50/80 flex items-center justify-between text-xs text-slate-500 font-medium">
        <span className="flex items-center space-x-1.5">
          <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
          <span>Real Project Literature & Ontology Suggestions</span>
        </span>
        <span className="text-[11px] text-slate-400">↑↓ to navigate · Enter to select · Esc to dismiss</span>
      </div>

      <div className="max-h-64 overflow-y-auto p-1.5 space-y-1">
        {isLoading && (
          <div className="flex items-center justify-center space-x-2 py-4 text-xs text-slate-500">
            <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
            <span>Searching candidate gaps and MeSH ontology...</span>
          </div>
        )}

        {error && (
          <div className="py-2.5 px-3 text-xs text-amber-600 bg-amber-50 rounded-lg">
            {error} — you can continue typing freely.
          </div>
        )}

        {!isLoading && !error && suggestions.length === 0 && query.trim().length >= 2 && (
          <div className="py-3 px-3 text-xs text-slate-500 text-center">
            No direct candidate gap or ontology match found. You can submit your custom theme.
          </div>
        )}

        {!isLoading &&
          suggestions.map((item, idx) => {
            const isSelected = idx === selectedIndex;
            return (
              <button
                key={`${item.type}-${item.text}-${idx}`}
                type="button"
                onClick={() => onSelectSuggestion(item.text)}
                onMouseEnter={() => setSelectedIndex(idx)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs transition flex items-start space-x-2.5 ${
                  isSelected
                    ? 'bg-indigo-50 text-indigo-900 border border-indigo-200'
                    : 'hover:bg-slate-50 text-slate-700'
                }`}
              >
                <div className="mt-0.5 text-slate-400">
                  {item.type === 'gap' && <Layers className="w-3.5 h-3.5 text-indigo-500" />}
                  {item.type === 'concept' && <BookOpen className="w-3.5 h-3.5 text-emerald-500" />}
                  {item.type === 'topic' && <Sparkles className="w-3.5 h-3.5 text-sky-500" />}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="font-medium text-slate-900 truncate">{item.text}</div>
                  {item.details && (
                    <div className="text-[11px] text-slate-400 mt-0.5 flex items-center space-x-2">
                      <span className="capitalize font-semibold text-slate-500">{item.category}</span>
                      <span>·</span>
                      <span>{item.details}</span>
                    </div>
                  )}
                </div>
              </button>
            );
          })}
      </div>
    </div>
  );
};
