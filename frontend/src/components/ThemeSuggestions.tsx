// frontend/src/components/ThemeSuggestions.tsx — Debounced live suggestion dropdown (Stitch design)

import React, { useEffect, useState, useRef } from 'react';
import { Layers, BookOpen, Sparkles, Loader2 } from 'lucide-react';
import { api } from '../services/api';
import type { SuggestionItem } from '../types';

interface ThemeSuggestionsProps {
  query: string;
  onSelectSuggestion: (suggestion: string) => void;
  onClose: () => void;
  isOpen: boolean;
  triggerRef?: React.RefObject<HTMLElement | null>;
}

export const ThemeSuggestions: React.FC<ThemeSuggestionsProps> = ({
  query,
  onSelectSuggestion,
  onClose,
  isOpen,
  triggerRef,
}) => {
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number>(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounced fetch
  useEffect(() => {
    // If user opened with empty/short query, fetch general suggestions with default term
    const searchTerm = query && query.trim().length >= 2 ? query.trim() : 'health';

    setIsLoading(true);
    setError(null);
    const timeoutId = setTimeout(async () => {
      try {
        const res = await api.getSuggestions(searchTerm);
        setSuggestions(res.suggestions);
        setSelectedIndex(-1);
      } catch {
        setError('Suggestions temporarily unavailable');
        setSuggestions([]);
      } finally {
        setIsLoading(false);
      }
    }, 250);

    return () => clearTimeout(timeoutId);
  }, [query]);

  // Outside click handler
  useEffect(() => {
    if (!isOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;

      // If clicked inside the suggestions container, do nothing
      if (containerRef.current && containerRef.current.contains(target)) {
        return;
      }

      // If clicked inside the trigger element (e.g., Ontology Suggestions button), do nothing (it handles its own toggle)
      if (triggerRef?.current && triggerRef.current.contains(target)) {
        return;
      }

      // Otherwise, dismiss
      onClose();
    };

    document.addEventListener('pointerdown', handlePointerDown);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
    };
  }, [isOpen, onClose, triggerRef]);

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

  if (!isOpen) {
    return null;
  }

  const typeIcon = (type: string) => {
    switch (type) {
      case 'gap':
        return <Layers className="w-3.5 h-3.5 text-teal-accent" />;
      case 'concept':
        return <BookOpen className="w-3.5 h-3.5 text-teal-accent" />;
      default:
        return <Sparkles className="w-3.5 h-3.5 text-studio-500" />;
    }
  };

  const typeBadgeColor = (type: string) => {
    switch (type) {
      case 'gap':
        return 'bg-teal-light text-teal-dark';
      case 'concept':
        return 'bg-studio-100 text-teal-dark';
      default:
        return 'bg-studio-100 text-studio-600';
    }
  };

  return (
    <div
      ref={containerRef}
      className="absolute left-0 right-0 top-full mt-1 bg-white border border-studio-200 rounded-lg shadow-lg z-50 overflow-hidden"
    >
      {/* Header */}
      <div className="px-3 py-2 bg-studio-50 border-b border-studio-200 flex items-center justify-between">
        <span className="stitch-kicker flex items-center gap-1.5">
          <Sparkles className="w-3 h-3 text-teal-accent" />
          Ontology &amp; Literature Suggestions
        </span>
        <span className="text-[10px] font-code text-studio-400">
          ↑↓ navigate · Enter select · Esc close
        </span>
      </div>

      <div className="max-h-64 overflow-y-auto p-1.5 space-y-0.5">
        {isLoading && (
          <div className="flex items-center justify-center gap-2 py-4 text-xs text-studio-500">
            <Loader2 className="w-4 h-4 animate-spin text-teal-accent" />
            <span>Searching candidate gaps and MeSH ontology…</span>
          </div>
        )}

        {error && (
          <div className="py-2.5 px-3 text-xs text-studio-600 bg-studio-50 rounded">
            {error} — you can continue typing freely.
          </div>
        )}

        {!isLoading && !error && suggestions.length === 0 && query.trim().length >= 2 && (
          <div className="py-3 px-3 text-xs text-studio-500 text-center">
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
                className={`w-full text-left px-3 py-2 rounded text-xs transition flex items-start gap-2.5 ${
                  isSelected
                    ? 'bg-teal-light/50 text-studio-900'
                    : 'hover:bg-studio-50 text-studio-700'
                }`}
              >
                <div className="mt-0.5 shrink-0">{typeIcon(item.type)}</div>

                <div className="flex-1 min-w-0">
                  <div className="font-medium text-studio-900 truncate">{item.text}</div>
                  {item.details && (
                    <div className="text-[11px] text-studio-400 mt-0.5 flex items-center gap-2">
                      <span className={`inline-flex px-1.5 py-0.5 rounded text-[10px] font-semibold capitalize ${typeBadgeColor(item.type)}`}>
                        {item.category || item.type}
                      </span>
                      <span>·</span>
                      <span className="truncate">{item.details}</span>
                    </div>
                  )}
                </div>

                {item.score > 0 && (
                  <span className="font-code text-[10px] text-studio-400 shrink-0 mt-0.5">
                    {item.score.toFixed(2)}
                  </span>
                )}
              </button>
            );
          })}
      </div>
    </div>
  );
};
