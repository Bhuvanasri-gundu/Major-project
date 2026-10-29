// frontend/src/services/api.ts — Centralized API client

import type {
  GenerateRequest,
  RunResponse,
  SuggestionsResponse,
  MatchedGap,
  EvaluationData,
} from '../types';

const API_BASE = '/api';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // fallback to status text
      errorDetail = response.statusText || errorDetail;
    }
    throw new ApiError(errorDetail, response.status);
  }

  return response.json();
}

export const api = {
  // Health check
  getHealth: async (): Promise<{ status: string; gemini_api_configured: boolean; configured_model: string }> => {
    return request('/health');
  },

  // Live theme suggestions
  getSuggestions: async (query: string): Promise<SuggestionsResponse> => {
    const encoded = encodeURIComponent(query);
    return request<SuggestionsResponse>(`/suggestions?q=${encoded}`);
  },

  // Generate hypotheses (starts and returns full run)
  generateHypotheses: async (data: GenerateRequest): Promise<RunResponse> => {
    return request<RunResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Get specific run
  getRun: async (runId: string): Promise<RunResponse> => {
    return request<RunResponse>(`/runs/${encodeURIComponent(runId)}`);
  },

  // Get latest run
  getLatestRun: async (): Promise<RunResponse> => {
    return request<RunResponse>('/runs/latest');
  },

  // Get candidate gaps
  getGaps: async (limit = 50): Promise<MatchedGap[]> => {
    return request<MatchedGap[]>(`/gaps?limit=${limit}`);
  },

  // Get project evaluation benchmarks
  getEvaluation: async (): Promise<EvaluationData> => {
    return request<EvaluationData>('/evaluation');
  },
};
