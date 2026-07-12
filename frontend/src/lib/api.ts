/**
 * API Client for ECG Arrhythmia Detection Backend
 */

import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface BeatPrediction {
  beat_index: number;
  sample_index: number;
  timestamp_sec: number;
  predicted_class: 'N' | 'S' | 'V' | 'F' | 'Q';
  confidence: number;
  probabilities: Record<string, number>;
  original_symbol?: string;
}

export interface ClassDistribution {
  class_name: string;
  count: number;
  percentage: number;
}

export interface AbnormalSegment {
  start_time_sec: number;
  end_time_sec: number;
  duration_sec: number;
  abnormal_beat_indices: number[];
  dominant_class: string;
  avg_confidence: number;
}

export interface AnalysisSummary {
  total_beats: number;
  normal_beats: number;
  abnormal_beats: number;
  class_distribution: ClassDistribution[];
  abnormal_segments: AbnormalSegment[];
  overall_confidence: number;
  flagged_for_review: boolean;
}

export interface Analysis {
  analysis_id: string;
  recording_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  summary: AnalysisSummary | null;
  beat_predictions: BeatPrediction[] | null;
  created_at: string;
  processing_time_sec: number | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  database_connected: boolean;
}

export interface ModelInfo {
  name: string;
  description: string;
  available: boolean;
}

// API Functions
export const healthCheck = async (): Promise<HealthResponse> => {
  const response = await api.get<HealthResponse>('/api/health');
  return response.data;
};

export const getModels = async (): Promise<ModelInfo[]> => {
  const response = await api.get<{ models: ModelInfo[] }>('/api/models');
  return response.data.models;
};

export const analyzeSampleRecord = async (
  recordName: string
): Promise<Analysis> => {
  const response = await api.post<Analysis>('/api/analyze/sample', {
    record_name: recordName,
  });
  return response.data;
};

export const getAnalyses = async (
  skip = 0,
  limit = 10
): Promise<{ analyses: Partial<Analysis>[] }> => {
  const response = await api.get('/api/analyses', {
    params: { skip, limit },
  });
  return response.data;
};

export const getAnalysis = async (analysisId: string): Promise<Analysis> => {
  const response = await api.get<Analysis>(`/api/analyses/${analysisId}`);
  return response.data;
};

export const getBeatPredictions = async (
  analysisId: string,
  offset = 0,
  limit = 100
): Promise<{ total: number; beats: BeatPrediction[] }> => {
  const response = await api.get(`/api/analyses/${analysisId}/beats`, {
    params: { offset, limit },
  });
  return response.data;
};

export const deleteAnalysis = async (analysisId: string): Promise<void> => {
  await api.delete(`/api/analyses/${analysisId}`);
};

export default api;
