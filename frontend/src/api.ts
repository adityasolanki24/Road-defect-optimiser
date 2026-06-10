import axios from 'axios';
import type {
  Defect,
  DefectDetectionResult,
  Metrics,
  MLStatus,
  OptimisedRoute,
  SeverityForecast,
  Weights,
} from './types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8001',
  timeout: 15000,
});

export const getDefects = async (): Promise<Defect[]> => {
  const response = await api.get<Defect[]>('/defects');
  return response.data;
};

export const getMetrics = async (): Promise<Metrics> => {
  const response = await api.get<Metrics>('/metrics');
  return response.data;
};

export const getWeights = async (): Promise<Weights> => {
  const response = await api.get<Weights>('/weights');
  return response.data;
};

export const computeRoute = async (weights: Weights): Promise<OptimisedRoute> => {
  const response = await api.post<OptimisedRoute>('/compute-route', weights);
  return response.data;
};

export const getMlStatus = async (): Promise<MLStatus> => {
  const response = await api.get<MLStatus>('/ml/status');
  return response.data;
};

export const detectDefect = async (
  file: File,
  options: {
    latitude?: number;
    longitude?: number;
    registerDefect?: boolean;
  } = {},
): Promise<DefectDetectionResult> => {
  const formData = new FormData();
  formData.append('image', file);

  if (options.latitude !== undefined) {
    formData.append('latitude', String(options.latitude));
  }
  if (options.longitude !== undefined) {
    formData.append('longitude', String(options.longitude));
  }
  if (options.registerDefect) {
    formData.append('register_defect', 'true');
  }

  const response = await api.post<DefectDetectionResult>('/ml/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getDefectForecast = async (defectId: string): Promise<SeverityForecast> => {
  const response = await api.get<SeverityForecast>(`/ml/forecast/${defectId}`);
  return response.data;
};
