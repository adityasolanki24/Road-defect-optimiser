export interface Defect {
  id: string;
  latitude: number;
  longitude: number;
  severity: number;
  defect_type: string;
  timestamp: string;
  traffic_density: number;
  location_importance: number;
  risk_factor: number;
  estimated_repair_time_hours: number;
}

export interface Weights {
  severity: number;
  traffic_density: number;
  location_importance: number;
  risk_factor: number;
}

export interface RoutePoint {
  defect_id: string;
  latitude: number;
  longitude: number;
  priority_score: number;
  severity: number;
  estimated_repair_time_hours: number;
  sequence: number;
}

export interface OptimisedRoute {
  route: RoutePoint[];
  total_distance: number;
  total_time: number;
  estimated_cost: number;
}

export interface ResourceStatus {
  crew_count: number;
  available_crews: number;
  availability_percent: number;
  daily_cost: number;
}

export interface Metrics {
  total_defects: number;
  active_high_priority_defects: number;
  total_repair_time: number;
  total_distance: number;
  estimated_cost: number;
  cost_reduction_percent: number;
  fuel_savings_percent: number;
  resource_status: ResourceStatus;
}

export interface MLStatus {
  ready: boolean;
  defect_classifier_loaded: boolean;
  severity_forecaster_loaded: boolean;
  severity_regressor_loaded: boolean;
  error: string | null;
}

export interface DefectDetectionResult {
  defect_type: string;
  confidence: number;
  severity: number;
  estimated_repair_time_hours: number;
  class_probabilities: Record<string, number>;
  latitude: number | null;
  longitude: number | null;
  created_defect: Defect | null;
}

export interface SeverityForecast {
  defect_id: string;
  current_severity: number;
  predicted_severity_7d: number;
  predicted_severity_14d: number;
  days_until_critical: number | null;
  risk_trend: 'rising' | 'stable' | 'falling' | string;
  age_days: number;
  forecasted_at: string;
}
