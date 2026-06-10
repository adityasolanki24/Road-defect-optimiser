import { motion } from 'framer-motion';
import MlPanel from './MlPanel';
import type { Metrics, OptimisedRoute, Weights } from '../types';

interface DashboardProps {
  route: OptimisedRoute | null;
  metrics: Metrics | null;
  weights: Weights;
  isRouteLoading: boolean;
  onWeightsChange: (weights: Weights) => void;
  onDefectRegistered: () => void;
}

const presets: Array<{ label: string; weights: Weights }> = [
  {
    label: 'Balanced',
    weights: {
      severity: 0.4,
      traffic_density: 0.3,
      location_importance: 0.2,
      risk_factor: 0.1,
    },
  },
  {
    label: 'Safety first',
    weights: {
      severity: 0.48,
      traffic_density: 0.18,
      location_importance: 0.14,
      risk_factor: 0.2,
    },
  },
  {
    label: 'Traffic impact',
    weights: {
      severity: 0.28,
      traffic_density: 0.42,
      location_importance: 0.2,
      risk_factor: 0.1,
    },
  },
];

const sliderDefinitions: Array<{
  key: keyof Weights;
  label: string;
}> = [
  { key: 'severity', label: 'Severity' },
  { key: 'traffic_density', label: 'Traffic' },
  { key: 'location_importance', label: 'Location' },
  { key: 'risk_factor', label: 'Risk' },
];

const currencyFormatter = new Intl.NumberFormat('en-AU', {
  style: 'currency',
  currency: 'AUD',
  maximumFractionDigits: 0,
});

const normaliseWeights = (weights: Weights): Weights => {
  const total =
    weights.severity + weights.traffic_density + weights.location_importance + weights.risk_factor;

  if (total <= 0) {
    return {
      severity: 0.25,
      traffic_density: 0.25,
      location_importance: 0.25,
      risk_factor: 0.25,
    };
  }

  return {
    severity: weights.severity / total,
    traffic_density: weights.traffic_density / total,
    location_importance: weights.location_importance / total,
    risk_factor: weights.risk_factor / total,
  };
};

const Dashboard = ({
  route,
  metrics,
  weights,
  isRouteLoading,
  onWeightsChange,
  onDefectRegistered,
}: DashboardProps) => {
  const updateWeight = (key: keyof Weights, percentValue: number) => {
    onWeightsChange(
      normaliseWeights({
        ...weights,
        [key]: percentValue / 100,
      }),
    );
  };

  const resourceStatus = metrics?.resource_status;
  const routeStops = route?.route.slice(0, 8) ?? [];

  return (
    <motion.aside
      initial={{ opacity: 0, x: 18 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="surface-panel h-full min-h-0 overflow-hidden"
    >
      <div className="panel-scroll h-full overflow-y-auto p-4 md:p-5">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">Optimization</p>
            <h2 className="panel-title">Route control</h2>
          </div>
          <span className="status-pill status-pill--compact">
            {isRouteLoading ? 'Calculating' : 'Ready'}
          </span>
        </div>

        <section className="surface-card mt-4">
          <div className="section-header">
            <div>
              <p className="section-kicker">Optimized Routes</p>
              <h3>Active plan</h3>
            </div>
            <span>{route?.route.length ?? 0} stops</span>
          </div>

          <div className="route-summary-grid mt-4">
            <div>
              <span>Distance</span>
              <strong>{route ? `${route.total_distance.toFixed(1)} km` : '0.0 km'}</strong>
            </div>
            <div>
              <span>Crew time</span>
              <strong>{route ? `${route.total_time.toFixed(1)} h` : '0.0 h'}</strong>
            </div>
            <div>
              <span>Spend</span>
              <strong>{route ? currencyFormatter.format(route.estimated_cost) : '$0'}</strong>
            </div>
          </div>

          <div className="mt-4 space-y-2">
            {routeStops.map((point) => (
              <div key={point.defect_id} className="route-row">
                <div className="route-row__index">{point.sequence}</div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-zinc-100">{point.defect_id}</p>
                  <p className="truncate text-xs text-zinc-500">
                    {point.estimated_repair_time_hours.toFixed(1)} h repair window
                  </p>
                </div>
                <div className="route-row__score">
                  <strong>{Math.round(point.priority_score * 100)}%</strong>
                  <span>priority</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="surface-card mt-4">
          <div className="section-header">
            <div>
              <p className="section-kicker">Resources</p>
              <h3>Crew status</h3>
            </div>
            <span>{resourceStatus?.availability_percent ?? 0}% available</span>
          </div>

          <div className="resource-grid mt-4">
            <div>
              <span>Crew count</span>
              <strong>{resourceStatus?.crew_count ?? 0}</strong>
            </div>
            <div>
              <span>Available</span>
              <strong>{resourceStatus?.available_crews ?? 0}</strong>
            </div>
            <div>
              <span>Daily cost</span>
              <strong>{currencyFormatter.format(resourceStatus?.daily_cost ?? 0)}</strong>
            </div>
          </div>

          <div className="mt-4 h-2 overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-emerald-400 transition-all duration-300"
              style={{ width: `${resourceStatus?.availability_percent ?? 0}%` }}
            />
          </div>
        </section>

        <section className="surface-card mt-4">
          <div className="section-header">
            <div>
              <p className="section-kicker">Weights</p>
              <h3>Dynamic ranking</h3>
            </div>
            <span>Live recompute</span>
          </div>

          <div className="mt-4 grid grid-cols-3 gap-2">
            {presets.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="preset-button"
                onClick={() => onWeightsChange(preset.weights)}
              >
                {preset.label}
              </button>
            ))}
          </div>

          <div className="mt-4 space-y-4">
            {sliderDefinitions.map((slider) => {
              const percentValue = Math.round(weights[slider.key] * 100);

              return (
                <div key={slider.key} className="slider-shell">
                  <div className="flex items-center justify-between gap-4">
                    <label>{slider.label}</label>
                    <span>{percentValue}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="1"
                    value={percentValue}
                    onChange={(event) => updateWeight(slider.key, Number(event.target.value))}
                    className="mt-3 w-full"
                  />
                </div>
              );
            })}
          </div>
        </section>

        <MlPanel onDefectRegistered={onDefectRegistered} />
      </div>
    </motion.aside>
  );
};

export default Dashboard;
