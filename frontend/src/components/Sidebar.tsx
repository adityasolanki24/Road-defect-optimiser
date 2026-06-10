import { motion } from 'framer-motion';
import type { Defect, Metrics } from '../types';

interface SidebarProps {
  defects: Defect[];
  metrics: Metrics | null;
  isLoading: boolean;
  lastUpdated: Date | null;
}

const numberFormatter = new Intl.NumberFormat('en-AU');
const currencyFormatter = new Intl.NumberFormat('en-AU', {
  style: 'currency',
  currency: 'AUD',
  maximumFractionDigits: 0,
});

const severityBands = [
  {
    label: 'High',
    className: 'bg-red-500',
    predicate: (defect: Defect) => defect.severity >= 0.7,
  },
  {
    label: 'Medium',
    className: 'bg-yellow-400',
    predicate: (defect: Defect) => defect.severity >= 0.4 && defect.severity < 0.7,
  },
  {
    label: 'Low',
    className: 'bg-emerald-400',
    predicate: (defect: Defect) => defect.severity < 0.4,
  },
];

const Sidebar = ({ defects, metrics, isLoading, lastUpdated }: SidebarProps) => {
  const totalDefects = metrics?.total_defects ?? defects.length;
  const highPriorityDefects =
    metrics?.active_high_priority_defects ??
    defects.filter((defect) => defect.severity >= 0.7).length;
  const averageRepairTime = defects.length
    ? defects.reduce((sum, defect) => sum + defect.estimated_repair_time_hours, 0) / defects.length
    : 0;

  const overviewCards = [
    {
      label: 'Total defects',
      value: numberFormatter.format(totalDefects),
      detail: 'Registered records',
    },
    {
      label: 'High priority',
      value: numberFormatter.format(highPriorityDefects),
      detail: 'Severity >= 70%',
    },
    {
      label: 'Route distance',
      value: metrics ? `${metrics.total_distance.toFixed(1)} km` : '0.0 km',
      detail: 'Optimised crew path',
    },
    {
      label: 'Route cost',
      value: metrics ? currencyFormatter.format(metrics.estimated_cost) : '$0',
      detail: 'Labour + vehicle estimate',
    },
  ];

  return (
    <motion.aside
      initial={{ opacity: 0, x: -18 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className="surface-panel h-full min-h-0 overflow-hidden"
    >
      <div className="panel-scroll h-full overflow-y-auto p-4 md:p-5">
        <div className="panel-heading">
          <div>
            <p className="section-kicker">Overview</p>
            <h2 className="panel-title">Network posture</h2>
          </div>
          <span className="status-pill status-pill--compact">
            {isLoading ? 'Loading' : 'Current'}
          </span>
        </div>

        <div className="metric-grid mt-4">
          {overviewCards.map((card) => (
            <section key={card.label} className="surface-card surface-card--metric">
              <p className="card-label">{card.label}</p>
              <p className="card-value">{card.value}</p>
              <p className="card-caption">{card.detail}</p>
            </section>
          ))}
        </div>

        <section className="surface-card mt-4">
          <div className="section-header">
            <div>
              <p className="section-kicker">Severity</p>
              <h3>Defect mix</h3>
            </div>
            <span>{totalDefects} total</span>
          </div>

          <div className="mt-4 space-y-4">
            {severityBands.map((band) => {
              const count = defects.filter(band.predicate).length;
              const share = totalDefects ? Math.round((count / totalDefects) * 100) : 0;

              return (
                <div key={band.label}>
                  <div className="mb-2 flex items-center justify-between text-sm">
                    <span className="font-medium text-zinc-200">{band.label}</span>
                    <span className="text-zinc-500">
                      {count} / {share}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className={`h-full rounded-full ${band.className} transition-all duration-300`}
                      style={{ width: `${share}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="surface-card mt-4">
          <div className="section-header">
            <div>
              <p className="section-kicker">Execution</p>
              <h3>Operating forecast</h3>
            </div>
            <span>{lastUpdated ? 'Live' : 'Pending'}</span>
          </div>

          <div className="forecast-list mt-4">
            <div>
              <span>Average repair time</span>
              <strong>{averageRepairTime.toFixed(1)} h</strong>
            </div>
            <div>
              <span>Route distance</span>
              <strong>{metrics ? `${metrics.total_distance.toFixed(1)} km` : '0.0 km'}</strong>
            </div>
            <div>
              <span>Total crew hours</span>
              <strong>{metrics ? `${metrics.total_repair_time.toFixed(1)} h` : '0.0 h'}</strong>
            </div>
          </div>
        </section>
      </div>
    </motion.aside>
  );
};

export default Sidebar;
