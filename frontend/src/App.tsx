import { useCallback, useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import Dashboard from './components/Dashboard';
import Map from './components/Map';
import Sidebar from './components/Sidebar';
import { computeRoute, getDefects, getMetrics, getWeights } from './api';
import type { Defect, Metrics, OptimisedRoute, Weights } from './types';
import './App.css';

const DEFAULT_WEIGHTS: Weights = {
  severity: 0.4,
  traffic_density: 0.3,
  location_importance: 0.2,
  risk_factor: 0.1,
};

function App() {
  const [defects, setDefects] = useState<Defect[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [route, setRoute] = useState<OptimisedRoute | null>(null);
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [isLoading, setIsLoading] = useState(true);
  const [isRouteLoading, setIsRouteLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const weightsRef = useRef(weights);
  const hasHydrated = useRef(false);

  useEffect(() => {
    weightsRef.current = weights;
  }, [weights]);

  const refreshSnapshot = useCallback(async (nextWeights: Weights, showLoader = false) => {
    if (showLoader) {
      setIsLoading(true);
    }

    try {
      const [defectsResponse, routeResponse] = await Promise.all([
        getDefects(),
        computeRoute(nextWeights),
      ]);
      const metricsResponse = await getMetrics();

      setDefects(defectsResponse);
      setRoute(routeResponse);
      setMetrics(metricsResponse);
      setLastUpdated(new Date());
      setApiError(null);
    } catch (error) {
      console.error('Error refreshing dashboard data:', error);
      setApiError('Backend unavailable. Start the FastAPI service on port 8001.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    const hydrate = async () => {
      try {
        const serverWeights = await getWeights();
        if (!isMounted) {
          return;
        }
        setWeights(serverWeights);
        await refreshSnapshot(serverWeights, true);
      } catch (error) {
        console.error('Error hydrating dashboard:', error);
        if (isMounted) {
          await refreshSnapshot(DEFAULT_WEIGHTS, true);
        }
      } finally {
        hasHydrated.current = true;
      }
    };

    hydrate();

    const interval = window.setInterval(() => {
      refreshSnapshot(weightsRef.current);
    }, 5000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, [refreshSnapshot]);

  useEffect(() => {
    if (!hasHydrated.current) {
      return undefined;
    }

    const timer = window.setTimeout(async () => {
      setIsRouteLoading(true);
      try {
        const routeResponse = await computeRoute(weights);
        const metricsResponse = await getMetrics();
        setRoute(routeResponse);
        setMetrics(metricsResponse);
        setLastUpdated(new Date());
        setApiError(null);
      } catch (error) {
        console.error('Error recomputing route:', error);
        setApiError('Route recompute failed. Check the backend service.');
      } finally {
        setIsRouteLoading(false);
      }
    }, 300);

    return () => window.clearTimeout(timer);
  }, [weights]);

  return (
    <div className="app-shell">
      <div className="flex h-screen min-h-0 flex-col overflow-hidden">
        <motion.header
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          className="app-topbar"
        >
          <div>
            <p className="section-kicker">Road Operations Command</p>
            <h1 className="app-title">Smart Road Maintenance Optimiser</h1>
          </div>

          <div className="topbar-status">
            <span className="status-pill">
              <span className="status-dot" />
              {isRouteLoading ? 'Recomputing route' : 'Live system'}
            </span>
            {lastUpdated && (
              <span className="status-pill">
                Updated{' '}
                {lastUpdated.toLocaleTimeString('en-AU', {
                  hour: '2-digit',
                  minute: '2-digit',
                  second: '2-digit',
                })}
              </span>
            )}
          </div>
        </motion.header>

        {apiError && <div className="mx-4 mb-3 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">{apiError}</div>}

        <main className="app-workspace min-h-0 flex-1 overflow-auto px-4 pb-4 xl:overflow-hidden">
          <Sidebar
            defects={defects}
            metrics={metrics}
            isLoading={isLoading}
            lastUpdated={lastUpdated}
          />

          <Map
            defects={defects}
            route={route}
            weights={weights}
            isLoading={isLoading}
          />

          <Dashboard
            route={route}
            metrics={metrics}
            weights={weights}
            isRouteLoading={isRouteLoading}
            onWeightsChange={setWeights}
            onDefectRegistered={() => refreshSnapshot(weightsRef.current)}
          />
        </main>
      </div>
    </div>
  );
}

export default App;
