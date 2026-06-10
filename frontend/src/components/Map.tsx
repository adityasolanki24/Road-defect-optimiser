import { useEffect, useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  CircleMarker,
  MapContainer,
  Pane,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { getDefectForecast } from '../api';
import type { Defect, OptimisedRoute, RoutePoint, SeverityForecast, Weights } from '../types';

interface MapProps {
  defects: Defect[];
  route: OptimisedRoute | null;
  weights: Weights;
  isLoading: boolean;
}

interface OsrmRouteResponse {
  routes?: Array<{
    geometry?: {
      coordinates?: Array<[number, number]>;
    };
  }>;
}

type RouteStatus = 'idle' | 'loading' | 'street' | 'unavailable';

type MarkerItem =
  | { type: 'defect'; id: string; defect: Defect }
  | {
      type: 'cluster';
      id: string;
      latitude: number;
      longitude: number;
      defects: Defect[];
      severity: number;
    };

const OSRM_BASE_URL = 'https://router.project-osrm.org/route/v1/driving';
const ROAD_ROUTE_STOP_LIMIT = 24;
const ROAD_ROUTE_CHUNK_SIZE = 6;
const CLUSTER_THRESHOLD = 90;
const CLUSTER_GRID_SIZE = 0.012;
const routeGeometryCache = new Map<string, [number, number][]>();

const severityBand = (severity: number) => {
  if (severity >= 0.7) {
    return {
      label: 'High',
      color: '#ef4444',
      border: '#fecaca',
      radius: 9,
    };
  }

  if (severity >= 0.4) {
    return {
      label: 'Medium',
      color: '#facc15',
      border: '#fef08a',
      radius: 8,
    };
  }

  return {
    label: 'Low',
    color: '#34d399',
    border: '#bbf7d0',
    radius: 7,
  };
};

const priorityScore = (defect: Defect, weights: Weights) => {
  const total =
    weights.severity + weights.traffic_density + weights.location_importance + weights.risk_factor;
  const safeTotal = total > 0 ? total : 1;

  return (
    (weights.severity / safeTotal) * defect.severity +
    (weights.traffic_density / safeTotal) * defect.traffic_density +
    (weights.location_importance / safeTotal) * defect.location_importance +
    (weights.risk_factor / safeTotal) * defect.risk_factor
  );
};

const formatTimestamp = (timestamp: string) =>
  new Date(timestamp).toLocaleString('en-AU', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });

const buildMarkerItems = (defects: Defect[]): MarkerItem[] => {
  if (defects.length <= CLUSTER_THRESHOLD) {
    return defects.map((defect) => ({
      type: 'defect',
      id: defect.id,
      defect,
    }));
  }

  const cells = new Map<string, Defect[]>();

  defects.forEach((defect) => {
    const key = `${Math.round(defect.latitude / CLUSTER_GRID_SIZE)}:${Math.round(
      defect.longitude / CLUSTER_GRID_SIZE,
    )}`;
    cells.set(key, [...(cells.get(key) ?? []), defect]);
  });

  const items: MarkerItem[] = [];

  Array.from(cells.entries()).forEach(([key, groupedDefects]) => {
    if (groupedDefects.length < 4) {
      items.push(...groupedDefects.map((defect) => ({
        type: 'defect' as const,
        id: defect.id,
        defect,
      })));
      return;
    }

    const latitude =
      groupedDefects.reduce((sum, defect) => sum + defect.latitude, 0) / groupedDefects.length;
    const longitude =
      groupedDefects.reduce((sum, defect) => sum + defect.longitude, 0) / groupedDefects.length;
    const severity = Math.max(...groupedDefects.map((defect) => defect.severity));

    items.push({
      type: 'cluster',
      id: `cluster_${key}`,
      latitude,
      longitude,
      defects: groupedDefects,
      severity,
    });
  });

  return items;
};

const appendGeometry = (
  target: [number, number][],
  segment: [number, number][],
) => {
  target.push(...(target.length ? segment.slice(1) : segment));
};

const fetchOsrmGeometry = async (points: RoutePoint[]): Promise<[number, number][]> => {
  const coordinates = points.map((point) => `${point.longitude},${point.latitude}`).join(';');
  const response = await fetch(
    `${OSRM_BASE_URL}/${coordinates}?overview=full&geometries=geojson&steps=false`,
  );

  if (!response.ok) {
    throw new Error(`OSRM request failed with ${response.status}`);
  }

  const data = (await response.json()) as OsrmRouteResponse;
  const geometry = data.routes?.[0]?.geometry?.coordinates;

  if (!geometry?.length) {
    throw new Error('OSRM response did not include road geometry');
  }

  return geometry.map(([longitude, latitude]) => [latitude, longitude] as [number, number]);
};

const fetchChunkedRoadGeometry = async (routePoints: RoutePoint[]) => {
  const positions: [number, number][] = [];

  for (let index = 0; index < routePoints.length - 1; index += ROAD_ROUTE_CHUNK_SIZE - 1) {
    const chunk = routePoints.slice(index, index + ROAD_ROUTE_CHUNK_SIZE);
    const geometry = await fetchOsrmGeometry(chunk);
    appendGeometry(positions, geometry);
  }

  return positions;
};

const fetchPairwiseRoadGeometry = async (routePoints: RoutePoint[]) => {
  const positions: [number, number][] = [];

  for (let index = 0; index < routePoints.length - 1; index += 1) {
    const geometry = await fetchOsrmGeometry([routePoints[index], routePoints[index + 1]]);
    appendGeometry(positions, geometry);
  }

  return positions;
};

const fetchRoadNetworkGeometry = async (routePoints: RoutePoint[]) => {
  try {
    return await fetchChunkedRoadGeometry(routePoints);
  } catch (chunkError) {
    console.warn('Chunked road route failed; retrying segment-by-segment.', chunkError);
    return fetchPairwiseRoadGeometry(routePoints);
  }
};

const RoadMap = ({ defects, route, weights, isLoading }: MapProps) => {
  const [streetRoutePositions, setStreetRoutePositions] = useState<[number, number][]>([]);
  const [routeStatus, setRouteStatus] = useState<RouteStatus>('idle');
  const [selectedDefect, setSelectedDefect] = useState<Defect | null>(null);
  const [forecast, setForecast] = useState<SeverityForecast | null>(null);
  const [forecastError, setForecastError] = useState<string | null>(null);

  const markerItems = useMemo(() => buildMarkerItems(defects), [defects]);
  const routeKey = useMemo(
    () =>
      route?.route
        .slice(0, ROAD_ROUTE_STOP_LIMIT)
        .map((point) => `${point.defect_id}:${point.latitude}:${point.longitude}`)
        .join('|') ?? '',
    [route],
  );
  const highSeverityCount = defects.filter((defect) => defect.severity >= 0.7).length;

  useEffect(() => {
    let isCancelled = false;

    const buildStreetRoute = async () => {
      const routePoints = route?.route.slice(0, ROAD_ROUTE_STOP_LIMIT) ?? [];

      if (routePoints.length < 2) {
        setStreetRoutePositions([]);
        setRouteStatus('idle');
        return;
      }

      const cachedGeometry = routeGeometryCache.get(routeKey);
      if (cachedGeometry) {
        setStreetRoutePositions(cachedGeometry);
        setRouteStatus('street');
        return;
      }

      setRouteStatus('loading');

      try {
        const roadGeometry = await fetchRoadNetworkGeometry(routePoints);

        if (roadGeometry.length < 2) {
          throw new Error('Road geometry returned fewer than two points');
        }

        if (!isCancelled) {
          routeGeometryCache.set(routeKey, roadGeometry);
          setStreetRoutePositions(roadGeometry);
          setRouteStatus('street');
        }
      } catch (error) {
        console.error('Error fetching road route:', error);
        if (!isCancelled) {
          setStreetRoutePositions([]);
          setRouteStatus('unavailable');
        }
      }
    };

    buildStreetRoute();

    return () => {
      isCancelled = true;
    };
  }, [routeKey, route]);

  useEffect(() => {
    if (!selectedDefect) {
      setForecast(null);
      setForecastError(null);
      return undefined;
    }

    let isCancelled = false;

    getDefectForecast(selectedDefect.id)
      .then((response) => {
        if (!isCancelled) {
          setForecast(response);
          setForecastError(null);
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setForecast(null);
          setForecastError('ML forecast unavailable');
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [selectedDefect]);

  const selectedSeverity = selectedDefect ? severityBand(selectedDefect.severity) : null;
  const selectedPriority = selectedDefect ? priorityScore(selectedDefect, weights) : 0;

  return (
    <section className="map-frame relative h-full min-h-[440px] overflow-hidden">
      <div className="map-overlay map-overlay--top-left">
        <p className="section-kicker">Map</p>
        <h2>Defect network</h2>
      </div>

      <div className="map-overlay map-overlay--top-right">
        <div>
          <span>Defects</span>
          <strong>{defects.length}</strong>
        </div>
        <div>
          <span>High</span>
          <strong>{highSeverityCount}</strong>
        </div>
        <div>
          <span>Stops</span>
          <strong>{route?.route.length ?? 0}</strong>
        </div>
        <div>
          <span>Path</span>
          <strong>
            {routeStatus === 'street'
              ? 'Road'
              : routeStatus === 'loading'
                ? 'Routing'
                : routeStatus === 'unavailable'
                  ? 'Unavailable'
                  : 'Pending'}
          </strong>
        </div>
      </div>

      <div className="map-overlay map-overlay--bottom-left">
        <div className="map-legend">
          <span>
            <i className="legend-dot legend-dot--high" />
            High
          </span>
          <span>
            <i className="legend-dot legend-dot--medium" />
            Medium
          </span>
          <span>
            <i className="legend-dot legend-dot--low" />
            Low
          </span>
        </div>
      </div>

      {isLoading && (
        <div className="map-overlay map-overlay--bottom-right">
          <span className="text-sm font-semibold text-zinc-100">Loading live data</span>
        </div>
      )}

      {!isLoading && routeStatus === 'unavailable' && (
        <div className="map-overlay map-overlay--bottom-right">
          <span className="text-sm font-semibold text-zinc-100">Road route unavailable</span>
        </div>
      )}

      <MapContainer
        center={[-37.8136, 144.9631]}
        zoom={12}
        scrollWheelZoom
        zoomControl={false}
        className="h-full w-full"
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap contributors &copy; CARTO'
        />

        <Pane name="route-pane" style={{ zIndex: 450 }}>
          {streetRoutePositions.length > 1 && (
            <>
              <Polyline
                positions={streetRoutePositions}
                color="#050505"
                weight={10}
                opacity={0.9}
              />
              <Polyline
                positions={streetRoutePositions}
                color="#f8fafc"
                weight={4}
                opacity={0.9}
              />
            </>
          )}
        </Pane>

        {markerItems.map((item) => {
          if (item.type === 'cluster') {
            const style = severityBand(item.severity);
            const highCount = item.defects.filter((defect) => defect.severity >= 0.7).length;

            return (
              <CircleMarker
                key={item.id}
                center={[item.latitude, item.longitude]}
                radius={Math.min(18, 10 + item.defects.length * 0.35)}
                pathOptions={{
                  color: style.border,
                  fillColor: style.color,
                  fillOpacity: 0.72,
                  weight: 2,
                }}
              >
                <Tooltip direction="top" offset={[0, -8]}>
                  {item.defects.length} defects
                </Tooltip>
                <Popup>
                  <div className="map-popup">
                    <p className="map-popup__eyebrow">Cluster</p>
                    <h3>{item.defects.length} defects</h3>
                    <div className="map-popup__grid">
                      <div>
                        <span>Highest severity</span>
                        <strong>{Math.round(item.severity * 100)}%</strong>
                      </div>
                      <div>
                        <span>High priority</span>
                        <strong>{highCount}</strong>
                      </div>
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          }

          const style = severityBand(item.defect.severity);

          return (
            <CircleMarker
              key={item.id}
              center={[item.defect.latitude, item.defect.longitude]}
              radius={style.radius}
              eventHandlers={{
                click: () => setSelectedDefect(item.defect),
              }}
              pathOptions={{
                color: style.border,
                fillColor: style.color,
                fillOpacity: 0.86,
                weight: 2,
              }}
            >
              <Tooltip direction="top" offset={[0, -8]}>
                {item.defect.id} | {style.label}
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {selectedDefect && selectedSeverity && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 16 }}
          className="defect-modal"
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="section-kicker">{selectedDefect.defect_type}</p>
              <h3>{selectedDefect.id}</h3>
            </div>
            <button type="button" onClick={() => setSelectedDefect(null)} className="modal-close">
              Close
            </button>
          </div>

          <div className="modal-grid mt-4">
            <div>
              <span>Severity</span>
              <strong style={{ color: selectedSeverity.color }}>{selectedSeverity.label}</strong>
            </div>
            <div>
              <span>Priority score</span>
              <strong>{Math.round(selectedPriority * 100)}%</strong>
            </div>
            <div>
              <span>Repair time</span>
              <strong>{selectedDefect.estimated_repair_time_hours.toFixed(1)} h</strong>
            </div>
            <div>
              <span>Reported</span>
              <strong>{formatTimestamp(selectedDefect.timestamp)}</strong>
            </div>
          </div>

          {forecast && (
            <div className="mt-4 rounded-2xl border border-zinc-700/80 bg-zinc-900/70 p-4">
              <p className="section-kicker">ML forecast</p>
              <div className="modal-grid mt-3">
                <div>
                  <span>7-day severity</span>
                  <strong>{Math.round(forecast.predicted_severity_7d * 100)}%</strong>
                </div>
                <div>
                  <span>14-day severity</span>
                  <strong>{Math.round(forecast.predicted_severity_14d * 100)}%</strong>
                </div>
                <div>
                  <span>Trend</span>
                  <strong className="capitalize">{forecast.risk_trend}</strong>
                </div>
                <div>
                  <span>Days to critical</span>
                  <strong>
                    {forecast.days_until_critical === null
                      ? 'Stable'
                      : `${forecast.days_until_critical.toFixed(1)} d`}
                  </strong>
                </div>
              </div>
            </div>
          )}

          {forecastError && !forecast && (
            <p className="mt-4 text-sm text-zinc-500">{forecastError}</p>
          )}
        </motion.div>
      )}
    </section>
  );
};

export default RoadMap;
