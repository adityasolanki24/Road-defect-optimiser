import { useEffect, useRef, useState } from 'react';
import { detectDefect, getMlStatus } from '../api';
import type { DefectDetectionResult, MLStatus } from '../types';

interface MlPanelProps {
  onDefectRegistered: () => void;
}

const DEFAULT_LAT = -37.8136;
const DEFAULT_LON = 144.9631;

const MlPanel = ({ onDefectRegistered }: MlPanelProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<MLStatus | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DefectDetectionResult | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [latitude, setLatitude] = useState(DEFAULT_LAT);
  const [longitude, setLongitude] = useState(DEFAULT_LON);

  useEffect(() => {
    getMlStatus()
      .then(setStatus)
      .catch(() =>
        setStatus({
          ready: false,
          defect_classifier_loaded: false,
          severity_forecaster_loaded: false,
          severity_regressor_loaded: false,
          error: 'ML service unavailable',
        }),
      );
  }, []);

  useEffect(
    () => () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    },
    [previewUrl],
  );

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
    setIsDetecting(true);

    try {
      const detection = await detectDefect(file, {
        latitude,
        longitude,
        registerDefect: true,
      });
      setResult(detection);
      onDefectRegistered();
    } catch (detectError) {
      console.error('Defect detection failed:', detectError);
      setError('Detection failed. Confirm the backend ML models are loaded.');
      setResult(null);
    } finally {
      setIsDetecting(false);
      event.target.value = '';
    }
  };

  return (
    <section className="surface-card mt-4">
      <div className="section-header">
        <div>
          <p className="section-kicker">Machine learning</p>
          <h3>Defect detection</h3>
        </div>
        <span>{status?.ready ? 'Models live' : 'Unavailable'}</span>
      </div>

      <p className="mt-3 text-sm text-zinc-400">
        Upload a road image. Type comes from the RDD2022-trained classifier; severity is predicted from
        damage size in the image.
      </p>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <label className="slider-shell">
          <span className="text-xs text-zinc-400">Latitude</span>
          <input
            type="number"
            step="0.0001"
            value={latitude}
            onChange={(event) => setLatitude(Number(event.target.value))}
            className="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100"
          />
        </label>
        <label className="slider-shell">
          <span className="text-xs text-zinc-400">Longitude</span>
          <input
            type="number"
            step="0.0001"
            value={longitude}
            onChange={(event) => setLongitude(Number(event.target.value))}
            className="mt-2 w-full rounded-xl border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-100"
          />
        </label>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      <button
        type="button"
        className="preset-button mt-4 w-full"
        disabled={!status?.ready || isDetecting}
        onClick={() => fileInputRef.current?.click()}
      >
        {isDetecting ? 'Analysing image...' : 'Upload road image'}
      </button>

      {previewUrl && (
        <img src={previewUrl} alt="Uploaded road surface" className="mt-4 h-28 w-full rounded-2xl object-cover" />
      )}

      {error && <p className="mt-3 text-sm text-red-300">{error}</p>}

      {result && (
        <>
          <div className="modal-grid mt-4">
            <div>
              <span>Type</span>
              <strong>{result.defect_type}</strong>
            </div>
            <div>
              <span>Confidence</span>
              <strong>{Math.round(result.confidence * 100)}%</strong>
            </div>
            <div>
              <span>Damage severity</span>
              <strong>{Math.round(result.severity * 100)}%</strong>
            </div>
            <div>
              <span>Repair estimate</span>
              <strong>{result.estimated_repair_time_hours.toFixed(1)} h</strong>
            </div>
          </div>

          <div className="mt-4 space-y-1">
            <p className="text-xs uppercase tracking-wide text-zinc-500">Class probabilities</p>
            {Object.entries(result.class_probabilities)
              .sort(([, a], [, b]) => b - a)
              .map(([label, score]) => (
                <div key={label} className="flex items-center justify-between text-sm text-zinc-300">
                  <span>{label}</span>
                  <span>{Math.round(score * 100)}%</span>
                </div>
              ))}
          </div>
        </>
      )}
    </section>
  );
};

export default MlPanel;
