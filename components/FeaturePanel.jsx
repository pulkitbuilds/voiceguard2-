'use client';

function Metric({ label, value, unit, flagged }) {
  return (
    <div className={`flex items-baseline justify-between py-1.5 border-b border-line/60 ${flagged ? 'text-danger' : 'text-slate-200'}`}>
      <span className="text-xs text-slate-400">{label}</span>
      <span className="font-mono text-sm tabular-nums">
        {value}
        {unit ? <span className="text-slate-500 ml-0.5">{unit}</span> : null}
      </span>
    </div>
  );
}

export default function FeaturePanel({ featureMap, heuristicFlags }) {
  if (!featureMap) {
    return <div className="text-sm text-slate-500">Run an analysis to see acoustic telemetry.</div>;
  }

  const flaggedIds = new Set((heuristicFlags || []).map((f) => f.id));

  const rows = [
    { label: 'Pitch (F0) mean', key: 'pitchMean', unit: 'Hz', flagId: null, fmt: (v) => v.toFixed(1) },
    { label: 'Pitch variability', key: 'pitchStd', unit: 'Hz', flagId: 'flat_prosody', fmt: (v) => v.toFixed(2) },
    { label: 'Jitter', key: 'jitter', unit: '', flagId: 'low_jitter', fmt: (v) => v.toFixed(4) },
    { label: 'Shimmer', key: 'shimmer', unit: '', flagId: 'low_shimmer', fmt: (v) => v.toFixed(4) },
    { label: 'Spectral flatness', key: 'spectralFlatnessMean', unit: '', flagId: 'spectral_flatness', fmt: (v) => v.toFixed(3) },
    { label: 'Spectral centroid', key: 'spectralCentroidMean', unit: 'Hz', flagId: null, fmt: (v) => v.toFixed(0) },
    { label: 'HF energy ratio', key: 'hfEnergyRatio', unit: '', flagId: 'hf_artifact', fmt: (v) => v.toFixed(3) },
    { label: 'Silence ratio', key: 'silenceRatio', unit: '', flagId: 'no_breaths', fmt: (v) => v.toFixed(3) },
    { label: 'Voiced ratio', key: 'pitchVoicedRatio', unit: '', flagId: null, fmt: (v) => v.toFixed(2) },
  ];

  return (
    <div>
      {rows.map((r) => (
        <Metric
          key={r.key}
          label={r.label}
          unit={r.unit}
          value={r.fmt(featureMap[r.key] ?? 0)}
          flagged={r.flagId ? flaggedIds.has(r.flagId) : false}
        />
      ))}
    </div>
  );
}
