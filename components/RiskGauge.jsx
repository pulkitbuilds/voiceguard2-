'use client';

const COLOR_HEX = {
  safe: '#34d399',
  warn: '#fbbf24',
  danger: '#f87171',
};

export default function RiskGauge({ score, level, color, summary }) {
  const clamped = Math.max(0, Math.min(100, score ?? 0));
  const radius = 90;
  const circumference = 2 * Math.PI * radius;
  const pct = clamped / 100;
  const dash = circumference * pct;
  const hex = COLOR_HEX[color] || '#5eead4';
  const isCritical = level === 'CRITICAL';

  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <div className={`relative ${isCritical ? 'pulse-danger' : ''} rounded-full`}>
        <svg width="220" height="220" viewBox="0 0 220 220" className="-rotate-90">
          <circle cx="110" cy="110" r={radius} fill="none" stroke="#1d2632" strokeWidth="14" />
          <circle
            cx="110"
            cy="110"
            r={radius}
            fill="none"
            stroke={hex}
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference - dash}`}
            style={{ transition: 'stroke-dasharray 700ms ease, stroke 400ms ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-5xl font-semibold tabular-nums" style={{ color: hex }}>
            {clamped}
          </span>
          <span className="text-xs text-slate-400 mt-1 font-mono">risk / 100</span>
        </div>
      </div>
      <div className="text-center">
        <div className="text-sm font-medium tracking-wide" style={{ color: hex }}>
          {level || '—'}
        </div>
        <div className="text-xs text-slate-400 mt-0.5 max-w-[220px]">{summary}</div>
      </div>
    </div>
  );
}
