'use client';

const STYLES = {
  safe: { border: 'border-safe/40', bg: 'bg-safe/5', text: 'text-safe' },
  warn: { border: 'border-warn/40', bg: 'bg-warn/5', text: 'text-warn' },
  danger: { border: 'border-danger/40', bg: 'bg-danger/10', text: 'text-danger' },
};

export default function AlertBanner({ risk }) {
  if (!risk) return null;
  const s = STYLES[risk.color] || STYLES.safe;

  return (
    <div className={`card ${s.border} ${s.bg} border p-4`}>
      <div className={`text-sm font-medium ${s.text}`}>
        {risk.level} — {risk.summary}
      </div>
      {risk.heuristicFlags?.length > 0 && (
        <ul className="mt-2 space-y-1">
          {risk.heuristicFlags.map((f) => (
            <li key={f.id} className="text-xs text-slate-300 flex gap-2">
              <span className="text-slate-500">•</span>
              <span>{f.message}</span>
            </li>
          ))}
        </ul>
      )}
      {risk.recommendations?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-line/60">
          <div className="text-xs uppercase tracking-wide text-slate-500 mb-1.5">Recommended actions</div>
          <ul className="space-y-1">
            {risk.recommendations.map((r, i) => (
              <li key={i} className="text-xs text-slate-200 flex gap-2">
                <span className={s.text}>→</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
