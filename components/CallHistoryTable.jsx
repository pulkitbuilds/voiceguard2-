'use client';

const COLOR_HEX = { safe: '#34d399', warn: '#fbbf24', danger: '#f87171' };

function timeAgo(iso) {
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.round(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  return `${h}h ago`;
}

export default function CallHistoryTable({ calls }) {
  if (!calls || calls.length === 0) {
    return <div className="text-sm text-slate-500 py-6 text-center">No calls analyzed yet this session.</div>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-slate-500 border-b border-line">
            <th className="py-2 pr-4 font-normal">Caller</th>
            <th className="py-2 pr-4 font-normal">Time</th>
            <th className="py-2 pr-4 font-normal">Duration</th>
            <th className="py-2 pr-4 font-normal">Channel</th>
            <th className="py-2 pr-4 font-normal">Risk</th>
            <th className="py-2 font-normal">Level</th>
          </tr>
        </thead>
        <tbody>
          {calls.map((c) => (
            <tr key={c.id} className="border-b border-line/50 hover:bg-panel2/60">
              <td className="py-2 pr-4 text-slate-200">{c.callerLabel}</td>
              <td className="py-2 pr-4 text-slate-400 font-mono text-xs">{timeAgo(c.timestamp)}</td>
              <td className="py-2 pr-4 text-slate-400 font-mono text-xs">
                {c.durationSec ? `${c.durationSec.toFixed(1)}s` : '—'}
              </td>
              <td className="py-2 pr-4 text-slate-400 text-xs">{c.context?.callChannel || '—'}</td>
              <td className="py-2 pr-4 font-mono tabular-nums" style={{ color: COLOR_HEX[c.risk.color] }}>
                {c.risk.score}
              </td>
              <td className="py-2">
                <span
                  className="text-xs px-2 py-0.5 rounded-full border"
                  style={{ color: COLOR_HEX[c.risk.color], borderColor: `${COLOR_HEX[c.risk.color]}55` }}
                >
                  {c.risk.level}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
