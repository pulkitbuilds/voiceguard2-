'use client';

import { useEffect, useRef } from 'react';

export default function Waveform({ samples, sampleRate, height = 120 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const width = canvas.clientWidth || 600;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // background grid (oscilloscope feel)
    ctx.strokeStyle = 'rgba(94,234,212,0.06)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.strokeStyle = 'rgba(148,163,184,0.15)';
    ctx.stroke();

    if (!samples || samples.length === 0) {
      ctx.fillStyle = 'rgba(148,163,184,0.4)';
      ctx.font = '13px ui-monospace, monospace';
      ctx.fillText('no signal — record or upload a clip', 16, height / 2 + 4);
      return;
    }

    const step = Math.max(1, Math.floor(samples.length / width));
    ctx.beginPath();
    ctx.strokeStyle = '#5eead4';
    ctx.lineWidth = 1.4;
    for (let x = 0; x < width; x++) {
      const start = x * step;
      let min = 1;
      let max = -1;
      for (let i = start; i < Math.min(start + step, samples.length); i++) {
        const v = samples[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (min > max) { min = 0; max = 0; }
      const yMin = height / 2 - max * (height / 2 - 4);
      const yMax = height / 2 - min * (height / 2 - 4);
      ctx.moveTo(x, yMin);
      ctx.lineTo(x, yMax);
    }
    ctx.stroke();
  }, [samples, sampleRate, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: `${height}px` }}
      className="block rounded-md"
    />
  );
}
