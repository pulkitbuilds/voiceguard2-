import { NextResponse } from 'next/server';
import { listCalls } from '../../../lib/store';

export const runtime = 'nodejs';

export async function GET(req) {
  const { searchParams } = new URL(req.url);
  const limit = Math.min(100, Number(searchParams.get('limit')) || 25);
  return NextResponse.json({ ok: true, calls: listCalls(limit) });
}
