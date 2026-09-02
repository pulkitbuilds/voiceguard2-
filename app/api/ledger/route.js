import { NextResponse } from 'next/server';

const {
  getLedger,
  validateLedger,
} = require('../../../lib/store');

export async function GET() {
  try {
    const ledger = getLedger();
    const validation = validateLedger();

    return NextResponse.json({
      ok: true,
      valid: validation.valid,
      length: ledger.length,
      brokenAtIndex: validation.brokenAtIndex,
      blocks: ledger,
    });
  } catch (error) {
    console.error('Ledger API error:', error);

    return NextResponse.json(
      {
        ok: false,
        error: error.message || 'Failed to read ledger',
      },
      { status: 500 }
    );
  }
}