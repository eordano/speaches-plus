import { NurClient } from 'nur-client';
import type { LayoutPageResult } from './generated/LayoutPageResult';

// Studio bases historically include the /v1 suffix; NurClient wants the server
// origin (its paths carry /v1), so strip the suffix before constructing.
const originOf = (base?: string | null): string =>
  (base || '').trim().replace(/\/+$/, '').replace(/\/v1$/, '');

const clients = new Map<string, NurClient>();
export function nurClientFor(base?: string | null): NurClient {
  const origin = originOf(base);
  let c = clients.get(origin);
  if (!c) {
    c = new NurClient({ baseURL: origin });
    clients.set(origin, c);
  }
  return c;
}

// /v1/ocr is not part of the nur-client surface (phase 1); keep the direct call.
export async function postOcr(fd: FormData): Promise<LayoutPageResult> {
  const r = await fetch('/v1/ocr', { method: 'POST', body: fd });
  if (!r.ok) throw new Error('ocr ' + r.status);
  return (await r.json()) as LayoutPageResult;
}
