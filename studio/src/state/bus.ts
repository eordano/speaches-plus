export interface BusEvent {
  t: number;
  kind: string;
  data: Record<string, unknown>;
}
export type BusListener = (ev: BusEvent) => void;

class EventBus {
  log: BusEvent[] = [];
  private subs = new Map<string, BusListener[]>();

  on(kind: string, fn: BusListener): () => void {
    const arr = this.subs.get(kind) || [];
    arr.push(fn);
    this.subs.set(kind, arr);
    return () => {
      const a = this.subs.get(kind) || [];
      const i = a.indexOf(fn);
      if (i >= 0) a.splice(i, 1);
    };
  }

  emit(kind: string, data?: Record<string, unknown>): BusEvent {
    const ev: BusEvent = { t: Date.now(), kind, data: data || {} };
    this.log.push(ev);
    if (this.log.length > 2000) this.log.splice(0, this.log.length - 2000);
    (this.subs.get(kind) || []).slice().forEach(f => { try { f(ev); } catch {  } });
    (this.subs.get('*') || []).slice().forEach(f => { try { f(ev); } catch {  } });
    return ev;
  }
}

export const bus = new EventBus();
