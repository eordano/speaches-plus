#!/usr/bin/env python3
"""Translate the shipped capture-ownership and eviction decisions into Shipped.v; refuse any shape it does not recognise."""
import os
import re
import sys
from pathlib import Path

root = Path(os.environ['PROOF_ROOT_CODE'])


def read(rel):
    return (root / rel).read_text()


def body(rel, header, closer=r'\n}\n'):
    m = re.search(re.escape(header) + r'.*?' + closer, read(rel), re.S)
    if not m:
        sys.exit(f'{rel}: no `{header}`: review Exclusion.v/Evict.v against it, then update bind.py')
    return [row.strip() for row in m.group().splitlines()[1:-1] if row.strip()]


def expect(rel, header, rows, closer=r'\n}\n'):
    got = body(rel, header, closer)
    if got != rows:
        sys.exit(f'{rel}: `{header}` is outside the translated grammar:\n  ' + '\n  '.join(got)
                 + '\nreview the model against it, then update bind.py')


def count(rel, pattern):
    return len(re.findall(pattern, read(rel), re.M))


def once(rel, pattern):
    found = re.findall(pattern, read(rel), re.M)
    if len(found) != 1:
        sys.exit(f'{rel}: expected exactly one match of {pattern!r}, found {len(found)}')
    return found[0]


CX = 'rust/crates/nv-kernels/src/capture_exclusion.rs'
holds_op = once(CX, r'^    s\.owner\.is_some_and\(\|o\| o (!=|==) me\)$')
other = 'negb (Bool.eqb d me)' if holds_op == '!=' else 'Bool.eqb d me'
expect(CX, "fn own() -> MutexGuard<'static, State> {", [
    'let me = thread::current().id();', 'let mut s = state();', 'while another_thread_owns(&s, me) {',
    's = wait(s);', '}', 's.owner = Some(me);', 's'])
expect(CX, 'fn disown(release: impl FnOnce(&mut State)) {', [
    'let mut s = state();', 'release(&mut s);', 'if s.captures == 0 && s.exclusions == 0 {', 's.owner = None;', '}',
    'drop(s);', 'CHANGED.notify_all();'])
expect(CX, 'pub fn capture_in_flight() -> CaptureInFlight {', ['own().captures += 1;', 'CaptureInFlight(())'])
expect(CX, 'pub fn no_capture_in_flight() -> NoCaptureInFlight {', ['own().exclusions += 1;', 'NoCaptureInFlight(())'])
expect(CX, 'impl Drop for CaptureInFlight {', ['fn drop(&mut self) {', 'disown(|s| s.captures -= 1);', '}'])
expect(CX, 'impl Drop for NoCaptureInFlight {', ['fn drop(&mut self) {', 'disown(|s| s.exclusions -= 1);', '}'])

GR = 'rust/crates/nv-kernels/src/graph.rs'
sites = count(GR, r'^        let _capture_in_flight = crate::capture_exclusion::capture_in_flight\(\);$')
if sites != 2 or count(GR, r'capture_lock') != 0:
    sys.exit(f'{GR}: expected both capture sites to take capture_in_flight and no capture_lock beside it, '
             f'found {sites} sites and {count(GR, "capture_lock")} capture_lock mentions')

LOOP = 'rust/src/oapi/chat_engine/qwen4_exp_loop.rs'
idle_secs = once(LOOP, r'^pub\(crate\) const QWEN4_EXP_SESSION_SLOT_PARKS_AFTER_300_IDLE_SECS_BECAUSE_A_DISK_RESTORE_COSTS_ABOUT_A_SECOND: u64 = (\d+);$')
permit = count(LOOP, r'^        let Ok\(_permit\) = qwen4_exp_gate\(\)\.try_acquire\(\) else \{ return false \};$')
idle_op = once(LOOP, r'^            \.filter\(\|h\| limit\.is_none_or\(\|l\| h\.2\.elapsed\(\) (>=|>) l\)\)$')
idle = {'>=': 'l <=? elapsed', '>': 'l <? elapsed'}[idle_op]
once(LOOP, r'^        let limit = \(reason == crate::oapi::cold_tier::IDLE_REASON\)$')
ADAPTER = 'ADAPTER_SESSION_IS_HELD_SLOT_2_AGED_AND_RELEASED_BY_THE_COLD_TIER_LIKE_THE_TWO_BASE_SLOTS'
held_rows = body(LOOP, 'fn held(&self) -> Vec<(usize, u64, std::time::Instant)> {', r'\n    }\n')
if held_rows == ['let Some(m) = self.model() else { return Vec::new() };',
                 f'[0, 1, nv_models::qwen4_exp::{ADAPTER}].into_iter().filter_map(|i| m.held_session(i).map(|(b, t)| (i, b, t))).collect()']:
    adapter_held = 'true'
elif held_rows == ['let Some(m) = self.model() else { return Vec::new() };',
                   '[0, 1].into_iter().filter_map(|i| m.held_session(i).map(|(b, t)| (i, b, t))).collect()']:
    adapter_held = 'false'
else:
    sys.exit(f'{LOOP}: held() is outside the translated grammar:\n  ' + '\n  '.join(held_rows))

QX = 'rust/crates/nv-models/src/qwen4_exp.rs'
adapter_slot = once(QX, rf'^pub const {ADAPTER}: usize = (\d+);$')

CT = 'rust/src/oapi/cold_tier.rs'
expect(CT, 'pub fn idle_limit(r: &dyn Resident) -> Option<Duration> {', [
    'let global = env_u64(IDLE_SECS_ENV, DEFAULT_IDLE_SECS);', '(global > 0).then(|| {',
    'let global = Duration::from_secs(global);', 'r.idle_after().map_or(global, |own| own.min(global))', '})'])

RES = 'rust/crates/nv-models/src/qwen4_exp_residency.rs'
expect(RES, 'fn held_slot(&self, slot: usize) -> Option<&std::sync::Mutex<Option<Qwen4ExpSession>>> {', [
    f'if slot == super::{ADAPTER} {{', 'Some(&self.adapter_session)', '} else {', 'self.sessions.get(slot)', '}'],
    r'\n    }\n')
drops = count(RES, r'^        held\.extend\(self\.adapter_session\.lock\(\)\.ok\(\)\.and_then\(\|mut s\| s\.take\(\)\)\);$')
if drops > 1:
    sys.exit(f'{RES}: the adapter session is taken more than once in release_sessions_to_disk')

Path(os.environ['PROOF_BUILD'], 'Shipped.v').write_text(f'''From Stdlib Require Import Arith Bool.
From Evict Require Import Exclusion Evict.
Definition shipped : policy := Policy
  (fun o me => match o with Some d => {other} | None => false end)
  true.
Definition shipped_evict : epolicy := EPolicy
  {idle_secs}
  (fun global own => if 0 <? global then Some (match own with Some o => Nat.min o global | None => global end) else None)
  (fun limit elapsed => match limit with None => true | Some l => {idle} end)
  {'true' if permit == 1 else 'false'}
  {adapter_slot}
  {adapter_held}
  {'true' if drops == 1 else 'false'}.
''')
