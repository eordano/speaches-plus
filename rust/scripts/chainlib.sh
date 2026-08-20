#!/usr/bin/env bash

_CHAINLIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUQ="${GPUQ_BIN:-$_CHAINLIB_DIR/gpuq}"

chain_init() {
  CHAIN_DIR="${1:?chain_init needs a dir}"
  mkdir -p "$CHAIN_DIR/gates"
  CHAIN_LOG="$CHAIN_DIR/chain.log"
  CHAIN_RESULTS="$CHAIN_DIR/gates/.results.tsv"
  : > "$CHAIN_RESULTS"
  echo "CHAIN-START $(date -Is) pid=$$" >> "$CHAIN_LOG"
}

_gate_record() { # name rc dur
  echo "GATE-DONE $1 rc=$2 dur=${3}s $(date -Is)" >> "$CHAIN_LOG"
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$CHAIN_RESULTS"
}

_run_gate() { # use_gpu name timeout -- cmd...
  local use_gpu="$1" name="$2" tmo="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local glog="$CHAIN_DIR/gates/$name.log" t0=$SECONDS rc
  echo "GATE-START $name $(date -Is)" >> "$CHAIN_LOG"
  if [ "$use_gpu" = yes ]; then
    "$GPUQ" run -t "$tmo" -n "$name" -- "$@" > "$glog" 2>&1
    rc=$?
  else
    timeout --signal=TERM --kill-after=30 "$tmo" "$@" > "$glog" 2>&1
    rc=$?
  fi
  if [ "$rc" -eq 0 ] && grep -qE 'test result: FAILED|panicked at|error: test failed' "$glog"; then
    echo "GATE-FALSEGREEN $name rc=0 but log shows failure -> forcing rc=97" >> "$CHAIN_LOG"
    rc=97
  fi
  _gate_record "$name" "$rc" $((SECONDS-t0))
  return "$rc"
}

gate()      { _run_gate yes "$@"; }
gate_host() { _run_gate no  "$@"; }

gate_expect() { # name regex — assert against an already-run gate's log
  local name="$1" re="$2" glog="$CHAIN_DIR/gates/$1.log"
  if grep -qE "$re" "$glog"; then
    echo "GATE-EXPECT $name ok ($re)" >> "$CHAIN_LOG"
  else
    echo "GATE-EXPECT $name MISSING ($re)" >> "$CHAIN_LOG"
    printf '%s\t%s\t%s\n' "$name:expect" 98 0 >> "$CHAIN_RESULTS"
    return 98
  fi
}

prefetch() { # name timeout -- cmd...   downloads/copies; never holds the GPU lock
  local name="$1" tmo="$2"; shift 2
  [ "${1:-}" = "--" ] && shift
  local glog="$CHAIN_DIR/gates/$name.prefetch.log" t0=$SECONDS rc
  echo "PREFETCH-START $name $(date -Is)" >> "$CHAIN_LOG"
  timeout "$tmo" "$@" > "$glog" 2>&1; rc=$?
  echo "PREFETCH-DONE $name rc=$rc dur=$((SECONDS-t0))s" >> "$CHAIN_LOG"
  return "$rc"
}

chain_wait_marker() { # file regex timeout_s — bounded cross-chain wait
  local f="$1" re="$2" tmo="$3" t0=$SECONDS
  echo "CHAIN-WAIT $(date -Is) file=$f re=$re" >> "$CHAIN_LOG"
  until [ -f "$f" ] && grep -qE "$re" "$f"; do
    [ $((SECONDS-t0)) -ge "$tmo" ] && { echo "CHAIN-WAIT-TIMEOUT $(date -Is)" >> "$CHAIN_LOG"; return 1; }
    sleep 20
  done
  echo "CHAIN-WAIT-OK $(date -Is) waited=$((SECONDS-t0))s" >> "$CHAIN_LOG"
}

killpat() { # pattern — SIGTERM every match except ourselves and our ancestors.
  local pat="${1:?killpat needs a pattern}" self=$$ pids p keep
  keep=" $self $PPID "
  p=$PPID
  while [ "$p" -gt 1 ] 2>/dev/null; do
    p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break
    keep="$keep$p "
  done
  pids=$(pgrep -f -- "$pat")
  local killed=0
  for p in $pids; do
    case "$keep" in *" $p "*) continue ;; esac
    kill -TERM "$p" 2>/dev/null && killed=$((killed+1))
  done
  echo "KILLPAT pat=$pat killed=$killed $(date -Is)" >> "${CHAIN_LOG:-/dev/null}"
  [ "$killed" -gt 0 ]
}

chain_summary() {
  echo "CHAIN-EXIT $(date -Is)" >> "$CHAIN_LOG"
  local red=0
  echo "gate	rc	dur_s"
  while IFS=$'\t' read -r n rc d; do
    echo "$n	$rc	$d"
    [ "$rc" -ne 0 ] && red=$((red+1))
  done < "$CHAIN_RESULTS"
  echo "-- $red red"
  [ "$red" -eq 0 ]
}
