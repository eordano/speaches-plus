#!/usr/bin/env python3
"""Exclusive-GPU exec wrapper for platforms without flock(1) (darwin).

Usage: gpu-flock.py LOCKFILE CMD [ARGS...]

Takes a blocking BSD flock on LOCKFILE, then execs CMD. The lock lives on the
open file description and the fd is marked inheritable, so it survives the
exec and is held until the command exits -- identical semantics to
`flock LOCKFILE CMD` on linux. nvk.sh probe mode uses this as its fallback;
any non-cargo GPU consumer (training smokes, python benches) can use it
directly so every perf measurement on a box serializes on the same file:

    rust/scripts/gpu-flock.py /tmp/claude-501/nvk-gpu0.lock python bench.py

On acquire the holder's identity (pid, start time, argv) is written into the
lock file, because a holder is invisible to `ps | grep gpu-flock`: it has
exec'd into its command, so every process that LOOKS like gpu-flock is a
waiter. Waiters read that identity and re-announce who blocks them once a
minute with elapsed time, so a wedged holder is diagnosed from the waiting
run's own log instead of lsof forensics.
"""

import fcntl
import os
import sys
import time


def main() -> None:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: gpu-flock.py LOCKFILE CMD [ARGS...]\n")
        sys.exit(2)
    lock_path = sys.argv[1]
    cmd = sys.argv[2:]
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    t0 = time.time()
    announced = 0.0
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            now = time.time()
            if now - announced >= 60.0 or announced == 0.0:
                announced = now
                holder = "unknown holder (pre-identity lock or truncated write)"
                try:
                    with open(lock_path, "r") as f:
                        line = f.read(512).strip()
                    if line:
                        holder = line
                except OSError:
                    pass
                sys.stderr.write(
                    f"gpu-flock: waited {int(now - t0)}s for {lock_path}; "
                    f"held by: {holder}\n"
                )
            time.sleep(2.0)
    ident = f"pid={os.getpid()} since={time.strftime('%H:%M:%S')} cmd={' '.join(cmd)[:300]}\n"
    try:
        os.ftruncate(fd, 0)
        os.pwrite(fd, ident.encode(), 0)
    except OSError:
        pass
    os.set_inheritable(fd, True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
