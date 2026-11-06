"""Run a command, sampling its resident memory every 2 s; kill it if RSS exceeds CAP_GB. Prints the peak."""
import subprocess
import sys
import time

import psutil

cap = float(sys.argv[1])
cmd = sys.argv[2:]
p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
proc = psutil.Process(p.pid)
peak = 0.0
while p.poll() is None:
    try:                                   # the venv launcher spawns the real interpreter: sum the whole tree
        rss = sum(q.memory_info().rss for q in [proc, *proc.children(recursive=True)]) / 2**30
    except psutil.NoSuchProcess:
        break
    peak = max(peak, rss)
    if rss > cap:
        # kill the TREE, children first: a venv launcher is the direct child and the real interpreter is ITS child,
        # so p.kill() alone stopped the launcher and left the process holding the memory running (adversary, 10-14)
        for q in [*proc.children(recursive=True), proc]:
            try:
                q.kill()
            except psutil.NoSuchProcess:
                pass
        print(f"KILLED: rss {rss:.2f} GB > cap {cap} GB", flush=True)
        break
    time.sleep(2)
print(f"peak rss {peak:.2f} GB, exit {p.wait()}", flush=True)
