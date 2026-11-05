"""Prefix every streamed log line; never buffer across process exit."""
import datetime
import sys
for line in sys.stdin:
    print(datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), line.rstrip("\n"), flush=True)
