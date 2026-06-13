import os
import io
import builtins
from datetime import datetime

os.environ["FLAGS_use_virtual_memory_auto_growth"] = "1"


def print_log(*args, file=None, **kwargs):
    now = datetime.now().strftime("%H:%M:%S")
    buf = io.StringIO()
    _print(*args, **kwargs, file=buf)
    text = buf.getvalue()

    _print(end=f"\033[1;32m{now}\033[0m {text}", file=file, flush=True)
    _logfile.write(f"{now} {text}")
    _logfile.flush()


_logfile = open("./workerlog.0", "a")
_print = builtins.print
builtins.print = print_log
