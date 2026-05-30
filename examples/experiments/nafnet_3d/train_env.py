import os
import re
import io
import builtins
import importlib.util
from datetime import datetime

os.environ["FLAGS_use_virtual_memory_auto_growth"] = "1"

clip_path = os.path.join(
    importlib.util.find_spec("paddle").submodule_search_locations[0],
    "nn/clip.py"
)
norm_pattern = "global_norm_var = paddle.sqrt(global_norm_var)"
try:
    with open(clip_path + ".bak") as f:
        content = f.read()
except FileNotFoundError:
    with open(clip_path) as fi, open(clip_path + ".bak", "w") as fo:
        content = fi.read()
        fo.write(content)
content, count = re.subn(
    re.escape(norm_pattern),
    norm_pattern + "; paddle._global_norm = global_norm_var.item()",
    content, count=1,
)
if count > 0:
    print("inserted global_norm_var logging")
    with open(clip_path, "w") as f:
        f.write(content)


def print_log(*args, file=None, **kwargs):
    now = datetime.now().strftime("%H:%M:%S")
    buf = io.StringIO()
    _print(*args, **kwargs, file=buf)
    text = buf.getvalue()

    _print(f"\033[1;32m{now}\033[0m {text}", end="", file=file, flush=True)
    _logfile.write(f"{now} {text}")
    _logfile.flush()


_logfile = open("./workerlog.0", "a")
_print = builtins.print
builtins.print = print_log
