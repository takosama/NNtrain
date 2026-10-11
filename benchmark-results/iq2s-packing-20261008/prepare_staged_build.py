"""Export the current Git index into an isolated release verification tree.

This leaves the index unchanged and never replaces an existing export.
"""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
files = [p for p in files if p.startswith(("NNtrain.", "tools/iq2s-production-bench/", "tools/iq2s-packed-decode-bench/"))
         or p == "Directory.Build.targets"]
build = Path(__file__).parent / "staged-build"
build.mkdir(exist_ok=False)
subprocess.run(["git", "checkout-index", "--prefix=" + build.resolve().as_posix() + "/", "-z", "--stdin"],
               input=("\0".join(files) + "\0").encode(), cwd=root, check=True)
print(f"Exported {len(files)} staged source files to {build}")
