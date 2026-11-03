import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

RUN_TIMEOUT_SEC = 120
CXX_FLAGS = ["-std=c++20", "-O3", "-march=native", "-DNDEBUG"]


def runProcess(command: list[str], args: str, cwd: str) -> tuple[str, float]:
    """Run a command with the user's arguments and return (output, seconds)."""
    start = time.perf_counter()
    result = subprocess.run(command + shlex.split(args), capture_output=True, text=True, cwd=cwd,
                            timeout=RUN_TIMEOUT_SEC)
    elapsed = time.perf_counter() - start

    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code {result.returncode}]"
    return output, elapsed


def runPython(code: str, args: str) -> tuple[str, str]:
    """Run Python code in a temp dir and return (output, timing)."""
    if not code.strip():
        return "", "Nothing to run"

    with tempfile.TemporaryDirectory() as workDir:
        src = os.path.join(workDir, "main.py")
        with open(src, "w") as f:
            f.write(code)
        try:
            output, elapsed = runProcess([sys.executable, src], args, workDir)
        except subprocess.TimeoutExpired:
            return "", f"Timed out after {RUN_TIMEOUT_SEC}s"
    return output, f"Run: {elapsed:.3f}s"


def runCpp(code: str, args: str) -> tuple[str, str]:
    """Compile and run C++ code in a temp dir and return (output, timing)."""
    if not code.strip():
        return "", "Nothing to run"
    if not shutil.which("c++"):
        return "'c++' is not installed, can't compile C++", ""

    with tempfile.TemporaryDirectory() as workDir:
        src = os.path.join(workDir, "main.cpp")
        binary = os.path.join(workDir, "main")
        with open(src, "w") as f:
            f.write(code)

        try:
            start = time.perf_counter()
            result = subprocess.run(["c++", *CXX_FLAGS, src, "-o", binary], capture_output=True, text=True,
                                    timeout=RUN_TIMEOUT_SEC)
            compileTime = f"Compile: {time.perf_counter() - start:.3f}s"
            if result.returncode != 0:
                return f"Compilation failed:\n{result.stderr}", compileTime

            output, elapsed = runProcess([binary], args, workDir)
        except subprocess.TimeoutExpired:
            return "", f"Timed out after {RUN_TIMEOUT_SEC}s"
    return output, f"{compileTime} · Run: {elapsed:.3f}s"
