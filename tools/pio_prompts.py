"""PlatformIO pre-build hook: regenerate include/prompts.h from src/main.cpp.

Editing the prompt strings at the top of src/main.cpp and pressing Upload must
be enough. The firmware has a decode-only tokenizer and cannot tokenize text at
runtime, so the strings have to be converted on the PC before compiling -- this
hook makes that step invisible.

Registered in platformio.ini via:
    extra_scripts = pre:tools/pio_prompts.py

The hook is fail-soft on purpose: if the generator cannot run (e.g. the
tokenizer model or sentencepiece is missing), it prints a warning and leaves the
existing header in place rather than breaking the build. A stale prompt is far
less annoying than a build that will not start.
"""
import os
import subprocess
import sys

Import("env")  # noqa: F821  -- provided by PlatformIO

PROJECT = env.subst("$PROJECT_DIR")  # noqa: F821
SCRIPT = os.path.join(PROJECT, "scripts", "gen_prompts.py")
HEADER = os.path.join(PROJECT, "include", "prompts.h")


def candidates():
    """Interpreters that might have sentencepiece, best first."""
    out = [sys.executable]
    env_py = env.subst("$PYTHONEXE")  # noqa: F821
    if env_py and env_py not in out:
        out.append(env_py)
    # the interpreter used elsewhere in this project
    out.append(r"C:\Users\yancunh\AppData\Local\Microsoft\WindowsApps\python3.12.exe")
    out.append("python")
    return out


def regenerate():
    if not os.path.exists(SCRIPT):
        print("[pio_prompts] scripts/gen_prompts.py missing; keeping existing prompts.h")
        return
    if not os.path.exists(HEADER):
        print("[pio_prompts] include/prompts.h missing; it will be generated")

    for py in candidates():
        try:
            # encoding/errors are essential: without them subprocess decodes the
            # child's stdout with the system codepage (cp936 here), and the
            # generator prints Chinese prompts -> UnicodeDecodeError.
            proc = subprocess.run(
                [py, SCRIPT],
                cwd=PROJECT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=180,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError):
            continue
        if proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                if line.strip():
                    print(f"[pio_prompts] {line}")
            return
        # a missing sentencepiece shows up as ImportError on stderr; try the next
        stderr = proc.stderr or ""
        err = stderr.strip().splitlines()
        last = err[-1] if err else "unknown error"
        if "ModuleNotFoundError" in stderr or "No module named" in stderr:
            continue
        print(f"[pio_prompts] generator failed with {py}: {last}")
        break

    print("[pio_prompts] WARNING: could not regenerate prompts.h "
          "(no interpreter with sentencepiece found); keeping the existing one")


regenerate()
