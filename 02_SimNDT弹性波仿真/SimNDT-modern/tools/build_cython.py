from __future__ import annotations

import pathlib
import subprocess
import sys
import sysconfig

import numpy
import ziglang


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    backend_dir = root / "src" / "simndt_modern" / "backends"
    pyx = backend_dir / "_cython_core.pyx"
    generated_c = backend_dir / "_cython_core.c"
    extension = backend_dir / ("_cython_core" + sysconfig.get_config_var("EXT_SUFFIX"))
    zig = pathlib.Path(ziglang.__file__).resolve().parent / "zig.exe"
    python_include = pathlib.Path(sysconfig.get_paths()["include"])
    python_libs = pathlib.Path(sys.base_prefix) / "libs"

    subprocess.run(
        [sys.executable, "-m", "cython", "-3", "--fast-fail", str(pyx), "-o", str(generated_c)],
        check=True,
    )
    command = [
        str(zig), "cc", "-shared", "-O3", "-DNDEBUG", "-DMS_WIN64",
        f"-I{python_include}", f"-I{numpy.get_include()}", f"-L{python_libs}",
        "-lpython312", "-Wno-unused-function", "-Wno-deprecated-declarations",
        "-o", str(extension), str(generated_c),
    ]
    print("Compiling:", " ".join(command))
    subprocess.run(command, check=True)
    print("Built:", extension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

