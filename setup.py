# -*- coding: utf-8 -*-
"""Modern setup module for ndd."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from packaging.version import Version
from setuptools import Command, find_packages, setup
from setuptools.command.build_ext import build_ext

NAME = "ndd"
NUMPY_MIN_VERSION = "1.23"
PLATFORM = platform.system()
BASE_DIR = Path(__file__).parent.resolve()


def is_project_package(path: Path) -> bool:
    """Directory looks like the base package for the project."""
    return (path / "__init__.py").is_file() and (path / "package.py").is_file()


def get_package_path(path: Path = BASE_DIR) -> Path:
    """Return the top-level project package path."""
    for x in path.iterdir():
        if x.is_dir() and is_project_package(x):
            return x
    raise ImportError("Can't find the top package dir for the project.")


def get_long_description(readme: Path) -> str:
    """Get the long description from README."""
    return readme.read_text(encoding="utf-8")


def get_numpy_status() -> dict[str, str | bool]:
    """Return NumPy installation status."""
    status: dict[str, str | bool] = {}
    try:
        import numpy

        numpy_version = numpy.__version__
        status["up_to_date"] = Version(numpy_version) >= Version(NUMPY_MIN_VERSION)
        status["version"] = numpy_version
    except ImportError:
        status["up_to_date"] = False
        status["version"] = ""
    return status


class F2PYBuildExt(build_ext):
    """
    Build the Fortran extension using `python -m numpy.f2py`.

    This avoids numpy.distutils and works with modern setuptools-based builds.
    """

    description = "Build Fortran extension with NumPy F2PY"

    def run(self) -> None:
        self._check_numpy()
        self.build_f2py_extension()
        super().run()

    def _check_numpy(self) -> None:
        status = get_numpy_status()
        req = f"{NAME} requires NumPy >= {NUMPY_MIN_VERSION}"
        if not status["up_to_date"]:
            if status["version"]:
                raise RuntimeError(
                    f"Your installation of NumPy {status['version']} is out-of-date. {req}"
                )
            raise RuntimeError(f"NumPy is not installed. {req}")

    def build_f2py_extension(self) -> None:
        import numpy

        ext_fullname = "ndd.fnsb"
        modulename = "fnsb"

        build_lib = Path(self.build_lib).resolve()
        target_pkg_dir = build_lib / "ndd"
        target_pkg_dir.mkdir(parents=True, exist_ok=True)

        sources = [
            str(BASE_DIR / "ndd" / "nsb.pyf"),
            str(BASE_DIR / "ndd" / "exts" / "gamma.f90"),
            str(BASE_DIR / "ndd" / "exts" / "quad.f90"),
            str(BASE_DIR / "ndd" / "exts" / "estimators.f90"),
        ]

        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "-m",
            modulename,
            *sources,
        ]

        env = os.environ.copy()

        # Ensure the compiled extension lands in the package build directory.
        cwd = target_pkg_dir

        self.announce(
            "Building Fortran extension with command:\n  " + " ".join(cmd),
            level=3,
        )

        subprocess.check_call(cmd, cwd=str(cwd), env=env)

        # macOS note:
        # Old numpy.distutils passed '-undefined dynamic_lookup' on Darwin.
        # If your compiler/linker needs extra flags, export them before build, e.g.:
        #   LDFLAGS="-undefined dynamic_lookup" python -m build

        produced = list(target_pkg_dir.glob("fnsb*.so")) + list(target_pkg_dir.glob("fnsb*.pyd"))
        if not produced:
            raise RuntimeError(
                "F2PY build finished but no compiled extension was found in "
                f"{target_pkg_dir}"
            )


class CleanCommand(Command):
    """Simple clean command."""

    user_options: list[tuple[str, str | None, str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        patterns = [
            "build",
            "dist",
            "*.egg-info",
            "ndd/*.so",
            "ndd/*.pyd",
            "ndd/*.dll",
            "ndd/*.pyc",
            "ndd/__pycache__",
        ]
        for pattern in patterns:
            for path in BASE_DIR.glob(pattern):
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)


readme_file = BASE_DIR / "README.rst"
package_path = get_package_path()
package_name = package_path.name

project_info: dict[str, str] = {}
with open(package_path / "package.py", encoding="utf-8") as fp:
    exec(fp.read(), project_info)  # noqa: S102

project_name = project_info["__title__"]
project_version = project_info["__version__"]

long_description = get_long_description(readme_file)

setup(
    name=project_name,
    version=project_version,
    description="Bayesian entropy estimation from discrete data",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author=project_info.get("__author__"),
    author_email=project_info.get("__email__"),
    url=project_info.get("__url__"),
    packages=find_packages(include=["ndd", "ndd.*"]),
    include_package_data=True,
    package_data={"": ["LICENSE.txt", "README.rst", "requirements.txt"]},
    entry_points={"console_scripts": ["ndd=ndd.entry:main"]},
    python_requires=">=3.9",
    setup_requires=[],
    install_requires=[f"numpy>={NUMPY_MIN_VERSION}"],
    extras_require={"test": ["pytest"]},
    license="BSD 3-Clause",
    cmdclass={
        "build_ext": F2PYBuildExt,
        "clean": CleanCommand,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)