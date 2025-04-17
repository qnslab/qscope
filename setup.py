# -*- coding: utf-8 -*-
import setuptools
import pathlib
import site
import sys

# odd bug with develop (editable) installs, see: https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

required = [
    "setuptools-git",  # see link at bottom
    "numpy",
    "matplotlib>=3.4.0",
    "matplotlib-scalebar>=0.7.2",
    "simplejson>= 3.19.2",
    "pyvisa",
    "pyvisa_py",
    "zeroconf",  # for pyvisa_py autodiscovery...
    "pyusb",
    "pylablib>=1.4.2",
    "mashumaro[msgpack]",
    "pytest_asyncio>=0.24.0",
    "numba",
    "PyQt6",
    "pyzmq",
    "loguru",
    "yagmail[all]",
    "reportlab",
    "cmap",
    "rich>=13.0.0",
    "setproctitle",
    "click>=8.0.0",
    "click-option-group",
    "psutil>=6.1.0",
    "scipy",
    "tqdm",
]

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version
version = {}
with open("src/qscope/_version.py", "r") as f:
    exec(f.read(), version)

if __name__ == "__main__":
    setuptools.setup(
        name="qscope",
        version=version["__version__"],
        author="David Broadway and Sam Scholten",
        author_email="broadwayphysics@gmail.com",
        description="Quantum Diamond Microscope (QDM) control software.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/DavidBroadway/qscope",
        keywords=[
            "NV",
            "QDM",
            "Diamond",
            "Quantum",
            "Quantum Sensing",
        ],
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Development Status :: 2 - Pre-Alpha",
        ],
        license="MIT",
        package_dir={"": "src"},
        packages=setuptools.find_packages(
            where="src",
            exclude=["*.test", "*.test.*", "test.*", "test", "test", "test_*"],
        ),
        entry_points={
            "console_scripts": [
                "qscope=qscope.cli:cli",
            ],
        },
        install_requires=required,
        python_requires=">= 3.11",
        package_data={"": ["*.md", "*.json"]},
        setup_requires=["wheel"],  # force install of wheel first
    )
# https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
