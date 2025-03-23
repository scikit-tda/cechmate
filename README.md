[![PyPI version](https://badge.fury.io/py/cechmate.svg)](https://badge.fury.io/py/cechmate)
![PyPI - Downloads](https://img.shields.io/pypi/dm/cechmate)
[![image](https://img.shields.io/pypi/pyversions/cechmate.svg)](https://pypi.python.org/pypi/cechmate)
[![codecov](https://codecov.io/gh/scikit-tda/cechmate/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-tda/cechmate)
[![Build and Upload Python Package](https://github.com/scikit-tda/cechmate/actions/workflows/build_and_deploy.yml/badge.svg)](https://github.com/scikit-tda/cechmate/actions/workflows/build_and_deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This library provides easy to use constructors for custom filtrations that are suitable for use with [Gudhi](https://gudhi.inria.fr/python/latest/).
Gudhi provides an easy to install python interface for fast persistence computation.
Currently, we support construction of Alpha, Rips, and Cech filtrations, and provide a simple interface for Gudhi.

<!--
[Phat](https://github.com/xoltar/phat).
Phat currently provides a clean interface for persistence reduction algorithms for boundary matrices.
This tool helps bridge the gap between data and boundary matrices.
-->

If you have a particular filtration you would like implemented, please feel free to reach out and we can work on helping with implementation and integration, so others can use it.

# Setup

The dependencies of this project are listed in the `dependencies` table in `pyproject.toml`. For completeness, they are

- Matplotlib
- Numpy
- Scipy

The latest version of Cechmate can be found on Pypi and installed with pip:

```
pip install cechmate
```

# Contributions

We welcome contributions of all shapes and sizes. There are lots of
opportunities for potential projects, so please get in touch if you would like
to help out. Everything from an implementation of your favorite distance,
notebooks, examples, and documentation are all equally valuable so please don't
feel you can't contribute.

To contribute please fork the project make your changes and submit a pull
request. We will do our best to work through any issues with you and get your
code merged into the main branch.

## Documentation

Check out complete documentation at [cechmate.scikit-tda.org](https://cechmate.scikit-tda.org/)
