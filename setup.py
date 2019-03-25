#!/usr/bin/env python

from setuptools import setup, find_packages


import re
VERSIONFILE="cechmate/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('README.md') as f:
    long_description = f.read()

setup(name='cechmate',
      version=verstr,
      description='Custom filtration constructors for Python',
      long_description=long_description,
      long_description_content_type="text/markdown",	
      author='Christopher Tralie, Nathaniel Saul',
      author_email='chris.tralie@gmail.com, nat@saulgill.com',
      url='https://cechmate.scikit-tda.org',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'phat',
        'persim'
      ],
      extras_require={ # use `pip install -e ".[testing]"`
        'testing': [
          'pytest-cov',
          'mock',
          'kmapper',
          'networkx',
        ],
        'docs': [ # `pip install -e ".[docs]"`
          'sktda_docs_config'
        ]
      },
      python_requires='>=3.4',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='persistent homology, persistence images, persistence diagrams, topology data analysis, algebraic topology, unsupervised learning, filtrations, Cech, Alpha, Rips'
     )
