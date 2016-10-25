#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

import io
import os
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import relpath
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()

tests_require=['nose']

setup(
    name='hdhp',
    version='0.1.0',
    license='ISC',
    description='An implementation of the hierarchical Dirichlet-Hawkes process.',
    long_description='%s' % (read('README.rst')),
    author='Charalampos Mavroforakis',
    author_email='cmav@bu.edu',
    url='https://github.com/Networks-Learning/hdhp.py',
    packages=['hdhp'],
    include_package_data=True,
    zip_safe=False,
    test_suite='nose.collector',
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Natural Language :: English'
    ],
    keywords=['learning activity modeling',
              'user interest tracking',
              'continuous-time data clustering',
              'point process',
              'topic modeling',
              'machine learning'
    ],
    install_requires=[
        'numpy==1.11.2',
        'scipy==0.18.1',
        'seaborn==0.7.1',
        'matplotlib==1.5.3'
    ],
    extras_require={
        'tests': tests_require,
    },
)
