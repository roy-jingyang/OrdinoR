"""Setup script for orgminer

Install orgminer with

python setup.py install
"""

import os
import sys
import setuptools

if sys.argv[-1] == "setup.py":
    print('To start installation, run "python setup.py install"')
    print()

if sys.version_info[:2] < (3, 6): # python version >= 3.6
    error = (
        'OrgMiner requires Python 3.6 or later ' +
        '({0[0]}.{0[1]} detected).'.format(sys.version_info[:2])
    )
    sys.stderr.write(error + '\n')
    sys.exit(1)

# load GitHub repository README.md as long description
with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='orgminer',
    version='0.0.1a20200213-1530',

    namespace_packages=['orgminer'],
    package_dir={'': '.'},
    packages=setuptools.find_namespace_packages(
        where='.', include=['orgminer.*']),

    # meta data to display on PyPI
    author='Jing Yang (Roy)',
    author_email='roy.j.yang@qut.edu.au',
    description='Process Mining on the organizational perspective',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/roy-jingyang/OrgMiner',
    classifier=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU GPLv3',
        'Operating System :: OS Independent'
    ],

    install_requires=[
        'numpy>=1.17.2',
        'scipy>=1.3.1',
        'scikit-learn>=0.21.3',
        'scikit-fuzzy>=0.4.1',
        'pandas>=0.25.3',
        'networkx>=2.4',
        'pm4py>=1.2.4',
        'pygraphviz',
        'Deprecated>=1.2.6', # aka 'deprecated' anaconda cloud
    ],
    python_requires='>=3.6',
)
