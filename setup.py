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

setuptools.setup(
    name='orgminer',
    version='0.0.1b3',

    namespace_packages=['orgminer'],
    package_dir={'': '.'},
    packages=setuptools.find_namespace_packages(
        where='.', include=['orgminer.*']),

    # meta data to display on PyPI
    author='Jing Yang (Roy)',
    author_email='roy.j.yang@qut.edu.au',
    description='Python tookit for Process Mining on the organizational perspective',
    url='https://github.com/roy-jingyang/OrgMiner',
    project_url={
        'Documentation': 'https://orgminer.readthedocs.io',
        'Source': 'https://github.com/roy-jingyang/OrgMiner',
        'Tracker': 'https://github.com/roy-jingyang/OrgMiner/issues'
    },
    classifier=[
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: GNU GPLv3',
        'Operating System :: OS Independent'
    ],

    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.17.2',
        'scipy>=1.3.1',
        'scikit-learn>=0.21.3',
        'scikit-fuzzy>=0.4.1',
        'pandas>=0.25.3',
        'python-igraph>=0.8.0',
        'networkx>=2.4',
        'pm4py>=1.2.4',
        'Deprecated>=1.2.6',
    ],
    extras_require={
        'arya': [
            'Flask', 
            'flask-cors',
            'flask-bootstrap',
            'flask-wtf',
            'flask-session'
        ]
    }
)
