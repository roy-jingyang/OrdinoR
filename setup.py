"""Setup script for OrdinoR

Install OrdinoR with

python setup.py install
"""

import sys
import setuptools

if sys.argv[-1] == "setup.py":
    print('To start installation, run \n\tpython setup.py install\n')

py_ver = sys.version_info[:2]

if py_ver < (3, 7) or py_ver > (3, 8):
    sys.stderr.write(
        f"""
        OrdinoR requires Python 3.7, or 3.8.
        You have Python {py_ver[0]}.{py_ver[1]}.\n
        """
    )
    sys.exit(1)

setuptools.setup(
    name='ordinor',
    version='0.1.0',
    python_requires='>=3.7, <=3.8',

    namespace_packages=['ordinor'],
    package_dir={'': '.'},
    packages=setuptools.find_namespace_packages(
        where='.', include=['ordinor.*']
    ),

    # meta data to display on PyPI
    author='Jing (Roy) Yang',
    author_email='roy.j.yang@qut.edu.au',
    description='Python toolkit for organizational model mining',
    url='https://github.com/roy-jingyang/OrdinoR',
    project_url={
        'Documentation': 'https://ordinor.readthedocs.io',
        'Source': 'https://github.com/roy-jingyang/OrdinoR',
        'Tracker': 'https://github.com/roy-jingyang/OrdinoR/issues'
    },
    classifier=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
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
        'Deprecated>=1.2.6',

        # webapp requirements
        'Flask', 
        'flask-cors',
        'flask-bootstrap',
        'flask-wtf',
        'flask-session',
    ],
)
