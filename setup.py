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
    version='0.1.0rc6',
    python_requires='>=3.7, <=3.8',

    package_dir={'': 'src'},
    packages=setuptools.find_packages(),

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
        'numpy',
        'scipy',
        'scikit-learn',
        'scikit-fuzzy',
        'pandas',
        'networkx',
        'pm4py>=1.2.4',

        # webapp requirements
        'Flask', 
        'flask-cors',
        'flask-bootstrap',
        'flask-wtf',
        'flask-session',
    ],
)
