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

# Meta data to be displayed on PyPI
author          = 'Jing (Roy) Yang'
author_email    = 'roy.j.yang@qut.edu.au'
description     = 'Python toolkit for organizational model mining'
url             = 'https://github.com/roy-jingyang/OrdinoR'
project_url     = {
    'Documentation': 'https://ordinor.readthedocs.io',
    'Source': 'https://github.com/roy-jingyang/OrdinoR',
    'Tracker': 'https://github.com/roy-jingyang/OrdinoR/issues'
}
classifier      = [
    'Development Status :: 4 - Beta',
    'Programming Language :: Python',
    'License :: OSI Approved :: GNU GPLv3',
    'Operating System :: OS Independent'
]

# Package information
name                = 'ordinor'
version             = '0.1.0rc10'
python_requires     = '>=3.7, <=3.8'

packages            = [
    'ordinor',
    'ordinor.analysis',
    'ordinor.conformance',
    'ordinor.execution_context',
    'ordinor.io',
    'ordinor.org_model_miner',
    'ordinor.social_network_miner',
    'ordinor.utils',
]

def parse_requirements_file(filename):
    with open(filename) as f:
        requires = [l.strip() for l in f.readlines() if not l.startswith("#")]
    return requires

install_requires    = []
extras_require      = {
    "default": parse_requirements_file('./requirements.txt')
}


if __name__ == '__main__':
    setuptools.setup(
        author=author,
        author_email=author_email,
        description=description,
        url=url,
        project_url=project_url,
        classifier=classifier,

        name=name,
        version=version,
        python_requires=python_requires,
        install_requires=install_requires,
        extras_require=extras_require,

        #TODO
    )
