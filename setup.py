#  Copyright (c) 2021. Lucas G. S. Jeub

from setuptools import setup, find_packages

setup(
    name='local2global_embedding',
    description='',
    url='https://github.com/LJeub/Local2Global_embedding.git',
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    author='Lucas G. S. Jeub',
    python_requires='>=3.5',
    packages=find_packages(),

    install_requires=[
        'matplotlib',
        'networkx',
        'python-louvain',
        'torch',
        'torch-geometric >= 1.7',
        'torch-scatter',
        'torch-sparse',
        'torch-cluster',
        'torch-spline-conv',
        'scikit-learn',
        'pymetis',
        'local2global @ git+https://github.com/LJeub/Local2Global.git@master',
        'filelock',
        'docstring-parser',
        'tqdm >= 4.62'
    ],
)
