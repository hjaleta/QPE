from setuptools import setup, find_packages
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

setup(
    name='qpe',
    packages=find_packages(where='src')
    description="Quantum Phase Estimation in Qiskit",
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=INSTALL_PACKAGES,
    version='0.0.2',
    url='https://github.com/hjaleta/QPE',
    author='Hjalmar Lindstedt, Riccardo Conte, Miguel Carrera Belo
    author_email='hjaleta@gmail.com',
    keywords=['quantum-computing', 'qiskit', 'quantum-phase-estimation'],
    # tests_require=[
    #     'pytest',
    #     'pytest-cov',
    #     'pytest-sugar'
    # ],
    # package_data={
    #     # include json and pkl files
    #     '': ['*.json', 'models/*.pkl', 'models/*.json'],
    # },
    # include_package_data=True,
    python_requires='>=3'
)