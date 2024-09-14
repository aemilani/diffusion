from setuptools import find_packages, setup

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name='diffusion',
    version='0.1.0',
    description='An implementation of the diffusion model on the MNIST dataset.',
    author='Ali Eftekhari Milani',
    license='',
    packages=find_packages(include=['diffusion']),
    install_requires=REQUIREMENTS,
    extras_require={
        'interactive': ['jupyter', 'matplotlib']
        },
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest']
)