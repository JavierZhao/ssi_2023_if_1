from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='classification and segmentation of images. Four type of particles (electron, photon, muon, and proton) are simulated in liquid argon medium and the 2D projections of their 3D energy deposition patterns ("trajectories") are recorded. The challenge is to develop a classifier algorithm that identify which of four types is present in an image.',
    author='Zihan Zhao, Tianqi Zhang, Wei Wei, Po-Wen Zhang',
    license='MIT',
)
