from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['scipy>=0.19','keras>=2.0.6','h5py>=2.7']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='For applications which use tflearn and h5py packed data.')

