from setuptools import find_packages
from setuptools import setup

setup(name='META_DOMAIN_TRANSFER',
      install_requires=['pyyaml', 'tensorboardX',
                        'easydict', 'matplotlib',
                        'scipy', 'scikit-image',
                        'future', 'setuptools',
                        'tqdm', 'cffi'],
      packages=find_packages())
