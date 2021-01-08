from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()
    
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Programming Language :: Python :: 3.7'
]

INSTALL_REQUIRES = [
    'networkx>=2.5',
    'numpy>=1.17.5',
    'numba>=0.48.0',
    'python-louvain>=0.14.0',
    'pyts>=0.11.0',
]
    
setup(
    name='tsia',
    version='0.1.5',
    maintainer='Michael Hoarau',
    maintainer_email='michael.hoarau@gmail.com',
    description='A Python package for time series analysis through images',
    license='MIT',
    url='https://github.com/michaelhoarau/tsia',
    download_url='https://github.com/michaelhoarau/tsia',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    zip_safe=False,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES
)