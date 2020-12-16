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
    
setup(
    name='tsia',
    version='0.1.2',
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
    classifiers=CLASSIFIERS
)