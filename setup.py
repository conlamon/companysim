"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
#with open(path.join(here, 'README.md')) as f:
#    long_description = f.read()

setup(
    name='companysim',

    version='0.2',

    description='Package to find similarity between pairs of companies based\
                 on textual information',
   # long_description=long_description,

    # The project's main homepage.
    url='https://github.com/conlamon/companysim',

    author='Connor Lamon',
    author_email='connor.lamon@gmail.com',

    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
         'Topic :: Scientific/Engineering :: Information Analysis',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='textualsimilarity NLP',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['companysim'],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy>=1.12.1',
                      'datasketch>=1.2.1',
                      'decorator>=4.0.11',
                      'pandas>=0.20.1',
                      'scipy>=0.19.0',
                      'python-dateutil>=2.6.0',
                      'pytz>=2017.2',
                      'setuptools>=18.1',
                     ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    #entry_points={
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
)