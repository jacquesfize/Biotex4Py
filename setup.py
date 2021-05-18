from setuptools import setup
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Biotex4Py',
    version='0.1',
    packages=['biotex', 'biotex.measure', 'biotex.pattern'],
    url='',
    license='MIT',
    author='Jacques Fize',
    author_email='jacques[dot]fize[at]gmail[dot]com',
    description='Python implement of Biotex, a system for Biomedical Terminology Extraction, Ranking, and Validation',
    package_data={"biotex":["resources/dataSetReference/*.txt","resources/patterns/*.csv","resources/stopWords/*.txt","resources/treetagger_spacy_mappings/*.csv"]},
    classifiers=[  # Optional
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable

            # Indicate who your project is intended for
            'Intended Audience :: Developers',

            # Pick your license as you wish
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            # These classifiers are *not* checked by 'pip install'. See instead
            # 'python_requires' below.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
    keywords='Biotex Automatic Terminology Extractor ',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
