#!/usr/bin/env python
from setuptools import setup

setup(
	name='looptools',
	version='1.2',
	description='Tools to monitor, extract data and have control during algorithm progress loops',
	license='GPL-3.0',
	author='Arthur Bouton',
	author_email='arthur.bouton@gadz.org',
	url='https://github.com/arthur-bouton/MachineLearning',
	py_modules=[ 'looptools' ],
	install_requires=[ 'matplotlib>=1.5.1' ]
)
