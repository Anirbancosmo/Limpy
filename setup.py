#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 23:03:21 2020

@author: anirbanroy
"""

import sys 
try:
	from setuptools import setup
	have_setuptools = True 
except ImportError:
	from distutils.core import setup 
	have_setuptools = False

setup_kwargs = {
'name': 'limpy', 
'version': '1.0', 
'description': 'Line Intensity Mapping code in PYthon', 
'author': 'Anirban Roy', 
'author_email': 'anirbanroy.personal@gmail.com', 
'packages':['limpy'],
'zip_safe': False, 

'install_requires':['camb', 'colossus', 'astropy']
}

if __name__ == '__main__': 
	setup(**setup_kwargs)
