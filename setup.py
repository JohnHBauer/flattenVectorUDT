from setuptools import setup

with open("README.md", "r") as fh:
   long_description = fh.read()

setup(
   name='flattenVectorUDT',
   version='0.1.0',
   author='john.h.bauer',
   author_email='john.h.bauer@gmail.com',
   packages=['flattenVectorUDT'],
   scripts=[],
   url='',
   license='LICENSE.txt',
   description='Flatten PySpark UserDefinedType vectors into a bunch of columns',
   long_description=long_description,
   install_requires=[],
)