from setuptools import setup, find_packages

setup(
    name='rtide',  
    version='1.0.0', 
    author='Thomas Monahan',
    author_email='thomas.monahan@eng.ox.ac.uk',
    description='RTide: A python implementation of the ML Response Framework for Tidal Analyis and Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thomasmonahan/RTide',  
    packages=find_packages(),
    license='Business Source License 1.1',
    install_requires=[
        'numpy',
        'pandas',
        'tensorflow',
        'utide',
        'scipy',
        'skyfield',
        'shap',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License', 
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.6',
)
