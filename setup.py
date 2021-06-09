from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='crackdect',  # Required
    version='0.1',  # Required
    description='crack detection for composite materials',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional
    url='https://github.com/mattdrvo/CrackDect',  # Optional
    author='Matthias Drvoderic',  # Optional
    author_email='matthias.drvoderic@unileoben.ac.at',  # Optional
    classifiers=[  # Optional
        'Intended Audience :: Science/Engineering',
        'Topic :: Crack Detection :: Composites :: Image Processing',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        'License :: MIT License'
    ],
    keywords='crackdetection composites imageprocessing imagestack',  # Optional
    # package_dir={'': 'crackdect'},  # Optional
    # packages=find_packages(where=''),  # Required
    packages=['crackdect'],
    python_requires='>=3.8',
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',
                      'sqlalchemy',
                      'numba',
                      'matplotlib'
                      ],
    package_data={},
)
