from setuptools import setup, find_packages
setup(
        name = 'atomap',
        packages = [
            'atomap',
            'atomap.tests',
            'atomap.external',
            ],
        version = '0.0.1a1',
        description = 'Library for analysing atomic resolution images',
        author = 'Magnus Nord',
        author_email = 'magnunor@gmail.com',
        license = 'GPL v3',
        url = 'http://atomap.org/',
        download_url = 'https://gitlab.com/atomap/atomap/repository/archive.tar?ref=0.0.1a1',
        keywords = [
            'STEM',
            'data analysis'
            'microscopy',
            ],
        install_requires = [
            'scipy',
            'numpy>=1.10',
            'h5py',
            'ipython>=2.0',
            'matplotlib>=1.2',
            'hyperspy>=1.1.1',
            ],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            ],
)