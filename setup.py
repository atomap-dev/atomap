from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
        name = 'atomap',
        packages = [
            'atomap',
            'atomap.tests',
            'atomap.tests.datasets',
            'atomap.external',
            'atomap.example_data',
            ],
        version = '0.3.0',
        description = 'Library for analysing atomic resolution images',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author = 'Magnus Nord',
        author_email = 'magnunor@gmail.com',
        license = 'GPL v3',
        url = 'https://atomap.org/',
        download_url = 'https://gitlab.com/atomap/atomap/repository/archive.tar?ref=0.3.0',
        keywords = [
            'STEM',
            'data analysis',
            'microscopy',
            ],
        install_requires = [
            'scipy>=1.4.0',
            'numpy>=1.13',
            'h5py',
            'matplotlib>=3.1.0',
            'scikit-learn',
            'scikit-image>=0.17.1',
            'hyperspy>=1.5.2',
            'ase>=3.17.0'
            'numba'
            ],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            ],
        package_data = {
            'atomap.tests.datasets': [
                'test_ADF_cropped.hdf5',
                'test_ABF_cropped.hdf5',
                'test_atom_lattice.hdf5'],
            'atomap.example_data': [
                'example_detector_image.hspy',
                'simulated_nanoparticle.hspy'],
            }
)
