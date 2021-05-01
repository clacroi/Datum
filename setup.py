"""File to enable a pip installation"""

import setuptools

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name='datum',
    version='0.1.0',
    author='Corentin Lacroix',
    author_email='corent1.lacroix@gmail.com',
    description='Package to manage datasets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6.8',
    install_requires=[
        'pandas>=0.24.2',
        'Pillow>=5.4.1',
        'tqdm>=4.31.1',
        'bidict>=0.21.2',
        'lxml>=4.6.2'
    ],
    entry_points={
        'console_scripts': [
            'list-existing-datastores = datum.utils.datastore:list_existing_datastores',
            'list-existing-datasets = datum.utils.datastore:list_existing_datasets',
            'register-dataset = datum.scripts.register_dataset:main'
        ],
    }
)