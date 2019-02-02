from setuptools import setup, find_packages

FULL_VERSION = '0.0.0'

with open('README.md') as f:
    readme = f.read()

setup(
    name='deep_da',
    version=FULL_VERSION,
    description='Deep learning based domain adaptation algorithms.',
    long_description=readme,
    author='Asahi Ushio',
    author_email='aushio@keio.jp',
    packages=find_packages(exclude=('test', 'dataset', 'random', 'tfrecord', 'checkpoint')),
    include_package_data=True,
    test_suite='test',
    install_requires=[
        'tensorflow-gpu',
        'scipy',
        'numpy',
        'toml',
        'POT'
        # 'Pillow'
        # 'pandas',
        # 'nltk',
        # 'sklearn',
        # 'flask',
        # 'werkzeug',
    ]
)
