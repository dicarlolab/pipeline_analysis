from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "brainio_dicarlo @ git+https://github.com/dicarlolab/brainio-dicarlo@adapter_version",
    "jupyter",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "tqdm",
]

setup(
    name='pipeline_analysis',
    description="Analyze pipeline neural data recordings.",
    long_description=readme,
    author="Martin Schrimpf",
    author_email='msch@mit.edu',
    url='https://github.com/brain-score/brain-score',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='neural-data,analysis,primate-vision,brainio',
    test_suite='tests',
)
