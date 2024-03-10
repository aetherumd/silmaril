from setuptools import setup
import setuptools

setup(
    name="silmaril",
    version="0.0.1",
    author="Seyong Park",
    description="Library for generating synthetic observations of strongly lensed high-redshift galaxies",
    url="https://github.com/syp2001/silmaril",
    packages=setuptools.find_packages(exclude=["tests"]),
    package_dir={'silmaril': 'silmaril'},
    package_data={'silmaril': ['data/*.txt','data/*.dat','data/mean_throughputs/*.txt']},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
    ),
    install_requires=["scipy","numpy","matplotlib","astropy","tqdm","pandas","numba"]
)