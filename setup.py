"""setup.py file."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup_requirements = ["setuptools"]

setup(
    author="Kitware, Inc.",
    author_email="kitware@kitware.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
    ],
    description="""Scripts for generating tests using vid aug""",
    name="gen_data",
    setup_requires=setup_requirements,
    install_requires=requirements,
    packages=find_packages(),
    url="https://github.com/darpa-sail-on/gen_data",
    zip_safe=False,
)
