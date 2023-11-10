from setuptools import setup, find_packages

setup(
    name="phiml",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "phiml = phiml.__main__:main",
        ]
    },
)
