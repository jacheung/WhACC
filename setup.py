import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="whacc",  # name of package
    version="0.0.2",
    author="Jon C",
    author_email="jacheung@usc.edu",
    description="A pre-trained CNN for touch classification of whisker to objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacheung/autocurator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
