import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="whacc",  # name of package
    version="0.0.10",
    author="Jonathan Cheung",
    author_email="jacheung@usc.edu",
    license='MIT',
    description="A pre-trained CNN for touch classification of whisker to objects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jacheung/whacc",
    packages=setuptools.find_packages(),
    package_data={
        'whacc': ['datasets/*.h5',
                  'model_checkpoints/*',
                  'model_checkpoints/model_201008/*',
                  'model_checkpoints/model_201008/variables/*',
                  'model_checkpoints/model_201008/assets/*'],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
