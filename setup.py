import setuptools

with open("README.md", "r",encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyloadnprep", 
    version="0.0.4",
    author="terri1102",
    author_email="terricodes@gmail.com",
    description="A simple preprocessing package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terri1102/pyloadnprep",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'pandas', 'numpy','mplfinance','pykrx'
      ],
    python_requires='>=3.6',
)