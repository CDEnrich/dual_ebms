import setuptools

long_description = """
Python package  
"""

setuptools.setup(
    name="energyflow",
    version="0.0.1",
    author="AB, JB, CED, MG, EVE",
    author_email="mgabrie@nyu.edu",
    description="python package for sampling with real-nvp flows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
