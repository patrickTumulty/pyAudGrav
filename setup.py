import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyAudGrav", # Replace with your own username
    version="0.0.1",
    author="Patrick Tumulty",
    author_email="ptumulty1@gmail.com",
    description="Algorithmically edit and rearrange audio clips, both in time and space, using the equation of gravity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patrickTumulty/audGrav",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.3',
)