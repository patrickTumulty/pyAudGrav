import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyAudGrav",
    version="0.1.0",
    author="Patrick Tumulty",
    author_email="ptumulty1@gmail.com",
    description="Algorithmically edit and rearrange audio clips, both in time and space, using the equation of gravity.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patrickTumulty/audGrav",
    packages=['pyAudGrav'],
    install_requires = ['numpy>=1.16.4',
                        'scipy>=1.3.0',
                        'SoundFile>=0.10.2',
                        'matplotlib>=3.1.0',
                        'pyloudnorm>=0.1.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.3',
    include_package_data=True
)
