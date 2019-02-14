import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name="pychanneling",
        packages=["pychanneling"],
        version="v0.4-alpha",
        description="Analysis package for channeling experiments",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="http://github.com/ebagli/pychanneling",
        author="Enrico Bagli",
        author_email="enrico.bagli@gmail.com",
        license="MIT",
        classifiers=[
                "Development Status :: 3 - Alpha",
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License"
                ],
        install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "uproot"
        ],
        include_package_data=True)
