import setuptools

setuptools.setup(
        name='pychanneling',
        version='v0.1-alpha',
        description='Analysis package for channeling experiments',
        url='http://github.com/ebagli/pychanneling',
        author='Enrico Bagli',
        author_email='enrico.bagli@gmail.com',
        license='MIT',
        packages=setuptools.find_packages('pychanneling'),
        classifiers=[
                "Development Status :: 3 - Alpha",
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"
                ],
        install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'uproot'
        ],
        include_package_data=True,
        zip_safe=False)
