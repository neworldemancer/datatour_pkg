import setuptools

setuptools.setup(
    name='datatour',
    version='0.2',
    scripts=['datatour'] ,
    author="Mykhailo Vladymyrov",
    author_email="neworldemancer@gmail.com",
    description="DataTour: See your data in its native dimensions!",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neworldemancer/datatour_pkg",
    packages=setuptools.find_packages(),
    py_modules=['datatour'],
    install_requires=[
        'numpy', 'scipy', 'plotly'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
 )