from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="delierium",
    version="0.1.0",
    description="Symmetry Analysis for ODEs using SageMath",
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=["more-itertools"],
    install_requires=["more-itertools", "pylatexenc", "sympy", "symengine"]
    python_requires = ">=3.12",
    authoremail='tapir@aon.at',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Mathematicians',
        'Topic :: Calculus :: ODE',
        'Topic :: Scientific/Engineering :: Mathematics'
        ],
    keywords='ODE PDE Lie Symmetry',
    # fixme: an empty key should be enough?
    package_dir={'delierium': 'src/delierium'},
    project_urls={'Source': 'https://github.com/tapir442/delierium'}
)
