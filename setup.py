from setuptools import setup

setup(
    name="kempnn",
    version="1.0.0",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "hyperopt",
        "numba",
        "pandas",
        "openpyxl",
    ],
    extras_require={"develop": ["autopep8"]},
)
