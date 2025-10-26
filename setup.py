from setuptools import setup, find_packages

setup(
    name="grid_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "pyyaml",
        "pandapower",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "jupyterlab",
        "networkx",
        "streamlit",
    ],
    python_requires=">=3.9",
)
