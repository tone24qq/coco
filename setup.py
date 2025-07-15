from setuptools import find_packages, setup

setup(
    name="rf_infer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["scikit-learn>=1.2.0", "joblib", "numpy"],
    entry_points={"console_scripts": ["rf-infer=rf_infer.cli:main"]},
)
