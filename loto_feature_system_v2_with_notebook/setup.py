from setuptools import setup, find_packages

setup(
    name="loto_feature_system_v2",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.9",
)
