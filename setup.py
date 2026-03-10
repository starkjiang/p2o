from setuptools import setup, find_packages

setup(
    name="p2o",
    version="0.1.0",
    description="Proximal Preference Optimisation — comparing DPO, IPO, KTO, P²O, PKTO on GPT-2",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.40.0",
        "datasets==2.19.0",
        "accelerate==0.29.3",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "tqdm",
    ],
)
