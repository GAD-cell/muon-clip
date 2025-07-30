from setuptools import setup,find_packages

setup(
    name='muon_clip',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        "ray==2.47.1",
        "vllm==0.9.2",
        "transformers==4.53.0",
        "accelerate",
        "torch"
    ],
)