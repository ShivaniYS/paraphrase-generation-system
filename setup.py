# setup.py
from setuptools import setup, find_packages

setup(
    name="paraphrase-system",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "nltk",
        "rouge-score",
        "bert-score",
    ],
)