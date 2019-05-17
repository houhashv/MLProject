from setuptools import setup
import setuptools


requirements = ['scikit-learn', 'pandas', 'xgboost', 'matplotlib', 'mlxtend', 'shap', 'numpy', 'requests', 'networkx']

setup(name='ml-project',
      packages=setuptools.find_packages(),
      version='0.1',
      description='for ml project',
      install_requires=requirements,
      include_package_data=True)
