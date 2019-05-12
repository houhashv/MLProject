from setuptools import setup
import setuptools


requirements = ['scikit-learn', 'pandas', 'xgboost', 'matplotlib', 'mlxtend', 'shap', 'numpy', 'requests', 'networkx',
                'keras', 'tensorflow']

setup(name='ml-project',
      packages=setuptools.find_packages(),
      version='0.2',
      description='automatic machine learning package build by Yoss the Boss of Data and g-stat community',
      install_requires=requirements,
      include_package_data=True)
