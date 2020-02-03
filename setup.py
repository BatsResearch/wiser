from setuptools import setup, find_packages

setup(
    name='WISER',
    version='0.0.1',
    url='https://github.com/BatsResearch/wiser.git',
    author='Esteban Safranchik, Shiying Luo, Stephen H. Bach, Daniel Ritter',
    author_email='esteban_safranchik@brown.edu, shiying_luo@brown.edu, sbach@cs.brown.edu',
    description='WISER (Weak and Indirect Supervision for Entity Recognition) is a ' +
                'framework for programming sequence tagging neural networks for named ' +
                'entity recognition',
    packages=find_packages(),
    package_data={
        '': ['viewer/viewer.html', 'viewer/viewer.js'],
    },
    install_requires=['allennlp==0.8.4', 'labelmodels'],
    dependency_links=[
        'git+https://github.com/BatsResearch/labelmodels.git'
    ],
)
