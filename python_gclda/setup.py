try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Python gcLDA Implementation',
    'author': 'Timothy Rubin',
    'url': 'https://github.com/timothyrubin/python_gclda/',
    'download_url': 'https://github.com/timothyrubin/python_gclda/',
    'author_email': 'tim.rubin@gmail.com',
    'version': '1.0',
    'install_requires': ['numpy','scipy','matplotlib'],
    'packages': ['python_gclda_package'],
    'name': 'python_gclda'
}

setup(**config)
