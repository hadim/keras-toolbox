from setuptools import setup
from setuptools import find_packages


setup(name='keras-toolbox',
      version='0.1.2',
      description='Everyday toolbox for Keras',
      author='Hadrien Mary',
      author_email='hadrien.mary@gmail.com',
      url='https://github.com/hadim/keras-toolbox',
      download_url='https://github.com/hadim/keras-toolbox/archive/master.zip',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'tqdm>=4.25.0,<5',
            'h5py>=2.8.0,<3',
            'matplotlib>=2.2.3<3',
            'scipy>=1.1.0,<2',
            'numpy>=1.15,<2',
            'pandas>0.23.4,<1',
      ]
     )
