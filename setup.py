from setuptools import setup

setup(
    name='Rover',
    version='',
    packages=['Rover', 'Rover.Environments', 'Rover.utils', 'Rover.algos'],
    url='',
    license='',
    author='Luiz Afonso Marão',
    author_email='',
    description='',
    install_requires=[
        'gymnasium==0.28.1',
        'numpy',
        'matplotlib',
        'sb3-contrib @ git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib@a7a4e540a6e8c7178c8b647ae2f753d869edd77d',
        'stable-baselines3 @git+https://github.com/DLR-RM/stable-baselines3@77b09503d7daf4ddb45992b4409a3b686d20aa24#egg=stable_baselines3',
        'torch==2.0.1',
        'mujoco==2.3.3',
        'opencv-python==4.7.0.72',
        'tqdm==4.65.0',
        'imageio==2.27.0',
        'imageio-ffmpeg==0.4.8',
        'rich==13.3.3'
    ],
)
