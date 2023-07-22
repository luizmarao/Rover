# Rover
Rover is a set of environments and tools used to perform the experiments of the 
doctorate thesis "Deep Reinforcement Learning Multi-Sensor Based Navigation and Control".

This thesis was presented by Luiz Afonso Batalha Marão to the São Carlos School of 
Engineering, of the University of São Paulo.

The Rover environment's are based in the MujocoEnv class of the Gymnasium package 
(https://github.com/Farama-Foundation/Gymnasium), and the agents are trained by an
adapted version of Stable-Baselines3 (https://github.com/DLR-RM/stable-baselines3) 
PPO algorithm.

## Installation
The Rover package was tested in Ubuntu 22.04. Thus, it may not be compatible
with former versions. 
### Install using pip
In your shell, at the desired root folder, run the following commands:
```
git clone https://github.com/luizmarao/Rover
cd Rover
pip install -e.
```

This will also download and install every package's dependencies. You may want
to do it in a new environment (I use conda).

## Example
You may launch your experiments calling run.py in your shell. Here follows an
example line:
```
python -m Rover.run --exp_name=4We_001 --exp_root_folder=~/Experiments --env=Rover4We-v2 --num_timesteps=200000 --n_epochs=50 --num_env=4 --n_steps=4096 --batch_size=8192 --networks_architecture=RovernetClassic --img_red_size='(32, 32)' --conv_layers='[(16, 4, 2), (32, 3, 1), (64, 2, 1)]' --seed=10
```
The run.py file is configured to perform MuJoCo renderings headless (egl). If you
don't have a compatible graphics card, make your own copy of the file and remove
this configurations' line.

You may also run experiments using a custom script. However, Rover envs were designed
to run in a SubProcVecEnv wrapper. Thus, they need to be called inside an 
```if __name__ == "__main__":``` statement.

## Citing the Project

To cite this repository in publications:

```bibtex
@misc{rover,
  author = {Mar{\~{a}}o, Luiz Afonso Batalha},
  title = {Rover},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/luizmarao/Rover}},
}
``