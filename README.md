# Pacman
In the game of pacman player controls pacman figure, with goal of collecting as many points as posible, antagonists are ghosts who are trying to stop player from doing it. Ghosts can kill player by conacting with him, player has 3 lives. There are four of them. To find with them player can eat one of the 4 "tabletka", by doing it, player will temporarily gain ability to kill ghost, by contacting with them, after short period of time ghost will respawn in their starting point.

To gain points player can:
 - eat tabletka - 50 points
 - kill ghost - 200 points
 - collect dot from map - 10 points
 
 # Enviroment
 In this code [Pacman](https://www.gymlibrary.dev/environments/atari/ms_pacman/) from OpenAi gym was used, with atari space, in this environment gym allows to pass action from neuron network directly to enviroment.

Pacman in this game contains following:
- state shape - (210, 160, 3)
- action size - 9 (stop, move west, move south-west, ...)
- in state pacman will be represented with value 210

To use gym with atari space its necessary to use [atari license](https://pypi.org/project/AutoROM.accept-rom-license/)

# Agent
In this code all models contains same neuron network. All models contains methods:
 - get_action - this function will return predicted action for specific state
 - train - this function will train network by using results of code
 - update_memory - this function will update memory by new states
 - load_model - it will load network from folder
 - save_model - it will save network from folder

Rewards returned from gym enviroment are modified in order to improve learning, because if you use stardart rewards model going to just try to kill one ghost and then will stop playing.

# How to run it
Firstly you need to clone git repo using
````
git clone https://github.com/atillabet/Packman.git
````
Then install all necessary dependencies using
````
pip install -r requirements.txt
````
To lunch model throw console you should go to specific model folder and run following command:
````
python train.py --output-file Model1.h --input-file Model2.h
````
- output-file parameter corresponds to file where model will be stored.
- input-file is optional parameter and can be used to download pre-treined model to train.
To lunch test model you should run next command
````
python test.py --input-file Model2.h
````

# Cuda 
To increase speed of trainig you should use CUDA, to install it you should follown this [insctructions](https://medium.com/pythoneers/cuda-installation-in-windows-2020-638b008b4639).

IF YOU DON"T HAVE CUDA installed you should comment(or delete) following two lines of code:
````
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
````

# Libs
To use this code it's nesesary to install all libs from requirements.txt.

Note: All libs from requirements.txt have verions to work with cuda 1.7.4 if you are using newer veriosn of cuda(or not using it) feel free to update verions of libs.
