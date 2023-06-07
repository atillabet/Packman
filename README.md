# Pacman
In the game op pacman player controls pacman figure, with goal of scoring as many points as posible, main antagonists are ghosts who are trying to stop player from doing it. Ghosts can capture player by conacting with them. There are four of them. To find with them player can eat one of the 4 "tabletkas", by doing it, player will temporarily gain ability to kill ghost, also by contacting with them, after short period of time ghost will respawn in their starting point. 
To score points player can:
 - eat tabletkas - 50 points
 - kill ghosts - 200 points
 - collect dots from map - 10 points
 
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
 - load_model - it will load network from folder
 - save_model - it will save network from folder

Rewards returned from gym enviroment are modified in order to improve learning, because if you use stardart rewards model going to just try to kill one ghost and then will stop playing.

# How to run it
In order to run program you should just run play.py from model you pick. 
If you want to train model comment following line:
````
agent.load_model()
````
To test model uncomment it and to visualisy game you should uncomment this line:
````
env.render()
````

# Cuda 
To increase speed of trainig you should use CUDA, to install it you should follown this [insctructions](https://medium.com/pythoneers/cuda-installation-in-windows-2020-638b008b4639)
IF YOU DON"T HAVE CUDA installed you should comment(or delete) following two lines of code:
````
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
````

# Libs
To use this code it's nesesary to install all libs from requirements.txt
Note: All libs from requirements.txt have verions to work with cuda 1.7.4 if you are using newer veriosn of cuda(or not using it) feel free to update verions of libs.
