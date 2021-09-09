# Reinforcement Learning Project



Repository for the project of the course "Reinforcement Learning" of the master degree "Data Science and Scientific Computing" @Units


## About the project

The aim of the project is to teach a computer agent to play the Atari 2600 game Breakout via the Reinforcement Learning technique of Double (Deep) Q-Learning.
    
We will use images of the screen game to make our agent learn a policy that can allow it to score a sufficient number of points in the game (how many depends on the computing power and the time at your disposal).


You can find the slides for the presentation [here](slides/Reinforcement_Learning_Project.pdf).



## The code

The notebooks files all have a corresponding `.py` file for the versioning (otherwise in the `.ipynb` files it's impossible to see the changes through git). They are made or by downloading the notebook as a `.py` file or through the Jupyter plugin `jupytext`, that reads and writes notebooks as plain text files: either Python, Julia, R scripts, Markdown, or R Markdown documents.

- The notebook [Breakout_intro.ipynb](code/Breakout_intro.ipynb) is used to understand the environment of the Breakout game.
- The notebook [Breakout.ipynb](code/Breakout.ipynb) is used to train the agent, and saves at specific checkpoints the weights of the neural networks and the statistics of the game (episode duration, rewards, loss).
- The notebook [Breakout_resume.ipynb](code/Breakout_resume.ipynb) is used to resume the training of the agent in case it stops (I noticed that it usually uses up all the available memory and it crashes). It uses automatically the last saved checkpoint and saves new ones.
- The notebook [Breakout_play.ipynb](code/Breakout_play.ipynb) is used to play a game with the agent. Depending on how much you want your agent to be trained, you can use different checkpoint weights.
- The file [Breakout_script.py](code/Breakout_script.py) is a cleaned version of the notebook [Breakout.ipynb](code/Breakout.ipynb) to be used in a cluster. It can be submitted as a job in the cluster using the script [script_cluster.sh](code/script_cluster.sh).

- The file [Breakout_resume_script.py](code/Breakout_resume_script.py) is a cleaned version of the notebook [Breakout_resume.ipynb](code/Breakout_resume.ipynb) to be used in a cluster. It can be submitted as a job in the cluster using the script [script_cluster_resume.sh](code/script_cluster_resume.sh).



## Info

To kill the annoying windows of the environment rendering that don't close, open a terminal and digit:

```bash
xkill
```

then click on the windows you want to kill.



## Linux Setup

Download, clone or fork (your choice) this repository in a directory `PATH_TO_DIR/`, then enter in the folder using:

```bash
cd PATH_TO_DIR/ReinforcementLProject/code
```

Create a virtual environment using `python3` (commands are provided for *Debian-like* GNU/Linux distributions --- see https://docs.python.org/3/library/venv.html for further explanation). If you don't have installed both `pip` and `virtualenv`, run
```bash
# Install pip for Python 3:
sudo apt-get install python3-pip
# Install virtualenv
python3 -m pip install virtualenv
```

If you have, skip the previous commands.

Then run

```bash
python3 -m virtualenv -p "$(which python3)" venv
```



If the following error

```bash
$ python3 -m virtualenv -p "$(which python3)" venv
ImportError: cannot import name 'ensure_file_on_disk'
```

appears, it is because of multiple versions of `virtualenv` are installed. Remove all the versions using `pip3 uninstall virtualenv` multiple times, until the message <font color="red"> `Cannot uninstall requirement virtualenv, not installed`</font> appears. Then re-run

```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv -p "$(which python3)" venv
```



Now you should see `PATH_TO_DIR/venv/` folder.
Activate the environment and install the requirements (don't specify the versions of the packages, so you'll get the latest versions):

```bash
source venv/bin/activate
python3 -m pip install -r ./requirements_INSTALLER.txt
```

Register the just-installed virtual environment for use with Jupyter:
```bash
python3 -m ipykernel install --user --name ReinforcementL --display-name "Python3 (RL virtualenv)"
```

Then type:

```bash
pip freeze
```

and save the output in a file named `requirements.txt`.



Too see the dependencies of a package, run

```bash
pip3 show <name_package>
```



Open your notebooks using jupyter-notebook (or jupyter-lab):

```
python3 -m jupyter notebook
```

To deactivate the environment use `deactivate` command.



To convert the `.ipynb` file in `HTML` use the following command

```bash
jupyter nbconvert --to html notebook.ipynb
```



### Breakout

If you want to use [Gym](https://gym.openai.com/) [Atari](https://gym.openai.com/envs/#atari) environments, among which there is Breakout, 

```
source venv/bin/activate
pip install gym[atari]
```

#### ROMs

In order to import ROMS, you need to download `Roms.rar` from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the `.rar` file. Save this file in a folder called `ROMs`.

To do so, open a terminal and install `unrar`:

```bash
sudo apt-get install unrar
```

Then to extract the file run:

```bash
unrar e -r Roms.rar
```

You'll get two `zip` files: `ROMS.zip` and `HC ROMS.zip`. Using the archive manager, unzip both files in the current directory.

Once you've done all that, run:

```
python -m atari_py.import_roms <path to folder>
```

In my case, this is:

```bash
python -m atari_py.import_roms ~/Documenti/Reinforcement\ Learning/Project/code/ROMs/ROMS/
```

This should print out the names of ROMs as it imports them. The ROMs will be copied to your `atari_py` installation directory (in my case: `/home/angela/Documenti/Reinforcement Learning/Project/code/venv/lib/python3.6/site-packages/atari_py/atari_roms/`).

##### Breakout's ROM

If you only want to use the game Breakout, you can only keep one ROM in the folder `ROMs`:

```dockerfile
Breakout - Breakaway IV (Paddle) (1978) (Atari, Brad Stewart - Sears) (CX2622 - 6-99813, 49-75107) ~.bin
```

and delete the other files. Then opening a terminal in that folder run

```bash
python -m atari_py.import_roms ./ROMs
```

