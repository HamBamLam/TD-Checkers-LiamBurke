# TD-Checkers

An implementation of the TD-Gammon algorithm to Checkers using TensorFlow 1.14. The code uses a very simple neural network with sigmoid activation to evaluate the current board state and select the best available action. Eligibility traces are applied to the gradients to factor in past actions. Action selection is done with a 3-ply alpha-beta pruninmg search. Training is done by initially playing against a random agent to ensure the agent explores a large amount of states. After the random training, it switches to a self training algorithm where the agent plays against itself. The idea is to get a basic strength in the game from the random training and then learn more advanced strategies by playing against itself.

## Setup/How to Use

This code was written on a Windows device, and thus I decided to use an older version of TensorFlow(1.14) in order to use the GPU. The easiest way to set up the environment for the code is as follows:

* clone the repository into your local device (<https://github.com/HamBamLam/TD-Checkers-LiamBurke.git>)

* create a suitable conda environment with Python 3.5 or 3.6, such as the example below. Note that depending on what packages you have installed, you may need to install additional packages or downgrade some packages to work with the chosen python version. I only included the packages I needed below.

    ```bash
    conda create --name tf114 python=3.5
    conda activate tf114
    conda install -c conda-forge tensorflow-gpu=1.14
    conda install -c conda-forge markdown==2.6.11 #if using tensorboard
    ```

* Make sure the Python Interpreter is the created conda environment, then navigate to the repository and run the code any of the following commands:

    ```bash
    python main.py #train new model
    python main.py --restore #continue training last trained model
    python main.py --restore --test #test trained model against random agent
    python main.py --restore --draw #view the board state as the agent plays against a random agent
    ```

## Future Improvements

Some improvements I would like to make are as follows:

* __Looping Behaviour.__ Sometimes, when very few pieces are left on the board, and the agent has a king near the back of its own side, it will get stuck in a loop of moving the king piece forward then backwards until a forced draw happens. The behaviour is not very frequent, but is still quite annoying when it does happen. Introducing some form of limit on repeated moves could fix this, but I feel it defeats the purpose of minimal human involvement in the agent's decision making, as it is technically legal behaviour
* __Improve runtime.__ Due to the alpha-beta pruning, the runtime becomes quite long, with every 10 games taking roughly 1 minute. When factoring in validation tests and the 10000 episodes per training, training can take almost a full day to complete at times
* __Improve the self-training method.__ Currently, from random training, the agent can become almost unbeatable against the random agent, reaching win percentages of over 98%. However, self-training can sometimes actually worsen the agent's performance against the random opponent. This problem was mitigated somewhat by adjusting features, but it still can happen.
* __Tuning Parameters.__ Due to my lacking knowledge of neural networks and reinforcement learning hyperparameters, many of the parameters were set somewhat arbitrarily, such as hidden layer size and lambda. I simply followed patterns for these that I saw in others' code, and did not have much time to experiment with these values much.
* __Additional Features.__ I think adding additional features based on what are known to be favourable positions, such as central control, piece grouping, etc. could improve the strategy of the agent, but I wanted to focus on using simple features to avoid adding too much human influence to the model

## Issues

If you face issues where the checkpoint saver fails to replace or rename files, this is due to an issue with TensorFlow's built in file writer. To fix this:

* Navigate to where the package is stored and open file_io.py
  * Example:  C:\Users\user\miniconda3\envs\tf114\Lib\site-packages\tensorflow\python\lib\io
* Replace the atomic_write_string_to_file method with the following code:

```python
def atomic_write_string_to_file(filename, contents, overwrite = True):
"""Writes to `filename` atomically.

This means that when `filename` appears in the filesystem, it will contain
all of `contents`. With write_string_to_file, it is possible for the file
to appear in the filesystem with `contents` only partially written.

Accomplished by writing to a temp file and then renaming it.

Args:
    filename: string, pathname for a file
    contents: string, contents that need to be written to the file
"""
temp_pathname = filename + ".tmp" + uuid.uuid4().hex
write_string_to_file(temp_pathname, contents)
try:
    if overwrite and os.path.exists(filename):
        os.remove(filename)
    rename(temp_pathname, filename, overwrite)
except errors.OpError:
    delete_file(temp_pathname)
    raise
```

This should fix any issues with renaming/replacing the checkpoint file. If the issue persists, ensure that the file permissions allow reading and writing.

## Acknowledgements

The Tensorflow code is based heavily on a TD-Gammon model made by __Jim Fleming__, which was made several years ago using a TensorFlow 0.x version on Python 2. Without this, I have to admit I would have been very lost, as I had no experience with packages such as PyTorch and Tensorflow beforehand. Additonally, the compact 8x4 representation of the checkers board was based on the one used in __Sam Ragusa's__ Q-Learning AI for checkers, which helped reduce the space complexity and make action lists.
