# EunkyoGo

The Eunkyo (pronounced "un-gyo") Go agent is named after the professional player and creator of the [Go Inside](https://www.youtube.com/@eunkyodo) YouTube channel. Its training is based on the AlphaGo strategy using reinforcement learning and a modified version of the Monte Carlo Search Tree. The dataset consists entirely of self-play experience.

## Installation

1. Create a new virtual environment using Anaconda:

```bash
conda create -n eunkyo python=3.9.18
conda activate eunkyo
```

2. Install required libraries:

```bash
conda install cudatoolkit=11.8.0 cudnn=8.9.2.26
```

```bash
pip install -q h5py==3.9.0 numpy==1.25.2 tensorflow==2.10.1
```

Additional step for Windows platform:

```bash
conda install nvcc_win-64=10.1
```

## Train Agent

The agent is trained via reinforcement learning using only self-play experience from simulated games.

```bash
python train_agent.py -h

========== Agent Training Module ==========

usage: python train_agent.py -e exp -s 10

optional arguments:
  -h, --help                 : show this help message and exit
  -a AGENT, --agent AGENT    : agent filename prefix
  -b BOARD, --board BOARD    : Go ban size (default = 9)
  -c, --cont                 : continue game simulations from last experience save point
  -d, --disp                 : print game results to screen
  -e EXP, --exp EXP          : experience input filename prefix
  -r ROUNDS, --rounds ROUNDS : number of rounds per move selection (default = 1)
  -s SIMS, --sims SIMS       : number of games to simulate (default = 0)
```

To simulate 1,000 self-play games with 500 rounds per move played on a 9x9 board and save the results to a new game experience file in HDF5 format:

```bash
python train_agent.py -r 500 -s 1000
```

To continue running simulations from a saved file (omit the file extension):

```bash
python train_agent.py -c -e <experience_file> -r 500 -s 1000
```

With game experience in hand, the Go agent can be trained:

```bash
python train_agent.py -e <experience_file> -r 500 -s 0
```

With a trained agent, more self-plays can be simulated using the agent:

```bash
python train_agent.py -a <agent_file> -r 500 -s 1000
```

```bash
python train_agent.py -a <agent_file> -c -e <experience_file> -r 500 -s 1000
```

## Evaluate Agent

Compare the performance of two agents by pitting them against each other.

```bash
python eval_agent.py -h

========== Agent Evaluation Module ==========

usage: python eval_agent.py -a agent -o opponent -s 10

optional arguments:
  -h, --help                 : show this help message and exit
  -a AGENT, --agent AGENT    : challenger agent filename prefix
  -b BOARD, --board BOARD    : Go ban size (default = 9)
  -d, --disp                 : print game results to screen
  -o OPPO, --oppo OPPO       : champion agent filename prefix
  -r ROUNDS, --rounds ROUNDS : number of rounds per move selection (default = 1)
  -s SIMS, --sims SIMS       : number of games to simulate (default = 1)
```

### Improvements

This agent is functional but slow. Very slow! There are a number of ways to improve the efficiency of the code and thus increase running speed:

1. To determine if a move would result in self-capture, just check for an empty adjacent point. There should be no need to use the very time-consuming *deepcopy()* method for this. If there are none, check if any opponent stones would be captured and if the agent's stones would subsequently be left with liberties. Consider using a data structure to track stone string liberty counts.

2. When checking for a *ko* violation before playing a move, compute what the Zobrist hash would be *if* the move were played without modifying the game state. This way would again avoid the need to copy the board.

3. As another alternative to the *deepcopy()* method, play the game on two boards to allow for rollbacks. Move 1 is played on the first board. Moves 1 and 2 are played on the second. Moves 2 and 3 are played on the first and so on until the game is finished.

### Notes

* This is a hobby project built out of personal interest for educational purposes. The agent is not fully functional, and its robustness is nowhere near that of the likes of KataGo.

* Much of the base code has been taken from the [Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go) book by Pumperla and Ferguson.

* The board (Go ban) mechanisms are not as efficient as they can be; as such, agent training can take a LONG time before it can play competently.

* This project is not otherwise affiliated with Eunkyo Do, the professional Go player after whom it is named, and its performance is in no way reflective of her skill level.

### References

* [Computer Go Community Discord](https://discord.com/invite/eTVCY5b)
* [Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go)
