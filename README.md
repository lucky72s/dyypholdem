# DyypHoldem

A python implementation of [DeepHoldem](https://github.com/happypepper/DeepHoldem) (which in turn is an adoption of [DeepStack](https://www.deepstack.ai/s/DeepStack.pdf) for No Limit Texas Hold'em, extended from [DeepStack-Leduc](https://github.com/lifrordi/DeepStack-Leduc)).

It uses a cross-platform software stack based on Python and PyTorch and has most data for bucketing pre-calculated and stored in data files or an sqlite3 database. This reduces loading times significantly and DyypHoldem loads and reacts on a fairly modern system in reasonable time frames.



## Setup

DyypHoldem runs with the following components. It has been tested on both Linux with kernel 5.10+ and Windows 10:

### Required

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.10+: install a suitable package (OS, package format, compute platform) as instructed on their website.
- [Git LFS](https://git-lfs.github.com/): the large data files are stored via Git LFS. Install it on your system and run the command `git lfs install`once **before** cloning this repository.

### Optional but recommended

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 11.3+ for GPU support (change the value in `settings/arguments.py` to `use_gpu = False` if no GPU is present)
- Python [sqlite3 module](https://docs.python.org/3/library/sqlite3.html) for the large category tables on the flop, turn and river. It can be easily installed via `pip install sqlite3` in typical Python environments. The lookup tables are also included as flat files and to use them instead, set `use_sqlite = False` in `settings/arguments.py` . This leads to longer startup times, but slightly improves performance for longer running tasks, like data generation or training.
- [Loguru module](https://github.com/Delgan/loguru) for extended logging. It can be installed via `pip install loguru`. If needed, the logging output can be directed to `stdout` only by setting the flag `use_loguru = False` in `settings/arguments.py`.



## Using / converting DeepHoldem models

DyypHoldem comes with a set of trained counterfactual value networks, converted from ones previously provided for [DeepHoldem](https://github.com/happypepper/DeepHoldem/issues/28#issuecomment-689021950). These models are stored in `data/models/`. To re-use your own models, they can be converted into DyypHoldem's format with the following commands:

```shell
cd src && python torch7/torch7_model_converter.py <file> <street> <mode>
```

with

- `<file>`: path to the model to be converted (both the .info and .model files are required)
- `<street>`: the street the model is for (`1`= preflop, `2`= flop, `3`= turn, `4`= river)
- `<mode>`: mode the model file was saved in torch7 (`binary`or `ascii`)



## Creating your own models

New models can be created in the same way as in DeepHoldem. First a set of data is generated for a street and then the value network is trained. This is repeated for the other streets, going from river to flop. For preflop the same auxiliary network as in DeepHoldem is used.

Step-by-step guide for creating new models:

1. Set the parameters `gen_batch_size` and `gen_data_count` in `settings/arguments.py` to control how much data is generated - the standard batch size is `10` and the standard data count is `100000`
1. Specify the folders in `settings/arguments.py` where the training data (`training_data_path`) and the trained models (`training_model_path`) should be stored
1. Generate data via `cd src && python data_generation/main_data_generation.py 4`
2. Convert the raw data to bucketed data via `python training/raw_converter.py 4`
5. Train the model for the street via `python training/main_train.py 4`
6. Models will be generated in the path specified in step `2`. Pick the model you like best and place it inside
   `data/models/river` and rename it to `final_<device>.tar`, with `<device>` either `gpu` or `cpu` depending on your system configuration. To automatically save the model with the lowest validation, set the flag `save_best_epoch = True` in `settings/arguments.py`.
7. Repeat steps 3-6 for turn and flop by replacing `4` with `3` or `2` and placing the models under the turn and flop folders.



## Playing against DyypHoldem

You can play manually against DyypHoldem via an ACPC server. Details on ACPC as well as the source code for the server can be found here: [Annual Computer Poker Competition](http://www.computerpokercompetition.org/). A pre-compiled server and run script for Linux is included. To play follow the these steps:

1. `cd acpc_server`
2. Run `./dyypholdem_match.sh <hands> <seed>` with `<hands>` for the number of hands to be played and `<seed>` for the seed of the random number generator. By default the ports used for the players are `18901`and `18902`.
3. Open a second terminal and `cd src && python player/manual_acpc_player.py <hostname> <port1>` with the IP address of the server (e.g. `127.0.0.1` if on the same machine) as `<hostnem>` and `<port1>` either `18901` to play as small blind  or `18902` to play as big blind.
4. Open a third terminal and `cd src && python player/dyypholdem_acpc_player.py <hostname> <port2>` with the same `<hostname>` and `<port2>` the port not used for the manual player.
5. You can play against DyypHoldem using the manual player terminal. Use the following commands to control your actions: `f` = fold, `c` = check/call, `450` = raise my total pot commitment to 450 chips.



## DyypHoldem vs. Slumbot

DyppHoldem also includes a player that can play against [Slumbot](https://www.slumbot.com/) using its API.

1. `cd src`
2. `python player/dyypholdem_slumbot_player.py <hands>`

Specify the number of `<hands>` you like DyypHoldem to play and enjoy the show :-).

