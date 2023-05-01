# aNNa

## Terminology

- profile: A profile describe the properties (rounds, decks size) of a given game (texas, leduc, ...)
- plan: A plan describe how a profile should be trained (epochs, network parameters, ...)
- snet: Neural network that is trained using MC simulation to predict the strength of a hand
- qnet: Neural network that is trained to play the game

## Plans

All plans but `texas_limit_zero` use compact encoders for actions and use snet based encoders for cards.
This works for well simple games but prevent the learning of advanced strategy when playing texas.

The plan `texas_limit_zero` use raw binary encoders (zero abstractions) for cards and actions.

## Training

- Generate snet and qnet graphs using python scripts (see `nnet-graphs/generate.sh`)
  - A qnet is specific to a given plan
  - A snet is specific to a given profile and a number of players
    - which means snet for lower number of players will be shared with plan of high number players for a given profile

- Train the snet until reaching a satisfying accuracy (see `learning/src/bin/snet_train.rs`)
  - use `ctrl+c` to stop the training and save the network (current epoch will finish)
  - once trained move the network `.data` in `resources/networks`

- Train the qnet until reaching a satisfying score (see `bin-arena/src/bin/train.rs`)
  - if you want to be able to resume a training (snapshot mode), `save` must be disabled in redis configuration
    - In NixOS: `services.redis.save = []`
    - snapshots can be
      - cleaned periodically using `bin-arena/cleanup.sh ./ 2`
      - restored by restarting redis `sudo systemctl restart redis.service`
  - require instance of redis running at "redis://127.0.0.1/ 
    - `FLUSHALL` to clear data in db0 before starting a fresh run
  - the worker will benchmark the network at the end of every epoch (see `bin-arena/src/train_worker.rs`)
    - the benchmark is run by taking each network and playing it against SNetLex (rule-based AI)
  - the results of the benchmark are stored in the `benchmarking.log` file
    - can be plotted using 
      `gnuplot -e "filename='benchmarking.log'" benchmarking.gnuplot` (see `bin-arena/benchmarking.gnuplot`)

## Playing

   cd bin-player
   cargo run ../resources/ ps 2> /tmp/anna-errors.log
