"Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer", by David Madras, Toniann Pitassi, and Richard Zemel (NIPS 2018). Available on ArXiv at <https://arxiv.org/abs/1711.06664>.

## Setup
This code uses Python 3.6. You can set up a virtual environment for the project as follows, with `python3` pointing to some Python 3.6+:
```mkdir venv
python3 -m venv ~/venv/ltd
source ~/venv/ltd/bin/activate
pip install -r requirements.txt
```

## Code
The code is structured partially based on Dustin Tran's suggestions (<http://dustintran.com/blog/a-research-to-engineering-workflow>). All scripts are stored in `src` and run from main project folder directly above that. The subfolder `src/codebase` contains most of the required code for running experiments (classes, utilities, etc.). `conf` contains config files with adjustable system and model parameters. `data` contains the data splits used in the paper.

## Running the code
The main script is `run_model.py`. An example of usage would be:
`python src/run_model.py -d compas -n usage_example -pc 0.1 -fc 0.2 -pass -def -dm highacc`
which will run a learning-to-defer model on the COMPAS dataset with a high-accuracy DM (scenario 1 in the paper), with gamma (the PASS coefficient) equal to 0.1, and alpha (the fair regularization coefficient) equal to 0.2. The experiment results will be saved in your experiment results directory (as defined in your directory config file), in a folder called `usage_example`.

## Config files
Config files are contained in `conf`. `conf/dirs` contains directory configs: `exp` is the experiment results directory, `log` is the Tensorboard logging directory, and `data` is where the data is stored. In `conf/model`, the config files define model and dataset related parameters.

## Data
Data splits are stored in the `data` subfolder for the COMPAS dataset. `x` is data features, `y` is labels, `attr` is the sensitive attribute, `ydm` is DM predictions, `y2` is the auxiliary information given to the DM (Z in the paper). In the "lowacc" DM, `make_noisy` indicates whether an example is in the DM's "inconsistent" group, `do_noisy_replace` indicates whether or not that example was flipped, and `ind` gives the feature index which determined the inconsistent group.

## References
David Madras, Toniann Pitassi, and Richard Zemel. Predict Responsibly: Increasing Fairness by Learning to Defer. In Advances in Neural Information Processing Systems, 2018.

