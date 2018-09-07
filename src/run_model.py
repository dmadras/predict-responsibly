import argparse
import os
import json
import sys
from codebase.dataset import Dataset
from codebase.models import *
from codebase.trainer import Trainer, make_dir_if_not_exist
from codebase.defaults import get_default_kwargs

parser = argparse.ArgumentParser(description='Run model')
parser.add_argument('-d', '--dataset', help='dataset name', default='')
parser.add_argument('-n', '--expname', help='experiment name', default='temp')
parser.add_argument('-pc', '--pass_coeff', help='coefficient on pass term', default=0., type=float)
parser.add_argument('-fc', '--fair_coeff', help='coefficient on fairness term', default=0., type=float)
parser.add_argument('-pass', '--pass_model', action='store_true', help='run PASS model?', default=False)
parser.add_argument('-def', '--defer', action='store_true', help='learn to defer? (as opposed to reject)', default=False)
parser.add_argument('-ne', '--num_epochs', help='number of training epochs', default=10000, type=int)
parser.add_argument('-pa', '--patience', help='training patience for early stopping', default=20, type=int)
parser.add_argument('-ua', '--use_attr', action='store_true', help='use sensitive attribute in data', default=True)
parser.add_argument('-mrs', '--model_random_seed', help='what random seed to use for model', default=0, type=int)
parser.add_argument('-drs', '--data_random_seed', help='what random seed to use for data', default=0, type=int)
parser.add_argument('-dm', '--dm_type', help='type of DM to use: inconsistent/biased/highacc', default='inconsistent')
parser.add_argument('-dirs', '--dirconf', help='config file for dirs', default='conf/dirs/home.json')
args = vars(parser.parse_args())

#load options
dirs = json.load(open(args['dirconf'], 'r'))
model_args = json.load(open('conf/model/{}.json'.format(args['dataset'])))

# get default params
data_kwargs, model_kwargs = get_default_kwargs(dirs['data'], args, model_args)

#get dataset
data = Dataset(**data_kwargs)
print('Data loaded.')
if not args['pass_model']:
    model = BinaryMLP(**model_kwargs)
else:
    if args['defer']:    
        model = MLPDeferModel(**model_kwargs)
    else:
        model = MLPRejectModel(**model_kwargs)
    
print('Model loaded.')

with tf.Session() as sess:
    print('Session created.')
    resdirname = os.path.join(dirs['exp'], args['expname'])
    logdirname = os.path.join(dirs['log'], args['expname'])
    ckptdirname = os.path.join(resdirname, 'checkpoints')
    for d in [resdirname, logdirname, ckptdirname]:
        make_dir_if_not_exist(d)

    #create Trainer
    trainer = Trainer(model, data, sess=sess, logs_path=logdirname, \
                 checkpoint_path=ckptdirname, results_path=resdirname)

    save_path = trainer.train(n_epochs=args['num_epochs'], patience=args['patience'])
    trainer.restore(save_path)
    trainer.test()
    print('Complete.')
    os._exit(0)
