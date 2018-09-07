import numpy as np
import collections

class Dataset(object):

    def __init__(self, name, attr0_name, attr1_name, npzfile, seed=0, use_attr=False, load_on_init=True):
        self.name = name
        self.attr0_name = attr0_name
        self.attr1_name = attr1_name
        self.npzfile = npzfile
        self.use_attr = use_attr
        self.loaded = False
        self.seed = seed
        if load_on_init:
            self.tensors = {}
            self.load()
            self.make_validation_set()

    def load(self):
        if not self.loaded:
            dat = np.load(self.npzfile)
            self.dat = dat

            #get valid inds
            self.train_inds = dat['train_inds']
            self.valid_inds = dat['valid_inds']

            self.tensor_names = ['x', 'y','attr']
            if 'y2_train' in dat: self.tensor_names.append('y2')
            if 'ydm_train' in dat: self.tensor_names.append('ydm')
            for t in self.tensor_names:
                self.tensors[t] = {}
                for phase in ['train', 'test']:
                    self.tensors[t][phase] = dat['{}_{}'.format(t, phase)]
                
            if self.use_attr:
                for phase in ['train', 'test']:
                    self.tensors['x'][phase] = np.concatenate([self.tensors['x'][phase], self.tensors['attr'][phase]], 1)
            self.loaded = True

    def make_validation_set(self):
        if not 'valid' in self.tensors['x']:
            for t in self.tensor_names:
                self.tensors[t]['valid'] = self.tensors[t]['train'][self.valid_inds]
                self.tensors[t]['train'] = self.tensors[t]['train'][self.train_inds]

    def get_batch_iterator(self, phase, mb_size):
        batch_tensor_names = ['x', 'y', 'attr', 'ydm']
        batch_tensors = [self.tensors[t][phase] for t in batch_tensor_names]
        sz = batch_tensors[0].shape[0]
        batch_inds = make_batch_inds(sz, mb_size, self.seed, phase)
        iterator = DatasetIterator(batch_tensors, batch_inds)
        return iterator

class DatasetIterator(collections.Iterator):

    def __init__(self, tensor_list, ind_list):
        self.tensors = tensor_list
        self.inds = ind_list
        self.curr = 0
        self.ttl_minibatches = len(self.inds)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr >= self.ttl_minibatches:
            raise StopIteration
        else:
            inds = self.inds[self.curr]
            minibatch = [t[inds] for t in self.tensors]
            self.curr += 1
            return minibatch

def make_batch_inds(n, mb_size, seed=0, phase='train'):
    np.random.seed(seed)
    if phase == 'train':
        shuf = np.random.permutation(n)
    else:
        shuf = np.arange(n)
    start = 0
    mbs = []
    while start < n:
        end = min(start + mb_size, n)
        mb_i = shuf[start:end]
        mbs.append(mb_i)
        start = end
    return mbs
