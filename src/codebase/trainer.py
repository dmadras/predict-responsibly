import numpy as np
import tensorflow as tf
import os
from codebase.metrics import *

# defaults
BATCH_SIZE = 64
LEARNING_RATE = 0.01
LOG_PATH = './tfboard_logs'

class Trainer(object):
    def __init__(self, model, data, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, sess=None, logs_path=LOG_PATH, \
                 checkpoint_path=None, results_path=None):
        self.data = data
        if not self.data.loaded:
            self.data.load()
            self.data.make_validation_set()
        self.model = model
        self.batch_size = batch_size
        self.logs_path = logs_path
        self.checkpoint_path = checkpoint_path
        self.results_path = results_path
        assert not None in [self.checkpoint_path, self.results_path]

        #create optimizer
        self.train_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)  
        self.train_op = self.train_opt.minimize(
            self.model.loss,
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        )

        self.sess = sess or tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()  

        #define which losses + metrics we want to track in learning
        self.losses = {'loss': self.model.loss, 'class_loss': self.model.class_loss, \
                  'fair_reg': self.model.fair_reg, 'idk_loss': self.model.idk_loss,
                  'loss_class': self.model.loss_class, 'loss_idk': self.model.loss_idk}
        self.tensors = {'Y_hat': self.model.Y_hat, 'idk': self.model.idks, 'A': self.model.A, 'Y': self.model.Y,
                   'class_loss': self.model.class_loss, 'Y_DM': self.model.Y_DM,
                   'Yt': self.model.Y_ttl}
        self.metrics = {'errRate': lambda T: errRate(T['Y'], T['Y_hat']), 'DI': lambda T: DI(T['Y'], T['Y_hat'], T['A']), \
                   'errRate-T': lambda T: errRate(T['Y'], T['Yt']),
                   'DI-T': lambda T: DI(T['Y'], T['Yt'], T['A']),
                   'rejErrRate': lambda T: rejErrRate(T['Y'], T['Y_hat'], T['idk']),
                   'IDKRate': lambda T: PR(T['idk'])}

    def process_minibatch(self, phase, feed_dict, losses, tensors, print_grad=False):
        loss_names = sorted(losses.keys())
        tensor_names = sorted(tensors.keys())
        ops = [losses[name] for name in loss_names] + [tensors[name] for name in tensor_names]
        if phase == 'train':
            ops = [self.train_op] + ops
            ret = self.sess.run(ops, feed_dict=feed_dict)
            ret = ret[1:]
        else:
            ret = self.sess.run(ops, feed_dict=feed_dict)
        loss_dict = {loss_names[i]: np.mean(ret[i]) for i in range(len(loss_names))}
        tensor_dict = {tensor_names[i]: ret[i + len(loss_names)] for i in range(len(tensor_names))}
        return loss_dict, tensor_dict

    def process_epoch(self, phase, losses, tensors, epoch):
        epoch_iter = self.data.get_batch_iterator(phase, self.batch_size)
        L = {l: 0. for l in losses}
        T = {t: None for t in tensors}
        self.batches_seen = 0
        for x, y, a, y_dm in epoch_iter:
            self.batches_seen += 1
            feed_dict = {self.model.X: x, self.model.Y: y, self.model.A: a, self.model.epoch: np.array([epoch]), self.model.Y_DM: y_dm}
            loss_dict, tensor_dict = self.process_minibatch(phase, feed_dict, losses, tensors)
            L = {k: L[k] + loss_dict[k] for k in L}
            T = {k: np.concatenate((T[k], tensor_dict[k])) if not T[k] is None else tensor_dict[k] for k in T}
        for k in L: L[k] /= self.batches_seen
        return L, T

    def get_metrics(self, tensors, metrics):
        met_dict = {}
        for m in metrics:
            m_fn = metrics[m]
            met_dict[m] = m_fn(tensors)
        return met_dict

    def create_res_str(self, epoch, L, M):
        ep_str = 'E{:d}: '.format(epoch) if not epoch is None else 'Test: '
        res_str = ep_str+ ', '.join(['{}:{:.3f}'.format(l, L[l]) for l in L]) \
                        + ', ' + ', '.join(['{}:{:.3f}'.format(m, M[m]) for m in M])
        return res_str

    def train(self, n_epochs, patience):
        min_val_loss, min_epoch = np.finfo(np.float32).max, -100
        losses = self.losses
        tensors = self.tensors
        metrics = self.metrics
        save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
        for epoch in range(n_epochs):
            #run train epoch
            train_L, train_T = self.process_epoch('train', losses, tensors, epoch)
            train_metrics = self.get_metrics(train_T, metrics)
            train_res_str = self.create_res_str(epoch, train_L, train_metrics)

            #run validation epoch
            valid_L, valid_T = self.process_epoch('valid', losses, tensors, epoch)
            valid_metrics = self.get_metrics(valid_T, metrics)
            valid_res_str = self.create_res_str(epoch, valid_L, valid_metrics)

            #print to command line
            msg = 'Train: {} | Valid: {}'.format(train_res_str, valid_res_str)
            print(msg)

            #do tensorboard tracking if required
            if not self.logs_path is None:
                summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)
                summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)
                summary = tf.Summary()
                for l in losses.keys():
                    summary.value.add(tag=l, simple_value=valid_L[l])
                for m in metrics:
                    summary.value.add(tag=m, simple_value=valid_metrics[m])
                summary_writer.add_summary(summary, epoch)

            #if min validation loss, checkpoint model
            l = valid_L['loss']
            if l < min_val_loss and epoch - min_epoch > 5:
                min_val_loss = l
                min_epoch = epoch
                if not self.checkpoint_path is None:
                    #checkpoint
                    save_path = self.saver.save(self.sess, save_path)
                    print('Best validation loss so far of {:.3f}, model saved to {}'.format(l, save_path))

            #if past patience, return for testing
            if epoch == n_epochs - 1 or epoch - min_epoch >= patience:
                print('Finished training: min validation loss was {:.3f} in epoch {:d}'.format(min_val_loss, min_epoch))
                break
        return save_path

    def restore(self, save_path):
        #restore checkpointed model
        self.saver.restore(self.sess, save_path)

    def test(self):
        losses = self.losses
        tensors = self.tensors
        metrics = self.metrics
        
        #run model on test set
        test_L, test_T = self.process_epoch('test', losses, tensors, 1000)
        test_metrics = self.get_metrics(test_T, metrics)
        test_res_str = self.create_res_str(None, test_L, test_metrics)

        #print to command line
        msg = 'Test: {}'.format(test_res_str)
        print(msg)
        
        #save results and tensors
        self.save_results(test_L, test_metrics)
        self.save_tensors(test_T)
        return

    def save_results(self, test_L, test_M):
        #record losses and metrics of interest on test set to a csv file
        testcsvname = os.path.join(self.results_path, 'test_results.csv')
        testcsv = open(testcsvname, 'w')
        # D is a dictionary of metrics: string to float
        for D in [test_L, test_M]:
            for k in D:
                s = '{},{:.7f}\n'.format(k, D[k])
                testcsv.write(s)
        testcsv.close()
        print('Metrics saved to {}'.format(testcsvname))

    def save_tensors(self, D):
        #save output tensors of interest as npz files
        for k in D:
            fname = os.path.join(self.results_path, '{}.npz'.format(k))
            np.savez(fname, X=D[k])
        print('Tensors saved to {}'.format(self.results_path))

def make_dir_if_not_exist(d):
    if not os.path.exists(d):
        os.makedirs(d)

