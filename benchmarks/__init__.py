import os
import time
import sys
import queue

import pandas as pd
import numpy as np

from multiprocessing import Process, Queue
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

from .data import categoric_to_numeric


class Dataset:
    """ Dataset representation. """

    def __init__(self, name, data, prepocess=None, isbinary=True):
        self._name = name
        self._data = data if prepocess is None else categoric_to_numeric(data)

        if isbinary:
            y = self._data['class']
            yun = sorted(y.unique())

            if -1 != yun[0]:
                self._data.loc[y == yun[0], 'class'] = -1
            if 1 != yun[1]:
                self._data.loc[y == yun[1], 'class'] = 1

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def x(self):
        x = self._data.filter(regex='^((?!class).)*$') # Remove `class` column.
        x = (x - x.min()) / (x.max() - x.min() + 1e-20) # min-max normalization.
        return x.values

    @property
    def y(self):
        return self._data['class'].values

    @property
    def dimension(self):
        try:
            return self._data.shape
        except:
            return 0


class Benchmarking:
    """ Class for benchmarking task. """

    def __init__(self, preproces=('tonumeric', )):
        self._datasets = {}
        self._datapaths = {}
        self._methods = {}
        self._process = {}

        self.prepocess = preproces

    @property
    def datasets(self):
        return self._datasets

    @property
    def datapaths(self):
        return list(self._datapaths.keys())

    def add_dataset(self, name, data, preprocess):
        """ Add a single dataset. """

        if not issubclass(data.__class__, pd.DataFrame):
            raise ValueError('`data` must be a pandas.DataFrame.')
        self._datasets[name] = Dataset(name, data, preprocess)

    def add_datapath(self, dpath, filetypes=('dat', 'data', 'csv'), sep=','):
        """ Add a directory with only datasets to be used in the benchmarkig. """

        if os.path.exists(dpath):
            self._datapaths[dpath] = (filetypes, sep)
        else:
            raise FileNotFoundError('Directory not found.')

    def add_method(self, name, method, params_init={}, params_fit={}):
        """ Add a method for the bechmarking. """

        if name in self._methods:
            raise Exception('')
        self._methods[name] = method, params_init, params_fit

    def _load(self):

        if len(self._datapaths) > 0:
            for dpath, (filetypes, sep) in self._datapaths.items():
                allfiles = os.listdir(dpath)
                datasets = [file for file in allfiles
                                if os.path.isfile(os.path.join(dpath, file)) and
                                file.split('.')[-1] in filetypes]

                for dnamef in datasets:
                    data = pd.read_csv(os.path.join(dpath, dnamef), sep=sep)
                    self.add_dataset(dnamef.split('.')[0], data, self.prepocess)

    def run(self, dname=None, wait=5, maxprocess=None, folds=10, pathres=None):
        """ Start the benchmarking. """
        self._load()

        if len(self._datasets) == 0:
            raise Exception('No one dataset was added.')

        maxprocess = os.cpu_count() if maxprocess is None else maxprocess
        cprocess = 0
        tprocess = 0

        running = {}
        ended = {}

        methodsnames = list(self._methods.keys())
        datasetsnames = list(self._datasets.keys())

        endedall = False
        cursorm = 0
        cursord = 0

        summary = pd.DataFrame()
        kfold = KFold(n_splits=folds)
        kfolds_index = {}

        while not endedall:

            while cursorm < len(methodsnames) and cprocess < maxprocess:

                mname = methodsnames[cursorm]
                method, params_init, params_fit = self._methods[mname]

                while cprocess < maxprocess:

                    dname = datasetsnames[cursord]
                    dataset = self._datasets[dname]

                    exec_name = (mname + '|' + dname).upper()

                    x = dataset.x
                    y = dataset.y

                    if dname not in kfolds_index:
                        kfolds_index[dname] = list(kfold.split(x))

                    q = Queue()
                    process = Process(target=self._run, name=exec_name,
                                      args=(method, x, y, params_init, params_fit,
                                            kfolds_index[dname], q))

                    running[exec_name] = process, q
                    cprocess += 1
                    tprocess += 1
                    cursord += 1

                    print(f'Starting process: #{cprocess}/{tprocess}-{process.name}')
                    process.start()

                    if cursord == len(datasetsnames): # Has started for all datasets to the current method.
                        cursord = 0
                        cursorm += 1

                        break

            time.sleep(wait) # Wait `wait` to eval if the each process has endend.

            for m in list(running.keys()):
                (p, q) = running[m]

                if not p.is_alive():

                    meth, dat = m.split('|')
                    res =  q.get()
                    ended[m] = (running.pop(m), res)
                    summary.loc[dat, meth] = res['result']
                    cprocess -= 1

                    # Save results in a csv file at current directory or, if given, `pathres`.
                    path = './'
                    if pathres is not None:
                        path = pathres
                    summary.to_csv(path + 'resultados.csv', index=True, index_label='Datasets')

            print('Running: ', len(running), ' | Ended: ', len(ended))
            if len(ended) == (len(self._methods) * len(self._datasets)):
                endedall = True

        print(summary)


    def _run(self, method, X, Y, params_init, params_fit, kfolds_index, q):
        """ Auxiliar method for multiprocessing. """
        res = []

        for (train_index, test_index) in kfolds_index:

            xtrain, ytrain = X[train_index, :], Y[train_index]
            xtest, ytest = X[test_index, :], Y[test_index]

            method_instance = method(**params_init)
            method_instance.fit(xtrain, ytrain, **params_fit)

            ypred_test = method_instance.predict(xtest)
            res.append(f1_score(ytest, ypred_test))

        res = np.mean(res)

        try:
            q.put({'result': res})
        except Exception as e:
            print(e)