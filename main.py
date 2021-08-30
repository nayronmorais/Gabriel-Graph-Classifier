import pandas as pd

from benchmarks import Benchmarking
from ggclassification.ggc import GGClassifier, GGCEnhanced

## Simple bechmarking.

if __name__ == '__main__':
    dpath1 = 'rawdatasets'

    # data = pd.read_csv('rawdatasets/bupa.data')
    # data2 = pd.read_csv('rawdatasets/tic-tac-toe.data')
    # data3 = pd.read_csv('rawdatasets/climate_model_simulation_crashes.data')

    bench = Benchmarking()
    bench.add_datapath(dpath1)
    # bench._load()
    # bench.add_dataset('bupa', data, preprocess=True)
    # bench.add_dataset('tic-tac-toe', data2, preprocess=True)
    # bench.add_dataset('climate', data3, preprocess=True)

    # bench.add_method('ggcnon', GGClassifier, params_fit=dict(remove_noise=False))
    bench.add_method('ggcnoise', GGClassifier, params_fit=dict(remove_noise=True))
    bench.add_method('ggc0', GGClassifier, params_init=dict(se_deep=0), params_fit=dict(remove_noise=True))
    bench.add_method('ggc1', GGClassifier, params_init=dict(se_deep=1), params_fit=dict(remove_noise=True))
    bench.add_method('ggc2', GGClassifier, params_init=dict(se_deep=2), params_fit=dict(remove_noise=True))
    bench.add_method('ggc3', GGClassifier, params_init=dict(se_deep=3), params_fit=dict(remove_noise=True))
    bench.add_method('ggc4', GGClassifier, params_init=dict(se_deep=4), params_fit=dict(remove_noise=True))
    bench.add_method('ggc5', GGClassifier, params_init=dict(se_deep=5), params_fit=dict(remove_noise=True))
    bench.add_method('ggce', GGCEnhanced, params_fit=dict(maxevals=100, verbose=False))
    bench.run(maxprocess=40)



