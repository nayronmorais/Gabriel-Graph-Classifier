"""
Author: Nayron Morais <nayronmorais@gmail.com>.
References:
    -
    -
    -
"""

import os

import networkx as nx
import numpy as np

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import f1_score, accuracy_score


# For the plot functions.
_Z_ORDER_V = 10
_Z_ORDER_SE = _Z_ORDER_V - 1
_Z_ORDER_SSV = _Z_ORDER_V + 1

_IS_NOT_FITTED_MSG = 'The model must be fitted before this call.'


def _calculate_dist_matrix(X):
    (nsamples, ndim) = X.shape
    dist_matrix = -np.ones(shape=(nsamples, nsamples), dtype=np.float64)

    # Calculate distance matrix.
    for i in range(nsamples):
        xi = X[i, :].reshape(1, ndim)
        dist_matrix[i, :] = dist_matrix[:, i] = np.linalg.norm(xi - X, axis=1, ord=2) ** 4
        dist_matrix[i, i] = np.inf

    return dist_matrix


class GabrielGraph(nx.Graph):
    """
    Gabriel Graph (GG)
    """

    def __init__(self, X, Y=None):

        super().__init__()
        self._build(X, Y)

    def _build(self, X, Y):

        (nsamples, _) = X.shape

        # Calculate distance matrix.
        self.dist_matrix = _calculate_dist_matrix(X)

        ## Gabriel Graph definition.
        # Nodes
        nodes = np.arange(nsamples)
        if Y is not None and hasattr(Y, '__iter__'):
            [self.add_node(node, y=c) for (node, c) in zip(nodes, Y)]
        else:
            self.add_nodes_from(nodes)

        # Adjacency
        for vi in range(nsamples - 1):
            for vj in range(vi + 1, nsamples):
                if self.is_continguous(vi, vj):
                    self.add_edge(vi, vj)

    def _rebuild(self, Y=None):

        self.clear()

        ## Gabriel Graph definition.
        # Nodes
        nsamples = self.dist_matrix.shape[0]
        nodes = np.arange(nsamples)
        if Y is not None and hasattr(Y, '__iter__'):
            [self.add_node(node, y=c) for (node, c) in zip(nodes, Y)]
        else:
            self.add_nodes_from(nodes)

        # Adjacency
        for vi in range(nsamples - 1):
            for vj in range(vi + 1, nsamples):
                if self.is_continguous(vi, vj):
                    self.add_edge(vi, vj)

    def delete_nodes(self, nodes, Y=None):
        """ Remove the vertices in `nodes` from the graph. """

        mask = np.ones(shape=len(self.nodes), dtype=bool)
        mask[nodes] = False

        dist_matrix = self.dist_matrix[mask, :]
        self.dist_matrix = dist_matrix[:, mask]

        if Y is not None:
            Y = Y[mask]

        self._rebuild(Y)

        return mask

    def is_continguous(self, vi, vj):
        """ Eval if the samples `vi` and `vj` are adjacent. """

        dvivj = self.dist_matrix[vi, vj]
        dvivjdvk = np.min(self.dist_matrix[vi, :] + self.dist_matrix[vj, :])

        return  dvivj <= dvivjdvk

    @staticmethod
    def plot(gg, X, points_color=None, point_scale=10**2, edge_color='k', edge_width=.5):
        """ Plot the graph. """

        from matplotlib import pyplot as plt

        if X.ndim != 2:
            raise ValueError("The visualization it's only in 2d data.")

        fig = plt.gcf() # Get current figure or create it.
        ax = fig.add_subplot(111)

        for (i, j) in gg.edges:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]],
                    color=edge_color, lw=edge_width, zorder=_Z_ORDER_SE)
        ax.scatter(*X.T, s=point_scale, zorder=_Z_ORDER_V,
                   c='lightgray' if points_color is None else points_color,
                   linewidths=0.5,
                   edgecolors='k')

        return ax


class GGClassifier:
    """
    Gabriel Graph Classifier (GGC)

    """

    def __init__(self, se_deep=-1):

        self._middle_points = None
        self._w = None
        self._bias = None
        self._ssv = None
        self._se = None
        self._gg = None
        self._labels = None
        self._is_fitted = False
        self._metrics = None
        self._se_deep = se_deep

    @property
    def ssv(self):
        """ Structural Support Vectors (SSV). """
        if self._ssv is None:
            self._se, self._ssv  = self._find_se()
        return self._ssv

    @property
    def se(self):
        """ Support Edges. """
        if self._se is None:
            self._se, self._ssv  = self._find_se()
        return self._se

    @property
    def middle_points(self):
        """ Middle points of each SE. """
        return self._middle_points

    @property
    def bias(self):
        """ Hyperplanes's bias. """
        return self._bias

    @property
    def w(self):
        """ Hyperplanes's inclination. """
        return self._w

    @property
    def gg(self):
        """ Gabriel Graph. """
        return self._gg

    def _find_se(self):
        """ Select the SEs based on the criteria defined in `se_deep`. """

        if self._gg is None:
            raise Exception(_IS_NOT_FITTED_MSG)

        se = []
        ssv = []
        if self._se_deep <= 0:

            for (vi, vj) in self._gg.edges:
                node_vi, node_vj = self._gg.nodes[vi], self._gg.nodes[vj]
                yi, yj = node_vi['y'], node_vj['y']
                if yi != yj:
                    if yi == self._labels[0]: # Ensures order (negative class first).
                        se.append((vi, vj))
                    else:
                        se.append((vj, vi))
                    ssv.append(vi)
                    ssv.append(vj)
        elif self._se_deep > 0:
             for (vi, vj) in self._gg.edges:
                node_vi, node_vj = self._gg.nodes[vi], self._gg.nodes[vj]
                yi, yj = node_vi['y'], node_vj['y']

                if yi != yj:

                    def walk(y_target, v_origin, v_new, v_previous, v_current, current_deep):
                        """ 'Walk' on adjacency vertices in a recursive way. """
                        v_previous.append(v_current)
                        adjs = np.array(list(self._gg.adj[v_current].keys()))

                        if len(adjs) > 0 and  current_deep >= 0  and current_deep < self._se_deep:
                            adjs = adjs[np.argsort(self._gg.dist_matrix[v_current, adjs])]

                            v_next = None
                            for v in adjs:
                                if v not in v_previous:
                                    node = self._gg.nodes[v]
                                    if node['y'] == y_target:
                                        v_new = v
                                    v_next = v
                                    break

                            if v_next is not None:
                                return walk(y_target, v_current, v_new, v_previous, v_next, current_deep + 1)

                            # print(current_deep)
                            if v_next is None:
                                return walk(y_target, v_origin, v_new, v_previous, v_origin, current_deep - 1)

                        return v_new

                    adivi = walk(yi, vi, vi, [], vi, 0)
                    adjvj = walk(yj, vj, vj, [], vj, 0)

                    if yi == self._labels[0]: # Ensures order (negative class first).
                        se.append((adivi, adjvj))
                    else:
                        se.append((adjvj, adivi))

                    ssv.append(adivi)
                    ssv.append(adjvj)


        return np.array(se), np.unique(ssv)

    def fit(self, X, Y, remove_noise=True):
        """ Fit the model. """

        y = np.unique(Y)
        if y.size != 2:
            raise NotImplementedError("This implementation it's only for binary classification.")

        # Remove duplicated samples.
        idxuniq = np.unique([np.nonzero(np.equal(x, X).all(axis=1))[0][0] for x in X])
        X = X[idxuniq, :]
        Y = Y[idxuniq]

        self._gg = GabrielGraph(X, Y)
        self._labels = y

        if remove_noise:
            self.noise_nodes, X, Y = self.filter_noise(X, Y, return_new_XY=True)

        self._middle_points, self._w, self._bias = self._calculate_model_params(X)
        self._is_fitted = True

        ypred = self.predict(X)
        self._metrics = {'f1': f1_score(Y, ypred), 'acc': accuracy_score(Y, ypred)}

        return self, X, Y

    def predict(self, X):
        """ Assign a class to samples in `X`. """

        if not self._is_fitted:
            raise Exception(_IS_NOT_FITTED_MSG)

        nsamples = X.shape[0]
        labels = np.zeros(shape=nsamples, dtype=np.int8)

        for t in range(nsamples):

            x = X[t, :][np.newaxis, :]

            # Projection on Local hyperplanes and sign function.
            hk = np.dot(x, self._w.T).T - self._bias
            geqzero = hk >= 0
            hk[geqzero] = 1
            hk[~geqzero] = -1

            # x's distance from each hiperplane
            ck = np.linalg.norm(x - self._middle_points, ord=2, axis=1, keepdims=True)

            # Computing weights
            ck = np.exp(-(np.max(ck) ** 2 / (ck + 1e-20)))
            ck = 1 / (ck + 1e-20)
            ck = ck / ck.sum()

            # Final output by applying sign function in the local hyperplanes combination
            h = (hk * ck).sum()
            if h >= 0:
                labels[t] = self._labels[1]
            else:
                labels[t] = self._labels[0]

        return labels

    def _calculate_filter_noise_params(self):

       qsn, qsp, nn, npp = 0, 0, 0, 0
       gg = self._gg

       for vid in self._gg.nodes:
            node = gg.nodes[vid]
            y = node['y']
            adj = gg.adj[vid]

            d = len(adj)
            dhat = 0
            for vid_adj in adj:
                if gg.nodes[vid_adj]['y'] == y:
                    dhat += 1
            q = dhat / d
            node['q'] = q

            if y == self._labels[0]:
                qsn += q
                nn += 1
            else:
                qsp += q
                npp += 1


       return (qsn / nn, qsp / npp)


    def _calculate_model_params(self, X):

        middle_points = []
        w = []
        bias = []

        for (vi, vj) in self.se:

            diff = X[vj, :] - X[vi, :]
            middp = (X[vj, :] + X[vi, :]) / 2

            middle_points.append(middp)
            w.append(diff)
            bias.append(np.dot(middp, diff.T))

        return (
                np.array(middle_points),
                np.array(w),
                np.array(bias)[:, np.newaxis]
            )

    def filter_noise(self, X, Y, return_new_XY=True):
        """ Remove outliers based on the value defined in `se_deep`. """

        self.tn, self.tp = tn, tp = self._calculate_filter_noise_params()
        toremove = []

        for vid in self._gg.nodes:
            node = self._gg.nodes[vid]

            if node['y'] == self._labels[0]:
                if node['q'] < tn and self._se_deep < 0 or node['q'] == 0 and self._se_deep >= 0:
                    toremove.append(vid)
            else:
                if node['q'] < tp and self._se_deep < 0 or node['q'] == 0 and self._se_deep >= 0:
                    toremove.append(vid)

        out = (toremove, )
        mask = slice(None)
        if len(toremove) > 0:
            mask = self._gg.delete_nodes(toremove, Y)
            self._se = self._ssv = None

        if return_new_XY:
            out += (X[mask, :], Y[mask])

        return out


    def score(self, metric=None):
        """ Return the f1-score or accuracy. """
        try:
            return self._metrics[metric]
        except:
            return self._metrics

    def plot(self, X):

        # Vertices with color by class
        idx = [0 if self._gg.nodes[node]['y'] == self._labels[0] else 1 for node in self._gg.nodes]
        colors = np.array(['white', 'black'])[idx]
        ax = GabrielGraph.plot(self._gg, X, points_color=colors)

        # Disciminator with middle points
        ax.scatter(*self._middle_points.T, s=10 ** 2, color='blue', zorder=_Z_ORDER_SSV, label='Middle point')
        x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10)
        for i, (w, b, m, se) in enumerate(zip(self.w, self.bias, self.middle_points, self.se)):

            if w[1] != 0:
                y = (--w[0]/ -w[1] * x) - (b / -w[1])
            else:
                y = 0 *x - b
            ax.plot(x, y, lw=3, color='green', zorder=_Z_ORDER_SE, label=f'Hyperplane {i + 1}')

        # SEs
        for i, se in enumerate(self.se):
            ax.plot(*X[se, :].T, lw=2, color='orange', zorder=_Z_ORDER_V, label=f'SE {i + 1}')

        # SSVs
        ax.scatter(*X[self.ssv, :].T, color='blue', marker='X', s=7 ** 2, zorder=_Z_ORDER_SSV, label='SSV')

        ax.legend(fontsize=8, framealpha=0.4, borderpad=1, handlelength=1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(ls=':', lw=0.75, alpha=0.4)

        ax.set_xlabel('$x_1$', fontsize=10)
        ax.set_ylabel('$x_2$', fontsize=10)
        plt.setp(ax.get_xticklabels(), fontsize=9)
        plt.setp(ax.get_yticklabels(), fontsize=9)

        return ax


class GGCEnhanced(GGClassifier):
    """
    GGClassification with Sequential Model-Based Optimization (SMBO).
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, Y, maxevals=100, loss_threshold=1e-5, verbose=True):

        y = np.unique(Y)
        if y.size != 2:
            raise NotImplementedError("This implementation it's only for binary classification.")

        # Remove duplicated samples.
        idxuniq = np.unique([np.nonzero(np.equal(x, X).all(axis=1))[0][0] for x in X])
        X = X[idxuniq, :]
        Y = Y[idxuniq]

        if self._gg is None:
            self._gg = GabrielGraph(X, Y)
            self._labels = y
            self._se_backup = self.se

        self.X, self.Y = X, Y
        self._is_fitted = True

        params = {f'se{i}': hp.randint(f'se{i}', 2) for i in range(len(self.se))}
        trials = Trials()
        best = fmin(self._obj,
                    space=hp.choice('ses', [params, ]),
                    algo=tpe.suggest,
                    max_evals=maxevals,
                    trials=trials,
                    max_queue_len=os.cpu_count(),
                    loss_threshold=loss_threshold,
                    verbose=verbose)

        self.best, self.trials = best, trials

        bestcomb = np.array([best[f'se{i}'] for i in range(len(self._se_backup))]).astype(bool)
        self._se = self._se_backup[bestcomb, :]
        self._ssv =  np.unique(self._se.flatten())
        self._middle_points, self._w, self._bias = self._calculate_model_params(self.X)

        ypred = self.predict(X)
        self._metrics = {'f1': f1_score(Y, ypred), 'acc': accuracy_score(Y, ypred)}

        return self

    def _fit_obj(self, se):

        self._se = self._se_backup[se, :]
        self._middle_points, self._w, self._bias = self._calculate_model_params(self.X)


    def _obj(self, x):

        seidxs = np.array([x[f'se{i}'] for i in range(len(self._se_backup))]).astype(bool)

        if seidxs.sum() > 0:

            self._fit_obj(seidxs)
            ypred = self.predict(self.X)
            loss = 1 - f1_score(self.Y, ypred)
        else:
            loss = np.inf

        return {
                'loss': loss,
                'status': STATUS_OK
            }


## Toy example using (partially) iris dataset.
if __name__ == '__main__':

    from sklearn.datasets import load_iris
    from matplotlib import pyplot as plt

    data = load_iris(as_frame=False)

    X = data['data'][:, [0 , 3]]
    YD = data['target']

    X = ((X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)))
    YD[(YD == 2)] = 1
    YD[(YD == 0)] = -1

    plt.figure(figsize=(10, 8))
    ggc1, X1, _  = GGClassifier().fit(X, YD, remove_noise=False)
    ax1 = ggc1.plot(X1)
    ax1.set_title("With 'noise'")
    print('Summary GGC with noise: ', ggc1.score())

    plt.figure(figsize=(10, 8))
    ggc2, X1, _ = GGClassifier(se_deep=-1).fit(X, YD, remove_noise=True)
    XA = np.delete(X1, ggc2.noise_nodes, axis=0)
    ax2 = ggc2.plot(X1)
    ax2.set_title("Without 'noise'")
    print('Summary GGC without noise: ', ggc2.score())

    plt.figure(figsize=(10, 8))
    ggc3, X1, _ = GGClassifier(se_deep=0).fit(X, YD, remove_noise=True)
    XA = np.delete(X1, ggc3.noise_nodes, axis=0)
    ax3 = ggc3.plot(X1)
    ax3.set_title("Without 'noise' eq zero")
    print('Summary GGC without noise eq zero: ', ggc3.score())

    plt.figure(figsize=(10, 8))
    ggc4, X1, Y1 = GGClassifier(se_deep=2).fit(X, YD, remove_noise=True)
    XA = np.delete(X1, ggc4.noise_nodes, axis=0)
    ax4 = ggc4.plot(XA)
    ax4.set_title("Without 'noise' eq zero and deep 2")
    print('Summary GGC without noise eq zero and deep eq 2: ', ggc4.score())

    plt.figure(figsize=(10, 8))
    ggce = GGCEnhanced().fit(X1, Y1)
    ax5 = ggce.plot(X1)
    ax5.set_title("Without 'noise' eq zero and deep 2")
    print('Summary GGC without noise eq zero and deep eq 2: ', ggce.score())


