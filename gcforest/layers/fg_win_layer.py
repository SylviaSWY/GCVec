# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
#添加
#from ..estimators.base_estimator import predict_proba
from .base_layer import BaseLayer
#更改
from ..estimators.__init__ import get_estimator_kfold
from ..utils.metrics import accuracy_pb, accuracy_win_vote, accuracy_win_avg
from ..utils.win_utils import get_windows
from ..utils.debug_utils import repr_blobs_shape
from ..utils.log_utils import get_logger


#添加
import os, os.path as osp
import numpy as np

from ..utils.log_utils import get_logger
from ..utils.cache_utils import name2path

LOGGER = get_logger("gcforest.estimators.base_estimator")

def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)

class BaseClassifierWrapper(object):
    def __init__(self, name, est_class, est_args):
        """
        name: str)
            Used for debug and as the filename this model may be saved in the disk
        """
        self.name = name
        self.est_class = est_class
        self.est_args = est_args
        self.cache_suffix = ".pkl"
        self.est = None

    def _init_estimator(self):
        """
        You can re-implement this function when inherient this class
        """
        est = self.est_class(**self.est_args)
        return est

    def fit(self, X, y, cache_dir=None):
        """
        cache_dir(str): 
            if not None
                then if there is something in cache_dir, dont have fit the thing all over again
                otherwise, fit it and save to model cache 
        """
        LOGGER.debug("X_train.shape={}, y_train.shape={}".format(X.shape, y.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if self._is_cache_exists(cache_path):
            LOGGER.info("Find estimator from {} . skip process".format(cache_path))
            return
        est = self._init_estimator()
        self._fit(est, X, y)
        if cache_path is not None:
            # saved in disk
            LOGGER.info("Save estimator to {} ...".format(cache_path))
            check_dir(cache_path); 
            self._save_model_to_disk(est, cache_path)
        else:
            # keep in memory
            self.est = est

    def predict_proba(self, X, cache_dir=None, batch_size=None):
        LOGGER.debug("X.shape={}".format(X.shape))
        cache_path = self._cache_path(cache_dir)
        # cache
        if cache_path is not None:
            LOGGER.info("Load estimator from {} ...".format(cache_path))
            est = self._load_model_from_disk(cache_path)
            LOGGER.info("done ...")
        else:
            est = self.est
        batch_size = batch_size or self._default_predict_batch_size(est, X)
        if batch_size > 0:
            y_proba = self._batch_predict_proba(est, X, batch_size)
        else:
            y_proba = self._predict_proba(est, X)
        LOGGER.debug("y_proba.shape={}".format(y_proba.shape))
        return y_proba

    def _cache_path(self, cache_dir):
        if cache_dir is None:
            return None
        return osp.join(cache_dir, name2path(self.name) + self.cache_suffix)

    def _is_cache_exists(self, cache_path):
        return cache_path is not None and osp.exists(cache_path)

    def _batch_predict_proba(self, est, X, batch_size):
        LOGGER.debug("X.shape={}, batch_size={}".format(X.shape, batch_size))
        if hasattr(est, "verbose"):
            verbose_backup = est.verbose
            est.verbose = 0
        n_datas = X.shape[0]
        y_pred_proba = None
        for j in range(0, n_datas, batch_size):
            LOGGER.info("[progress][batch_size={}] ({}/{})".format(batch_size, j, n_datas))
            y_cur = self._predict_proba(est, X[j:j+batch_size])
            if j == 0:
                n_classes = y_cur.shape[1]
                y_pred_proba = np.empty((n_datas, n_classes), dtype=np.float32)
            y_pred_proba[j:j+batch_size,:] = y_cur
        if hasattr(est, "verbose"):
            est.verbose = verbose_backup
        return y_pred_proba

    def _load_model_from_disk(self, cache_path):
        raise NotImplementedError()

    def _save_model_to_disk(self, est, cache_path):
        raise NotImplementedError()

    def _default_predict_batch_size(self, est, X):
        """
        You can re-implement this function when inherient this class 

        Return
        ------
        predict_batch_size (int): default=0
            if = 0,  predict_proba without batches
            if > 0, then predict_proba without baches
            sklearn predict_proba is not so inefficient, has to do this
        """
        return 0

    def _fit(self, est, X, y):
        est.fit(X, y)

    def _predict_proba(self, est, X):
        return est.predict_proba(X)




#
LOGGER = get_logger("gcforest.layers.fg_win_layer")

#CV_POLICYS = ["data", "win"]
#CV_POLICYS = ["data"]

class FGWinLayer(BaseLayer):
    def __init__(self, layer_config, data_cache):
        """
        est_config (dict): 
            estimator的config
        win_x, win_y, stride_x, stride_y, pad_x, pad_y (int): 
            configs for windows 
        n_folds(int): default=1
             1 means do not use k-fold
        n_classes (int):
             
        """
        super(FGWinLayer, self).__init__(layer_config, data_cache)
        # estimator
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.win_x = self.get_value("win_x", None, int, required=True)
        self.win_y = self.get_value("win_y", None, int, required=True)
        self.stride_x = self.get_value("stride_x", 1, int)
        self.stride_y = self.get_value("stride_y", 1, int)
        self.pad_x = self.get_value("pad_x", 0, int)
        self.pad_y = self.get_value("pad_y", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        #self.cv_policy = layer_config.get("cv_policy", "data")
        #assert(self.cv_policy in CV_POLICYS)
        assert len(self.bottom_names) >= 2
        assert len(self.est_configs) == len(self.top_names), "Each estimator shoud produce one unique top"
        # self.eval_metrics = [("predict", accuracy_pb), ("vote", accuracy_win_vote), ("avg", accuracy_win_avg)]
        self.eval_metrics = [("predict", accuracy_pb), ("avg", accuracy_win_avg)]
        self.estimator1d = [None for ei in range(len(self.est_configs))]

    def _init_estimators(self, ei, random_state):
        """
        ei (int): estimator index
        """
        top_name = self.top_names[ei]
        est_args = self.est_configs[ei].copy()
        est_name ="{}/{}_folds".format(top_name, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        random_state = (random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def fit_transform(self, train_config):
        LOGGER.info("[data][{}], bottoms={}, tops={}".format(self.name, self.bottom_names, self.top_names))
        phases = train_config.phases
        X_train_win, y_train_win = None, None
        test_sets = None

        for ti, top_name in enumerate(self.top_names):
            LOGGER.info("[progress][{}] ti={}/{}, top_name={}".format(self.name, ti, len(self.top_names), top_name))
            # check top cache
            if np.all(self.check_top_cache(phases, ti)):
                LOGGER.info("[data][{}] all top cache exists. skip progress".format(self.name))
                continue

            # init X, y, n_classes
            if X_train_win is None:
                for pi, phase in enumerate(phases):
                    bottoms = self.data_cache.gets(phase, self.bottom_names)
                    LOGGER.info('[data][{},{}] bottoms.shape={}'.format(self.name, phase, repr_blobs_shape(bottoms)))
                    X, y = np.concatenate(bottoms[:-1], axis=1), bottoms[-1]
                    # n x n_windows x channel
                    X_win = get_windows(X, self.win_x, self.win_y, self.stride_x, self.stride_y, self.pad_x, self.pad_y)
                    _, nh, nw, _ = X_win.shape
                    X_win = X_win.reshape((X_win.shape[0], -1, X_win.shape[-1]))
                    y_win = y[:,np.newaxis].repeat(X_win.shape[1], axis=1)
                    if pi == 0:
                        assert self.n_classes == len(np.unique(y)), \
                                "n_classes={}, len(unique(y))={}".format(self.n_classes, len(np.unique(y)))
                        X_train_win, y_train_win = X_win, y_win
                    else:
                        test_sets = [("test", X_win, y_win), ]

            # fit
            est = self._init_estimators(ti, train_config.random_state)
            y_probas = est.fit_transform(X_train_win, y_train_win, y_train_win[:,0], cache_dir=train_config.model_cache_dir, 
                    test_sets = test_sets, eval_metrics=self.eval_metrics,
                    keep_model_in_mem=train_config.keep_model_in_mem)

            for pi, phase in enumerate(phases):
                y_proba = y_probas[pi].reshape((-1, nh, nw, self.n_classes)).transpose((0, 3, 1, 2))
                LOGGER.info('[data][{},{}] tops[{}].shape={}'.format(self.name, phase, ti, y_proba.shape))
                self.data_cache.update(phase, self.top_names[ti], y_proba)
            if train_config.keep_model_in_mem:
                self.estimator1d[ti] = est
    
    def transform(self):
        phase = "test"
        for ti, top_name in enumerate(self.top_names):
            LOGGER.info("[progress][{}] ti={}/{}, top_name={}".format(self.name, ti, len(self.top_names), top_name))

            bottoms = self.data_cache.gets(phase, self.bottom_names[:-1])
            LOGGER.info('[data][{},{}] bottoms.shape={}'.format(self.name, phase, repr_blobs_shape(bottoms)))
            X = np.concatenate(bottoms, axis=1)
            # n x n_windows x channel
            X_win = get_windows(X, self.win_x, self.win_y, self.stride_x, self.stride_y, self.pad_x, self.pad_y)
            _, nh, nw, _ = X_win.shape
            X_win = X_win.reshape((X_win.shape[0], -1, X_win.shape[-1]))

            est = self.estimator1d[ti]
            y_proba = est.predict_proba(X_win)
            y_proba = y_proba.reshape((-1, nh, nw, self.n_classes)).transpose((0, 3, 1, 2))
            LOGGER.info('[data][{},{}] tops[{}].shape={}'.format(self.name, phase, ti, y_proba.shape))
            self.data_cache.update(phase, self.top_names[ti], y_proba)

    def score(self):
        eval_metrics = [("predict", accuracy_pb), ("avg", accuracy_win_avg)]
        for ti, top_name in enumerate(self.top_names):
            for phase in ["train", "test"]:
                y = self.data_cache.get(phase, self.bottom_names[-1])
                y_proba = self.data_cache.get(phase, top_name)
                y_proba = y_proba.transpose((0,2,3,1))
                y_proba = y_proba.reshape((y_proba.shape[0], -1, y_proba.shape[3]))
                y = y[:,np.newaxis].repeat(y_proba.shape[1], axis=1)
                for eval_name, eval_metric in eval_metrics:
                    acc = eval_metric(y, y_proba)
                    LOGGER.info("Accuracy({}.{}.{})={:.2f}%".format(top_name, phase, eval_name, acc*100))
