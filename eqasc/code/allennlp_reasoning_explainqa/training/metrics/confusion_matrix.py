from typing import Tuple

from overrides import overrides

# from allennlp.tools import squad_eval
from allennlp.training.metrics.metric import Metric

import os
import json
import copy
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import numpy as np
import torch
import random
from typing import Optional
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from sklearn.metrics import roc_curve


@Metric.register("confusion_matrix")
class ConfusionMatrix(Metric):

    def __init__(self, ) -> None:
        self._predictions = []
        self._gt = []
        self._count = 0

    @overrides
    def __call__(self, prediction, ground_truth, idx=None):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        self._predictions.append(prediction)
        self._gt.append(ground_truth)

    @overrides
    def get_metric(self, reset: bool = False):
        self.cm = confusion_matrix(self._gt, self._predictions)
        # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
        # sklearn.metrics.f1_score(y_true, y_pred, pos_label=1)
        return self.cm

    @overrides
    def reset(self):
        self._predictions = {}
        self._gt = {}
        self._count = 0

    def __str__(self):
        return str(self.cm)  # f"CocovalsMeasures(em={self._total_em}, f1={self._total_f1})"

import matplotlib.pyplot as plt

@Metric.register("f1custom")
class F1MeasureCustom(Metric):

    def __init__(self, pos_label=1) -> None:
        self._predictions = [ ]
        self._gt = [ ]
        self._pos_label = pos_label
        self._probs = [ ]
        self.f1 = 0.0

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 probs: torch.Tensor = None,
                 mask: Optional[ torch.Tensor ] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        num_classes = predictions.size(-1)
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError("gold_labels must have dimension == predictions.size() - 1 but "
                                     "found tensor of shape: {}".format(predictions.size()))
        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to Categorical Accuracy contains an id >= {}, "
                                     "the number of classes.".format(num_classes))
        predictions = predictions.view((-1, num_classes))
        gold_labels = gold_labels.view(-1).long()
        top_k = predictions.max(-1)[ 1 ].unsqueeze(-1)
        # For each row check if index pointed by gold_label is was 1 or not (among max scored classes)
        correct = top_k.eq(gold_labels.unsqueeze(-1)).float()
        self._predictions.extend(top_k.data.cpu().numpy().reshape(-1))
        self._gt.extend(gold_labels.data.cpu().numpy())
        if probs is not None:
            probs_pos_class = probs[ :, self._pos_label ]
            self._probs.extend(list(probs_pos_class.data.cpu().numpy()))  #

    @overrides
    def get_metric(self, reset: bool = False,
                   given_thresh:float = None):  # -> Dict[str,Float]:
        self.f1 = sklearn.metrics.f1_score(self._gt, self._predictions, pos_label=self._pos_label)
        self.prec, self.rec, _, _ = precision_recall_fscore_support(self._gt, self._predictions,
                                                                    pos_label=self._pos_label,
                                                                    labels=[ self._pos_label ])
        # print("self.prec, self.rec, = ", self.prec, self.rec, " f1 = ", self.f1)
        prediction_distribution = dict(Counter(self._predictions))
        gt_distribution = dict(Counter(self._gt))
        # print("gt_distribution = ", gt_distribution)
        auc_roc = -999
        f1_scores_max = -999
        threshold_max = -999
        f1_score_given_thresh = None

        if reset: # to save computation time
            if len(self._probs) > 0:
                fpr, tpr, thresholds = roc_curve(self._gt, self._probs)
                f1_scores = [ ]
                for thresh in thresholds:
                    f1_scores.append(sklearn.metrics.f1_score(self._gt,
                                                              [ 1 if m > thresh else 0 for m in self._probs ]))
                f1_scores = np.array(f1_scores)
                f1_scores_max = np.max(f1_scores)
                threshold_max = thresholds[ np.argmax(f1_scores) ]
                if len(gt_distribution) > 1:  # self._pos_label in gt_distribution:
                    auc_roc = sklearn.metrics.roc_auc_score(self._gt, self._probs)
                    # plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange',
                             lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
                    plt.plot([ 0, 1 ], [ 0, 1 ], color='navy', lw=lw, linestyle='--')
                    plt.xlim([ 0.0, 1.0 ])
                    plt.ylim([ 0.0, 1.05 ])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver operating characteristic example')
                    plt.legend(loc="lower right")
                    # plt.show()
                    # plt.savefig('roc_plot.png')
                else:
                    auc_roc = 0
            if given_thresh is not None:
                f1_score_given_thresh = sklearn.metrics.f1_score(self._gt,[1 if m > given_thresh else 0 for m in self._probs])
                prediction_distribution_thresh = dict(Counter([1 if m > given_thresh else 0 for m in self._probs]))
        if reset:
            self.reset()
        ret = {'f1': self.f1, 'rec': self.rec[ 0 ], 'prec': self.prec[ 0 ], 'auc_roc': float(auc_roc),
               'f1_scores_max': float(f1_scores_max), 'threshold_max': float(threshold_max)}
        ret.update({'f1_predlabel_' + str(k): v for k, v in prediction_distribution.items()})
        ret.update({'f1_gtlabel_' + str(k): v for k, v in gt_distribution.items()})
        if f1_score_given_thresh is not None:
            ret.update( {'f1_score_given_thresh': f1_score_given_thresh} )
            ret.update({'f1_thresh_predlabel_' + str(k): v for k, v in prediction_distribution_thresh.items()})
        return ret

    @overrides
    def reset(self):
        self._predictions = [ ]
        self._gt = [ ]
        self._probs = [ ]
        self.f1 = 0.0

    def __str__(self):
        return str(self.f1)



@Metric.register("f1custom_retrievaleval")
class F1MeasureCustomRetrievalEval():

    def __init__(self, pos_label=1, save_fig=False) -> None:
        self._predictions = [ ]
        self._gt = [ ]
        self._pos_label = pos_label
        self._probs = [ ]
        self.save_fig = save_fig

    def __call__(self,
                 label, score):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        # print("[F1MeasureCustomRetrievalEval] label, score = ", label, score)
        self._gt.append(label)
        self._probs.append(score)


    # def get_metric(self, figname='roc_plot.png'):  # -> Dict[str,Float]:
    def get_metric(self, reset: bool = False,
                   figname='roc_plot.png',
                   given_thresh=None):  # -> Dict[str,Float]:
        probs = np.array(self._probs)
        probs = (probs - probs.min())/ (probs.max() - probs.min())
        gt = np.array(self._gt)

        auc_roc = 0
        threshold_max = None
        f1_score_given_thresh = None
        if reset and len(probs) > 0:
            fpr, tpr, thresholds = roc_curve(gt, probs)
            f1_scores = [ ]
            for thresh in thresholds:
                f1_scores.append(sklearn.metrics.f1_score(gt,
                                                          [ 1 if m > thresh else 0 for m in probs ]))
            f1_scores = np.array(f1_scores)
            f1_scores_max = np.max(f1_scores)
            threshold_max = thresholds[ np.argmax(f1_scores) ]
            auc_roc = sklearn.metrics.roc_auc_score(gt, probs)
            lw = 2
            if self.save_fig:
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
                plt.plot([ 0, 1 ], [ 0, 1 ], color='navy', lw=lw, linestyle='--')
                plt.xlim([ 0.0, 1.0 ])
                plt.ylim([ 0.0, 1.05 ])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                # plt.show()
                # plt.savefig(figname)
            if given_thresh is not None:
                f1_score_given_thresh = sklearn.metrics.f1_score(gt,[1 if m > given_thresh else 0 for m in probs])
        else:
            auc_roc = 0
            f1_scores_max = 0
        # print("self._gt Counter ===> ", Counter(gt))
        if reset:
            self.reset()
        return {'auc_roc':auc_roc, 'f1_scores_max':f1_scores_max, 'threshold_max':threshold_max, 'f1_score_given_thresh':f1_score_given_thresh}

    def reset(self):
        self._gt = []
        self._probs = []