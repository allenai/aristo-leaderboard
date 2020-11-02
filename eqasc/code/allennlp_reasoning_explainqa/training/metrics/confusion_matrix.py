import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from allennlp.training.metrics.metric import Metric
from sklearn.metrics import roc_curve


@Metric.register("f1custom_retrievaleval")
class F1MeasureCustomRetrievalEval():

    def __init__(self, pos_label=1, save_fig=False) -> None:
        self._predictions = []
        self._gt = []
        self._pos_label = pos_label
        self._probs = []
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
        self._gt.append(label)
        self._probs.append(score)

    def get_metric(self, reset: bool = False,
                   given_thresh=None):  # -> Dict[str,Float]:
        probs = np.array(self._probs)
        probs = (probs - probs.min()) / (probs.max() - probs.min())
        gt = np.array(self._gt)

        threshold_max = None
        f1_score_given_thresh = None
        if reset and len(probs) > 0:
            fpr, tpr, thresholds = roc_curve(gt, probs)
            f1_scores = []
            for thresh in thresholds:
                f1_scores.append(sklearn.metrics.f1_score(gt,
                                                          [1 if m > thresh else 0 for m in probs]))
            f1_scores = np.array(f1_scores)
            f1_scores_max = np.max(f1_scores)
            threshold_max = thresholds[np.argmax(f1_scores)]
            auc_roc = sklearn.metrics.roc_auc_score(gt, probs)
            lw = 2
            if self.save_fig:
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % auc_roc)
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
            if given_thresh is not None:
                f1_score_given_thresh = sklearn.metrics.f1_score(gt, [1 if m > given_thresh else 0 for m in probs])
        else:
            auc_roc = 0
            f1_scores_max = 0

        if reset:
            self.reset()
        return {'auc_roc': auc_roc, 'f1_scores_max': f1_scores_max, 'threshold_max': threshold_max, 'f1_score_given_thresh': f1_score_given_thresh}

    def reset(self):
        self._gt = []
        self._probs = []
