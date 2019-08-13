import numpy as np
import matplotlib.pyplot as plt
import itertools
from subprocess import Popen, PIPE
cmd = ' uname -n'
proc = Popen(cmd, stdin=PIPE, stdout=PIPE, shell=True)
stdout, stderr = proc.communicate()
if not stdout.split('.')[0] == 'ramawks80':
    plt.switch_backend('agg')


class ConfusionMeter:

    def __init__(self, labels=[], normalize=False, save_path=''):
        self.save_path = save_path
        self.labels = labels
        self.num_classes = len(labels)
        self.confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        self.normalize = normalize

    def update(self, predictions, labels):
        assert predictions.shape == labels.shape
        for i in range(predictions.size(0)):
            self.confusion_matrix[labels[i].item()][predictions[i].item()] += 1

    def get_result(self):
        if self.normalize:
            return self.confusion_matrix / self.confusion_matrix.sum(1).clip(min=1e-10)[:, None]
        else:
            return self.confusion_matrix

    def save_confusion_matrix(self):
        final_confusion = self.get_result()
        plt.figure()
        plt.imshow(final_confusion, interpolation='nearest', cmap=plt.cm.YlOrRd)
        plt.colorbar()
        tick_marks = np.arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=90, fontsize=8)
        plt.yticks(tick_marks, self.labels, fontsize=8)
        fmt = '.2f' if self.normalize else 'd'
        thresh = final_confusion.mean()
        for i, j in itertools.product(range(self.num_classes), range(self.num_classes)):
            plt.text(j, i, format(final_confusion[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if final_confusion[i, j] > thresh else "black", fontsize=8)
        plt.ylabel('True label', fontsize=10)
        plt.xlabel('Predicted label', fontsize=10)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=600)
        plt.close()
