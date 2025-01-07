import torch
import torch.optim as optim
from sklearn.metrics import f1_score
import functools
import numpy as np

EVAL_METRICS = ['mic_f1', 'mac_f1']
DEFAULT_CMP_BY = 'mic_f1'

@functools.total_ordering
class EvalResults(dict):
    def __init__(self, mic_f1, mac_f1, cmp_by=DEFAULT_CMP_BY):
        super().__init__({
            'mic_f1': mic_f1,
            'mac_f1': mac_f1
        })
        self.cmp_by = cmp_by

    def __lt__(self, other):
        if not isinstance(other, EvalResults):
            return NotImplemented
        fst = self.cmp_by
        snd = [m for m in EVAL_METRICS if m != self.cmp_by][0]
        if self[fst] < other[fst]:
            return True
        elif self[fst] > other[fst]:
            return False
        # self[fst] = other[fst]
        elif self[snd] < other[snd]:
            return True
        else:
            return False 

    def __eq__(self, other):
        if not isinstance(other, EvalResults):
            return NotImplemented
        # Equal if all metrics are the same
        return all(self[m] == other[m] for m in EVAL_METRICS)

    def __le__(self, other):
        return self < other or self == other

    def __repr__(self):
        return (f"mic_f1: {self['mic_f1']:.4f}, "
                f"mac_f1: {self['mac_f1']:.4f}")

    def to_dict(self):
        return { 'mic_f1': self['mic_f1'], 'mac_f1': self['mac_f1'] }

    def to_serialisable(self):
        return { k:str(v) for k,v in self.to_dict().items() }

    @staticmethod
    def average(results, stderr=True):
        mic_f1s = [res['mic_f1'] for res in results]
        mac_f1s = [res['mac_f1'] for res in results]

        avgs = EvalResults(np.mean(mic_f1s), np.mean(mac_f1s))
        stds = None if not stderr else EvalResults(stderr(mic_f1s), stderr(mac_f1s))
        
        return avgs, stds
    

def stderr(xs):
    return np.std(xs, ddof=1) / np.sqrt(len(xs))


def evaluate(model, data, mask, cmp_by=DEFAULT_CMP_BY):
    """
    Evaluates model and returns its validation accuracy, 
    micro-F1 and macro-F1 scores on given mask.
    """
    model.eval()
    with torch.no_grad():  # disable gradient computation during evaluation
        # forward pass
        out = model(data.x, data.edge_index)
        # predict the class with max score
        pred = out.argmax(dim=1)
        true_labels = data.y[mask]
        # calculate F1 scores (`f1_score` expects the inputs to be on the CPU)
        mic_f1 = f1_score(true_labels.cpu(), pred[mask].cpu(), average='micro') # equivalent to accuracy for this task
        mac_f1 = f1_score(true_labels.cpu(), pred[mask].cpu(), average='macro')

    return EvalResults(mic_f1, mac_f1, cmp_by=cmp_by)