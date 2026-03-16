import numpy as np
import os


import torch
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class MultilabelPostprocessing:
    def __init__(self, threshold=0.6):
        self.threshold = threshold 

    def __call__(self, x: torch.Tensor, *args, **kwds):
        return (x > self.threshold).int()

class MultilabelMetric(object):
    def __init__(self, main_indicator="acc", n_classes=4):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.n_classes = n_classes
        self.reset()

    def __call__(self, preds, labels, *args, **kwargs):
        pass 

        # print(preds) 
        # print(labels)
        # print(kkk)

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

       

        self.labels += labels.tolist()
        self.preds += preds.tolist()

        return_dict = self.get_metric()
        return_dict = self.reduce(return_dict)

        return return_dict
    
    def get_metric(self, reduction=True):
        pass 

        preds = np.array(self.preds)
        labels = np.array(self.labels) 

        f1 = f1_score(labels, preds, average=None, zero_division=0)
        precision = precision_score(labels, preds, average=None, zero_division=0)
        recall = recall_score(labels, preds, average=None, zero_division=0)
        accuracy = f1_score(labels, preds, average=None, zero_division=0)

        
        return_dict = {
            "acc": accuracy,
            "recall": recall, 
            "precision": precision, 
            "f1": f1
            }
        
        if reduction:
            return self.reduce(return_dict)


        return return_dict


    def reset(self):
        
        # self.metrics = {cls: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cls in range(self.n_classes)} 
        self.labels = []
        self.preds = []

    @staticmethod
    def reduce(metrics):
        new_metrics = {}
        for k, v in metrics.items():
            new_metrics[k] = np.mean(v)

        return new_metrics


class ClsMetric(object):
    def __init__(self, main_indicator="acc", n_classes=4, **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.n_classes = n_classes
        self.reset()
    

    def __call__(self, preds, labels, *args, **kwargs):
        
        
        assert len(preds.shape) == 1 or len(preds.shape) == 2
        assert len(labels.shape) == 1 or len(labels.shape) == 2
        
        if preds.ndim == 2:
            preds = torch.argmax(preds, 1)
            
        if labels.ndim == 2:
            labels = torch.argmax(labels, 1)

        
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # print(preds, labels)

        # Create a dictionary to hold TP, TN, FP, FN for each class

        # Calculate TP, TN, FP, FN for each class
        for cls in range(self.n_classes):
            for true_label, pred_label in zip(labels, preds):
                if true_label == cls and pred_label == cls:
                    self.metrics[cls]["TP"] += 1
                elif true_label == cls and pred_label != cls:
                    self.metrics[cls]["FN"] += 1
                elif true_label != cls and pred_label == cls:
                    self.metrics[cls]["FP"] += 1
                elif true_label != cls and pred_label != cls:
                    self.metrics[cls]["TN"] += 1
                    
        return self.get_metric()
       
        
        

    def get_metric(self, reduction=True, round=3):
        """
        return metrics {
                 'acc': 0
            }
        """
        acc = []
        precision = []
        recall = []
        f1 = []
        
        total_tp = total_tn = total_fp = total_fn = 0
        macro_precision = macro_recall = macro_f1 = 0

        for cls in range(self.n_classes):
            TP = self.metrics[cls]["TP"]
            TN = self.metrics[cls]["TN"]
            FP = self.metrics[cls]["FP"]
            FN = self.metrics[cls]["FN"]

            total_tp += TP
            total_tn += TN
            total_fp += FP
            total_fn += FN
            
            acc_ = (TP + TN) / (TP + FP + TN + FN + self.eps) 
            pre_ = TP / (TP + FP + self.eps)
            rec_ = TP / (TP + FN + self.eps)  
            f1_ = 2 * (pre_ * rec_) / (pre_ + rec_ + self.eps) 
            
            acc.append(acc_)
            precision.append(pre_)
            recall.append(rec_)
            f1.append(f1_)
            
        
        return_dict = {
            "acc": acc,
            "recall": recall, 
            "precision": precision, 
            "f1": f1
            }
        if reduction:
            return_dict = self.reduce(return_dict)

        if round == -1:
            return return_dict
            
        for k, v in return_dict.items():
            return_dict[k] = np.round_(v, decimals = 3) 
        
        return return_dict

    @staticmethod
    def reduce(metrics):
        new_metrics = {}
        for k, v in metrics.items():
            new_metrics[k] = np.mean(v)

        return new_metrics

    def test(self):
        pred = np.array([[10,5,5],[5,5,100],[5,5,19],[1,5,2]])
        target = np.array([[0,1,0],[0,0,1],[1,0,0],[0,1,0]])
        
        dict = self.__call__(pred, target)
        print(dict)
        # print(self.metrics)
        
        # print(self.get_metric())


    def reset(self):
        
        self.metrics = {cls: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cls in range(self.n_classes)}

def convert_cuda(batches):
    new_batches = [x.cuda() if isinstance(x, torch.Tensor) else x for x in batches ]
    if isinstance(batches, tuple):
        return tuple(new_batches)
    if isinstance(batches, list):
        return new_batches 

def write_log(logs_path, log_str, prefix="valid", should_print=True, mode="a"):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path, exist_ok=True)
    """ prefix: valid, train, test """
    if prefix in ["valid", "test"]:
        with open(os.path.join(logs_path, "valid_console.txt"), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if prefix in ["valid", "test", "train"]:
        with open(os.path.join(logs_path, "train_console.txt"), mode) as fout:
            if prefix in ["valid", "test"]:
                fout.write("=" * 10)
            fout.write(log_str + "\n")
            fout.flush()
    else:
        with open(os.path.join(logs_path, "%s.txt" % prefix), mode) as fout:
            fout.write(log_str + "\n")
            fout.flush()
    if should_print:
        print(log_str)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_model(checkpoint=None, save_path="./", is_best=False, model_name="last"):

    os.makedirs(save_path, exist_ok=True)

    latest_fname = os.path.join(save_path, "latest.txt")
    model_path = os.path.join(save_path, f'{model_name}.pth.tar')
    
    with open(latest_fname, "w") as _fout:
        _fout.write(model_path + "\n")
    torch.save(checkpoint, model_path)

    if is_best:
        best_path = os.path.join(save_path, "model_best.pth.tar")
        torch.save(checkpoint, best_path)
            
        return best_path

    return model_path

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name=None):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): 

        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class MultiClassAverageMeter:

    """Multi Binary Classification Tasks"""

    def __init__(self, num_classes, balanced=False, **kwargs):

        super(MultiClassAverageMeter, self).__init__()
        self.num_classes = num_classes
        self.balanced = balanced

        self.counts = []
        for k in range(self.num_classes):
            self.counts.append(np.ndarray((2, 2), dtype=np.float32))

        self.reset()

    def reset(self):
        for k in range(self.num_classes):
            self.counts[k].fill(0)

    def add(self, outputs, targets):
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        for k in range(self.num_classes):
            output = np.argmax(outputs[:, k, :], axis=1)
            target = targets[:, k]

            x = output + 2 * target
            bincount = np.bincount(x.astype(np.int32), minlength=2 ** 2)

            self.counts[k] += bincount.reshape((2, 2))

    def value(self):
        mean = 0
        for k in range(self.num_classes):
            if self.balanced:
                value = np.mean(
                    (
                        self.counts[k]
                        / np.maximum(np.sum(self.counts[k], axis=1), 1)[:, None]
                    ).diagonal()
                )
            else:
                value = np.sum(self.counts[k].diagonal()) / np.maximum(
                    np.sum(self.counts[k]), 1
                )

            mean += value / self.num_classes * 100.0
        return mean

