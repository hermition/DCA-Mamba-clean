import numpy as np
import torch
from medpy.metric.binary import hd95

def acc_score(output, target, if_train=False, thod=0.5):

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > thod
    target_ = target > thod
    acc = (output_ == target_).sum() / (output_.shape[0]*output_.shape[1]*output_.shape[2]*output_.shape[3])
    # f1
    tp = (output_ * target_).sum().astype(float)
    fp = ((1 - target_) * output_).sum().astype(float)
    tn = ((1 - target_) * (1 - output_)).sum().astype(float)
    fn = (target_ * (1 - output_)).sum().astype(float)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    BS = np.mean((output - target) ** 2)

    # # hausdorff_distance95 = compute_hd95(output_, target_)
    # hd95_metric = HausdorffDistanceMetric(percentile=95)
    # # 输入需为张量（BCHW 格式）
    # pred_tensor = torch.tensor(output_)  # 增加 batch 和 channel 维度
    # gt_tensor = torch.tensor(target_)
    # # 计算并返回 HD95
    # hd95_metric(y_pred=pred_tensor, y=gt_tensor)
    # hausdorff_distance95 = hd95_metric.aggregate().item()
    if not if_train:
        try:
            hausdorff_distance95 = hd95(output_, target_)
        except Exception as e:
            # hausdorff_distance95 = hd95(output_, target_)
            hausdorff_distance95 = 0
            print(f"Error computing Hausdorff distance: {e}")

        # _output = output.flatten()
        # _output_ = output_.flatten()
        # _target = target.flatten()

        # f_mask = _output_>0.1
        # _output = _output[f_mask]
        # _output_ = _output_[f_mask]
        # _target = _target[f_mask]

        # n_index = _output_==0
        # _output[n_index] = 1 - _output[n_index]

        # try:
        #     ece = expected_calibration_error(_output, _output_, _target, num_bins=15)
        # except Exception as e:
        #     ece = 0
        #     print(f"Error computing ECE: {e}")
        ece = 0
    else:
        ece = 0
        hausdorff_distance95 = 0

    # hausdorff_distance95 = 0
    # ece = compute_ece(output, target)



    return acc, f1, precision, recall, spe, hausdorff_distance95, ece, BS

def iou_score(output, target, thod=0.5):
    smooth = 1e-8

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > thod
    target_ = target > thod
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    return iou, dice

def compute_calibration(true_labels, pred_labels, confidences, num_bins=15):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert (len(confidences) == len(pred_labels))
    assert (len(confidences) == len(true_labels))
    assert (num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)
    bin_accuracies = np.zeros(num_bins, dtype=np.float32)
    bin_confidences = np.zeros(num_bins, dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return {"accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "counts": bin_counts,
            "bins": bins,
            "avg_accuracy": avg_acc,
            "avg_confidence": avg_conf,
            "expected_calibration_error": ece,
            "max_calibration_error": mce}

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
