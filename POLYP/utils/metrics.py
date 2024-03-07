#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from medpy import metric
from scipy import ndimage


def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))


def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))


def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def recall(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    return sensitivity(test, reference, confusion_matrix, nan_for_nonexisting, **kwargs)


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))


def accuracy(test=None, reference=None, confusion_matrix=None, **kwargs):
    """(TP + TN) / (TP + FP + FN + TN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return float((tp + tn) / (tp + fp + tn + fn))


def fscore(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""

    precision_ = precision(test, reference, confusion_matrix, nan_for_nonexisting)
    recall_ = recall(test, reference, confusion_matrix, nan_for_nonexisting)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_)


def false_positive_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (FP + TN)"""

    return 1 - specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_omission_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TN + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(fn / (fn + tn))


def false_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FN / (TP + FN)"""

    return 1 - sensitivity(test, reference, confusion_matrix, nan_for_nonexisting)


def true_negative_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    return specificity(test, reference, confusion_matrix, nan_for_nonexisting)


def false_discovery_rate(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """FP / (TP + FP)"""

    return 1 - precision(test, reference, confusion_matrix, nan_for_nonexisting)


def negative_predictive_value(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FN)"""

    return 1 - false_omission_rate(test, reference, confusion_matrix, nan_for_nonexisting)


def total_positives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fp


def total_negatives_test(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fn


def total_positives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TP + FN"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tp + fn


def total_negatives_reference(test=None, reference=None, confusion_matrix=None, **kwargs):
    """TN + FP"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()

    return tn + fp


def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)


def hausdorff_distance_95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)


def avg_surface_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.asd(test, reference, voxel_spacing, connectivity)


def avg_surface_distance_symmetric(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.assd(test, reference, voxel_spacing, connectivity)


def data_process(pred, label, threshold=0.5):
    '''
    pred = np.array(pred*255).astype(np.uint8)
    label = np.array(label*255).astype(np.uint8)

    for N in range(pred.shape[0]):
        for C in range(pred.shape[1]):
            _, pred[N][C] = cv2.threshold(pred[N][C], 0, 255, cv2.THRESH_OTSU)
            _, label[N][C] = cv2.threshold(label[N][C], 0, 255, cv2.THRESH_OTSU)
    pred, label = pred // 255, label // 255
    '''
    pred = np.array(pred)
    label = np.array(label)

    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0

    return pred.astype(np.uint8), label.astype(np.uint8)


def dice_compute(test, reference):
    batch_size = reference.shape[0]
    dices = []

    for batch in range(batch_size):
        one_dice = dice(test=test[batch], reference=reference[batch])
        dices.append(one_dice)
    return dices


def asd_compute(test, reference):
    batch_size = reference.shape[0]
    asds = []

    for batch in range(batch_size):
        asd = avg_surface_distance(test=test[batch], reference=reference[batch])
        asds.append(asd)
    return asds


def dice_metric(pred, label):
    batch_size = pred.shape[0]
    dices = []
    smooth = 1e-6

    for batch in range(batch_size):
        intersection = (pred[batch] * label[batch]).sum()
        dice = (2 * intersection + smooth) / (pred[batch].sum() + label[batch].sum() + smooth)
        dices.append(dice*100.)
    return dices


def ea_metric(pred, label):
    batch_size = pred.shape[0]
    eas = []

    for batch in range(batch_size):
        ea = enhanced_align(pred[batch][0], label[batch][0])
        eas.append(ea*100.)
    return eas


def sm_metric(pred, label):
    batch_size = pred.shape[0]
    sms = []
    SM_Metric = Structure_measure()

    for batch in range(batch_size):
        sm = SM_Metric.get_score(pred[batch][0], label[batch][0])
        sms.append(sm*100.)
    return sms


# Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
def enhanced_align(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)

    th = 2 * pred.mean()
    if th > 1:
        th = 1
    FM = np.zeros(gt.shape)
    FM[pred >= th] = 1
    FM = np.array(FM, dtype=bool)
    GT = np.array(gt, dtype=bool)
    dFM = np.double(FM)
    if (sum(sum(np.double(GT))) == 0).all():
        enhanced_matrix = 1.0-dFM
    elif (sum(sum(np.double(~GT))) == 0).all():
        enhanced_matrix = dFM
    else:
        dGT = np.double(GT)
        align_matrix = AlignmentTerm(dFM, dGT)
        enhanced_matrix = EnhancedAlignmentTerm(align_matrix)
    [w, h] = np.shape(GT)
    score = sum(sum(enhanced_matrix)) / (w * h - 1 + 1e-8)
    return score

def AlignmentTerm(dFM, dGT):
    mu_FM = np.mean(dFM)
    mu_GT = np.mean(dGT)
    align_FM = dFM - mu_FM
    align_GT = dGT - mu_GT
    align_Matrix = 2. * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + 1e-8)
    return align_Matrix

def EnhancedAlignmentTerm(align_Matrix):
    enhanced = np.power(align_Matrix + 1, 2) / 4
    return enhanced


class Structure_measure():
    # Structure-measure: A new way to evaluate foreground maps (ICCV 2017)
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def get_score(self, pred, gt):
        pred = np.array(pred)
        gt = np.array(gt)

        gt = gt > 0.5
        score = self.cal(pred, gt)
        return score

    def cal(self, pred, gt):
        y = np.mean(gt)
        if y == 0:
            score = 1 - np.mean(pred)
        elif y == 1:
            score = np.mean(pred)
        else:
            score = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
        return score

    def object(self, pred, gt):
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)

        u = np.mean(gt)
        return u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, np.logical_not(gt))

    def s_object(self, in1, in2):
        x = np.mean(in1[in2])
        sigma_x = np.std(in1[in2])
        return 2 * x / (pow(x, 2) + 1 + sigma_x + 1e-8)

    def region(self, pred, gt):
        [y, x] = ndimage.center_of_mass(gt)
        y = int(round(y)) + 1
        x = int(round(x)) + 1
        [gt1, gt2, gt3, gt4, w1, w2, w3, w4] = self.divideGT(gt, x, y)
        pred1, pred2, pred3, pred4 = self.dividePred(pred, x, y)

        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def divideGT(self, gt, x, y):
        h, w = gt.shape
        area = h * w
        LT = gt[0:y, 0:x]
        RT = gt[0:y, x:w]
        LB = gt[y:h, 0:x]
        RB = gt[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = (h - y) * (w - x) / area

        return LT, RT, LB, RB, w1, w2, w3, w4

    def dividePred(self, pred, x, y):
        h, w = pred.shape
        LT = pred[0:y, 0:x]
        RT = pred[0:y, x:w]
        LB = pred[y:h, 0:x]
        RB = pred[y:h, x:w]

        return LT, RT, LB, RB

    def ssim(self, in1, in2):
        in2 = np.float32(in2)
        h, w = in1.shape
        N = h * w

        x = np.mean(in1)
        y = np.mean(in2)
        sigma_x = np.var(in1)
        sigma_y = np.var(in2)
        sigma_xy = np.sum((in1 - x) * (in2 - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + 1e-8)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score


def calculate_metrics(test, reference):
    # enhanced_metric = ea_metric(test, reference)
    # structure_metric = sm_metric(test, reference)

    test, reference = data_process(pred=test, label=reference, threshold=0.5)
    enhanced_metric = ea_metric(test, reference)
    structure_metric = sm_metric(test, reference)
    dice = dice_metric(test, reference)
    return [dice, enhanced_metric, structure_metric]


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": jaccard,
    "Hausdorff Distance": hausdorff_distance,
    "Hausdorff Distance 95": hausdorff_distance_95,
    "Precision": precision,
    "Recall": recall,
    "Avg. Symmetric Surface Distance": avg_surface_distance_symmetric,
    "Avg. Surface Distance": avg_surface_distance,
    "Accuracy": accuracy,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}
