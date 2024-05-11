import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import dill
from argparse import ArgumentParser
import argparse
import os
from sklearn import metrics
import sklearn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits
from methods import clip
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from PIL import Image
import torchvision
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

import pandas as pd
from sklearn.decomposition import PCA
from torch.nn import functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path
import classifier
import classifiersubpace
# TODO: Debug
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from scipy.optimize import linear_sum_assignment as linear_assignment



class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss







class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, clip_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
        self.clip_transform = clip_transform

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)] + [self.clip_transform(x)]








def test_kmeans(model, test_loader,
                epoch, save_name,
                args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().detach().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=None)
    print('epoch  Accuracies: epoch {:.4f} | All {:.4f} | Old {:.4f} | New {:.4f}'.format(epoch, all_acc, old_acc,
                                                                                          new_acc))

    return all_acc, old_acc, new_acc



def featureextract(model, test_loader, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().detach().numpy())
        targets = np.append(targets, label.cpu().numpy())

    all_feats = np.concatenate(all_feats)

    targets = targets.astype(int)

    all_feats = torch.from_numpy(all_feats).float()

    targets = torch.from_numpy(targets).long()

    return all_feats, targets


def featureextractunlabeled(model, test_loader, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().detach().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    mask = mask.astype(bool)
    targets = targets.astype(int)

    old_classes = set(targets[mask])
    new_classes = set(targets[~mask])
    Xseen = all_feats[mask, :]
    Yseen = targets[mask]
    Xnew = all_feats[~mask, :]
    Ynew = targets[~mask]
    seennum = Yseen.shape[0]
    att = Ynew.shape[0]
    old_classes = np.array(list(old_classes))
    new_classes = np.array(list(new_classes))
    oldnumber = old_classes.shape[0]
    Ynewopen = oldnumber * np.ones(shape=(att, 1), dtype=int)
    Ynewopen = np.squeeze(Ynewopen, axis=1)
    all_feats = np.concatenate((Xseen, Xnew), axis=0)
    targets = np.concatenate((Yseen, Ynew), axis=0)

    targetsopen = np.concatenate((Yseen, Ynewopen), axis=0)

    all_true = np.full((seennum), True, dtype=bool)
    all_false = np.full((att), False, dtype=bool)
    mask1 = np.concatenate((all_true, all_false), axis=0)

    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    preds = preds.astype(int)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acctrain, old_acctrain, new_acctrain = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask1,
                                                                   T=1, eval_funcs=args.eval_funcs, save_name=None,
                                                                   writer=None)
    print('  Accuracies: epoch {:.4f} | All {:.4f} | Old {:.4f} | New {:.4f}'.format(1, all_acctrain, old_acctrain,
                                                                                     new_acctrain))

    all_feats = torch.from_numpy(all_feats).float()
    preds = torch.from_numpy(preds).long()
    targets = torch.from_numpy(targets).long()
    targetsopen = torch.from_numpy(targetsopen).long()
    old_classes = torch.from_numpy(old_classes).int()
    new_classes = torch.from_numpy(new_classes).int()

    return all_feats, targets, targetsopen, preds, old_classes, new_classes


clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class opimage(object):
    def __call__(self, imgs):
        # 图片归一化
        imgs = imgs / 255.0
        return imgs


class cpu2gpu(object):
    def __int__(self):
        pass

    def __call__(self, img):
        return img.cuda()


class gpu2cpu(object):
    def __int__(self):
        pass

    def __call__(self, img):
        return img.cpu()


class clipencode(object):
    def __int__(self):
        pass

    def __call__(self, img):
        return clip_model.encode_image(img)


def clip_trans():
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
        opimage(),
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        cpu2gpu(),
        clipencode(),
        gpu2cpu(),
    ])



def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label



def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size




def clus_acc(t_codes, t_gt, mapping, gt_freq, sub_score):
    # Print the status of matching
    print(mapping, len(sub_score))
    for i, (src, tgt) in enumerate(mapping):
        mask = t_codes == tgt
        i_gt = torch.masked_select(t_gt, mask)
        i_acc = ((i_gt == src).sum().float()) / len(i_gt)
        if src in gt_freq:
            gt_cnt = gt_freq[src]
        else:
            gt_cnt = 1.0
        recall = i_acc * len(i_gt) / gt_cnt
        print(
            '{:0>2d}th Cluster ACC:{:.2f} Correct/Total/GT {:0>2d}/{:0>2d}/{:0>2d} Precision:{:.3f} Recall:{:.3f} Score:{:.2f}'.format(
                src, i_acc.item(), (i_gt == src).sum().item(), len(i_gt), int(gt_cnt), i_acc, recall, sub_score[i]))



def sklearn_kmeans(feat, num_centers, init=None):
    if init is not None:
        kmeans = KMeans(n_clusters=num_centers, init=init, random_state=0).fit(feat.cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat.cpu().numpy())
    center, t_codes = kmeans.cluster_centers_, kmeans.labels_
    score = sklearn.metrics.silhouette_score(feat.cpu().numpy(), t_codes)
    return torch.from_numpy(center), torch.from_numpy(t_codes), score





def cal_pairwise_dist(x):
    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    # 返回任意两个点之间距离的平方
    return dist


def rbf(dist, t=1.0):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist / t))


def cal_rbf_dist(data, n_neighbors=10, t=1):
    dist = cal_pairwise_dist(data)
    # dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)

    W = np.zeros((n, n))
    D = np.zeros_like(W)
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    for i in range(n):
        D[i, i] = np.sum(W[i])

    L = D - W

    return W, L




def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)




def ensembleYseen(Y1, Y2):
    row = Y1.size(0)

    result = torch.zeros(row, 1)
    indx = torch.zeros(row)
    for num in range(0, row):
        temp1 = Y1[num]
        temp2 = Y2[num]
        if temp1 == temp2:
            result[num, 0] = temp1
        else:
            indx[num] = num
    inddiffer = np.nonzero(indx).long()
    inddiffer = inddiffer.squeeze()
    tem1 = torch.linspace(0, row - 1, steps=row)
    tem1 = tem1.long()
    indsame = np.setdiff1d(tem1, inddiffer)
    indsame = torch.from_numpy(indsame)
    indsame = indsame.long()
    return result, indsame, inddiffer


def compute_lrW(X, Y, beta):
    Xtt = X.transpose(0, 1)
    temp1 = torch.mm(Xtt, X)
    d = X.size(1)
    temp3 = temp1 + beta * torch.eye(d)
    temp2 = torch.mm(Xtt, Y)
    W1 = torch.inverse(temp3)
    W = torch.mm(W1, temp2)
    return W


def zuizhongsvm(Rall, train_X, train_Y,  old_classes, Xtest, Xunlabeled, alpha,alpha1,ct ):
    batchsize = 100
    alr = 0.001
    abeta1 = 0.5
    anepoch = 20
    pretrain_clst = classifier.CLASSIFIER(Rall.cuda(), alpha,alpha1, train_X.cuda(), train_Y.cuda(), old_classes.size(0) + ct,
                                          train_X.size(1),
                                          True,
                                          alr, abeta1,anepoch, batchsize, '')

    predictst = pretrain_clst.model(Xtest.cuda())
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    predicted_labels = predicted_labels.cpu().numpy()
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Ytest.numpy()]))
    all_acc1, old_acc1, new_acc1 = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels, mask=mask,
                                                       eval_funcs=args.eval_funcs, save_name=None,
                                                       writer=None)
    print('svm test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc1, old_acc1, new_acc1))

    predictst = pretrain_clst.model(Xunlabeled.cuda())
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    predicted_labels = predicted_labels.cpu()
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Yunlabeled.numpy()]))
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels.numpy(),
                                                    mask=mask,
                                                    eval_funcs=args.eval_funcs, save_name=None,
                                                    writer=None)
    print(' svm trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    return all_acc1, old_acc1, new_acc1, all_acc, old_acc, new_acc

def kl_divergence(P, Q):
    # Make sure that P and Q are normalized probability distributions along the last dimension.
    P = P / P.sum(dim=-1, keepdim=True)
    Q = Q / Q.sum(dim=-1, keepdim=True)

    # Calculate the KL divergence element-wise for each pair of distributions
    kl_div = (P * torch.log(P / Q)).sum(dim=-1)

    return kl_div

def constructW(Label):
    nLabel = Label.size(0)
    G = torch.zeros(nLabel, nLabel)
    D = torch.zeros(nLabel, nLabel)
    for idx in range(nLabel):
        for idx2 in range(nLabel):
            if Label[idx] == Label[idx2]:
                G[idx][idx2] = 1
        G[idx][idx] = 0
        tem = G[idx, :]
        tem = torch.sum(tem)
        if tem != 0:
            G[idx, :] = 1 / tem * G[idx, :]
        D[idx, idx] = torch.sum(G[idx, :])

    L = D - G
    return G, L


def compute_lrWgraph(X, train_Ya, beta, gama):
    seenlabelone_hot = torch.zeros(train_Ya.size(0), ct + cs)
    seenlabelone_hot = seenlabelone_hot.scatter_(1, train_Ya.unsqueeze(1), 1)
    Xtt = X.transpose(0, 1)
    temp1 = torch.mm(Xtt, X)
    d = X.size(1)
    temp3 = temp1 + beta * torch.eye(d)
    G, L = constructW(train_Ya)
    temp1w = torch.mm(Xtt, L)
    tempXLX = torch.mm(temp1w, X)
    temp2 = torch.mm(Xtt, seenlabelone_hot)
    temp31 = temp3 + gama * tempXLX
    W1 = torch.inverse(temp31)
    W = torch.mm(W1, temp2)
    return W


def getMaxsecond(matrix):  # 返回每行最大元素和最小元素的索引矩阵 (x,result[x,0])是第x行最大元素的坐标，(x,result[x,1])是次大的坐标
    row = matrix.size(0)
    col = matrix.size(1)
    result = torch.zeros(row, 2)  # 每行0号位置为最大值索引 1号位置为次大值索引
    resultmix = torch.zeros(row, 1)
    for num in range(0, row):  # 取每一行的元素的最大和次大坐标放入 新的矩阵
        temp = matrix[num]  # 取第num行元素
        pos, fzhi = torch.sort(temp, descending=True)  # 将temp中元素进行排序 并记录索引
        result[num, 0] = pos[0]
        result[num, 1] = pos[1]
        resultmix[num, 0] = pos[0] - pos[1]
    return result, resultmix


def guiyihua(expMatrix):
    a = torch.sum(expMatrix, 1).unsqueeze(dim=1)
    be = expMatrix.size(1)
    fewf = a.repeat(1, be)
    fewf = fewf + 0.0001
    probMatrix = expMatrix / fewf
    probMatrix1 = L2Norm(probMatrix)
    return probMatrix1


def accpred(predicted_labelsdis, input_unseen_labelreal):
    qfda = torch.sum(input_unseen_labelreal == predicted_labelsdis)
    saccdis = qfda / input_unseen_labelreal.size(0)
    print(' 正确率',
          saccdis * 100)
    return saccdis

def disselectzhongxinphi(input_unseen_feature,  input_unseen_labelmap, input_unseen_labelreal,bili,pred,predicted_labelprost):
    # input_unseen_labelreal = input_unseen_labelreal
    # input_unseen_labelmap = input_unseen_labelmap
    # dfewe = np.unique(input_unseen_labelmap)
    # dfewe = torch.from_numpy(dfewe).long()
    # seenRslev = torch.FloatTensor([])
    # for i in range(dfewe.size(0)):
    #     ij = dfewe[i]
    #     temp2 = torch.nonzero(input_unseen_labelmap == ij)
    #     indx = temp2.squeeze().long()
    #     Rssel = input_unseen_feature[indx, :]
    #
    #     if Rssel.size(0) == input_unseen_feature.size(1):
    #         Rssel1 = Rssel.unsqueeze(1)
    #     else:
    #         Rssel1 = torch.mean(Rssel, 0)
    #         Rssel1 = Rssel1.unsqueeze(1)
    #
    #     Rssel1 = Rssel1.transpose(0, 1)
    #     seenRslev = torch.cat((seenRslev, Rssel1), 0)
    #
    # loss111 = pairwise_distances(input_unseen_feature, seenRslev)
    # predprod = chuanboyici(loss111)
    # predicted_labelprostdist, predicted_labelsdis = torch.max(predprod, 1)
    #
    # new_acct = cluster_acc(input_unseen_labelreal.numpy(), predicted_labelsdis.numpy())
    # print('distance clusteracc正确率',
    #       new_acct * 100)
    # qfda = torch.sum(input_unseen_labelreal == predicted_labelsdis)
    # saccdis = qfda / input_unseen_labelreal.size(0)
    # print('distance 正确率',
    #       saccdis * 100)
    indx = input_unseen_labelmap == pred
    cc = torch.nonzero(indx).squeeze()
    cc = cc.long()
    input_unseen_featureconfidence = torch.index_select(input_unseen_feature, 0, cc)
    input_unseen_labelconfidence = torch.index_select(input_unseen_labelmap, 0, cc)
    input_unseen_labelconfidencereal = torch.index_select(input_unseen_labelreal, 0, cc)
    new_acct = cluster_acc(input_unseen_labelconfidencereal.numpy(),  input_unseen_labelconfidence.numpy())
    print('distance clusteracc选择后的正确率',
          new_acct * 100)
    qfda = torch.sum(input_unseen_labelconfidencereal ==  input_unseen_labelconfidence)
    saccdis = qfda / input_unseen_labelconfidencereal.size(0)
    print('distance 选择后的正确率',
          saccdis * 100)
    predicted_labelprostaa = torch.index_select(predicted_labelprost, 0, cc)
    # mu0 = torch.mean(predicted_labelprostaa)
    # mumax, predicted_labelsdisfef = torch.max(predicted_labelprostaa, 0)
    # mumin, predicted_labelsdfisff = torch.min(predicted_labelprostaa, 0)
    # meiduan = mumax - mumin
    # meiduan = 0.1 * meiduan
    # mujiadeta = mumin + meiduan * bili
    # indx = torch.ge(predicted_labelprostaa, mujiadeta)
    # ccaa = torch.nonzero(indx).squeeze()
    sorted, indices = torch.sort(predicted_labelprostaa, descending=True)
    nb = 0.1 * bili  * predicted_labelprostaa.size(0)
    nb = int(nb)
    ccaa = indices[0:nb]




    input_unseen_featureconfidenceaa = torch.index_select(input_unseen_featureconfidence, 0, ccaa)
    input_unseen_labelconfidenceaa = torch.index_select(input_unseen_labelconfidence, 0, ccaa)
    input_unseen_labelconfidencerealaa = torch.index_select(input_unseen_labelconfidencereal, 0, ccaa)
    new_acct = cluster_acc(input_unseen_labelconfidencerealaa.numpy(), input_unseen_labelconfidenceaa.numpy())
    print('confidence distance clusteracc选择后的正确率',
          new_acct * 100)
    qfda = torch.sum(input_unseen_labelconfidencerealaa == input_unseen_labelconfidenceaa)
    saccdis = qfda / input_unseen_labelconfidencerealaa.size(0)
    print('confidence distance 选择后的正确率',
          saccdis * 100)



    return  input_unseen_featureconfidenceaa,  input_unseen_labelconfidenceaa, input_unseen_labelconfidencerealaa,cc, ccaa

def disselectzhongxinphifarawayRs(input_unseen_feature,  input_unseen_labelmap, input_unseen_labelreal,bili,pred,predicted_labelprost, seenRs,ratio):
    predprod = pairwise_distances(input_unseen_feature, seenRs)
    predicted_labelprost1, predicted_labels1 = torch.max(predprod, 1)
    ewfe = torch.max(predicted_labelprost1)
    fewe = torch.min(predicted_labelprost1)
    fewfew1 = 0.1 * (ewfe - fewe)
    mu0 = fewe + ratio * fewfew1
    print(mu0)
    ind = torch.ge(predicted_labelprost1, mu0)
    inefewd = np.nonzero(ind).long()
    inefewd = inefewd.cpu()
    cc = inefewd.squeeze()



    input_unseen_featureconfidence = torch.index_select(input_unseen_feature, 0, cc)
    input_unseen_labelconfidence = torch.index_select(input_unseen_labelmap, 0, cc)
    input_unseen_labelconfidencereal = torch.index_select(input_unseen_labelreal, 0, cc)
    new_acct = cluster_acc(input_unseen_labelconfidencereal.numpy(),  input_unseen_labelconfidence.numpy())
    print('distance clusteracc选择后的正确率',
          new_acct * 100)
    return input_unseen_featureconfidence, input_unseen_labelconfidence, input_unseen_labelconfidencereal, cc

def disselectzhongxin(input_unseen_feature,  input_unseen_labelmap, input_unseen_labelreal,cs,bili):
    input_unseen_labelreal = input_unseen_labelreal - cs
    input_unseen_labelmap = input_unseen_labelmap - cs
    dfewe = np.unique(input_unseen_labelmap)
    dfewe = torch.from_numpy(dfewe).long()
    seenRslev = torch.FloatTensor([])
    for i in range(dfewe.size(0)):
        ij = dfewe[i]
        temp2 = torch.nonzero(input_unseen_labelmap == ij)
        indx = temp2.squeeze().long()
        Rssel = input_unseen_feature[indx, :]
        Rssel1 = torch.mean(Rssel, 0)
        Rssel1 = Rssel1.unsqueeze(1)
        Rssel1 = Rssel1.transpose(0, 1)
        seenRslev = torch.cat((seenRslev, Rssel1), 0)

    loss111 = pairwise_distances(input_unseen_feature, seenRslev)
    predprod = chuanboyici(loss111)
    predicted_labelprostdist, predicted_labelsdis = torch.max(predprod, 1)
    new_acct = cluster_acc(input_unseen_labelreal.numpy(), predicted_labelsdis.numpy())
    print('distance clusteracc正确率',
          new_acct * 100)
    qfda = torch.sum(input_unseen_labelreal == predicted_labelsdis)

    saccdis = qfda / input_unseen_labelreal.size(0)
    print('distance 正确率',
          saccdis * 100)
    mu0 = torch.mean(predicted_labelprostdist)
    mujiadeta = mu0
    mumax, predicted_labelsdisfef = torch.max(predicted_labelprostdist, 0)
    mumin, predicted_labelsdfisff = torch.min(predicted_labelprostdist, 0)
    meiduan = mumax - mumin
    meiduan = 0.1 * meiduan
    mujiadeta = mumin + meiduan * bili

    indx = torch.ge(predicted_labelprostdist, mujiadeta)
    cc  = torch.nonzero(indx).squeeze()
    cc = cc.long()

    input_unseen_featureconfidence = torch.index_select(input_unseen_feature, 0, cc)
    input_unseen_labelconfidence = torch.index_select(predicted_labelsdis, 0, cc)
    input_unseen_labelconfidencereal = torch.index_select(input_unseen_labelreal, 0, cc)
    new_acct = cluster_acc(input_unseen_labelconfidencereal.numpy(),  input_unseen_labelconfidence.numpy())
    print('distance clusteracc选择后的正确率',
          new_acct * 100)
    qfda = torch.sum(input_unseen_labelconfidencereal ==  input_unseen_labelconfidence)
    saccdis = qfda / input_unseen_labelconfidencereal.size(0)
    print('distance 选择后的正确率',
          saccdis * 100)
    input_unseen_labelconfidence = input_unseen_labelconfidence + cs
    input_unseen_labelconfidencereal = input_unseen_labelconfidencereal + cs
    return  input_unseen_featureconfidence,  input_unseen_labelconfidence, input_unseen_labelconfidencereal

def maxsecondselectseen(input_unseen_feature, input_unseen_labelmap, input_unseen_labelreal, betaf, nsreal):
    seenlabelone_hot = torch.zeros(input_unseen_labelmap.size(0), ct + old_classes.size(0))
    seenlabelone_hot = seenlabelone_hot.scatter_(1, input_unseen_labelmap.unsqueeze(1), 1)
    Wlr = compute_lrW(input_unseen_feature, seenlabelone_hot, 0.1)
    predictst = torch.mm(input_unseen_feature, Wlr)
    predictst = guiyihua(predictst)
    maxsecon, maxmixsecon = getMaxsecond(predictst)
    maxmixsecon = maxmixsecon.squeeze()
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    mu0 = torch.mean(predicted_labelprost)
    mu1 = torch.mean(maxmixsecon)
    nall = input_unseen_feature.size(0)
    lamdset1 = torch.Tensor([2])
    fwew1 = int(betaf * nall)
    inefewdn = torch.Tensor([0.5])
    while inefewdn.size(0) < fwew1:
        lamdset1 = lamdset1 * 0.5
        mujiadeta = lamdset1 * mu0

        ind = torch.ge(predicted_labelprost, mujiadeta)
        inefewd = np.nonzero(ind).long()
        inefewd = inefewd.cpu()
        inefewd = inefewd.squeeze()
        nb = lamdset1 * mu1

        ind = torch.ge(maxmixsecon, nb)
        inefewdmsm = np.nonzero(ind).long()
        inefewdmsm = inefewdmsm.squeeze()
        inefewdn = set(inefewd.numpy()).intersection(set(inefewdmsm.numpy()))
        inefewdn = list(inefewdn)
        inefewdn = np.array(inefewdn)
        inefewdn = torch.from_numpy(inefewdn)
        inefewdn = inefewdn.long()
        fwaefasew = inefewdn.size(0)
    inputseend1 = input_unseen_feature[inefewdn, :]
    input_unseen_labelmapsee1 = input_unseen_labelmap[inefewdn]
    predicted_labelsseen = predicted_labels[inefewdn]
    input_unseen_labelrealselect = input_unseen_labelreal[inefewdn]
    qfda = torch.sum(input_unseen_labelrealselect == input_unseen_labelmapsee1)
    sacc1 = qfda / input_unseen_labelrealselect.size(0)
    print('mu0  {:.2f}, mu1  {:.2f}, xuanze yangben shuliang{:.2f}，高概率选择的样本正确率 {:.2f}', mujiadeta, nb, inefewdn.size(0),
          sacc1 * 100)
    # inefewdex  为unseen
    tem1 = torch.linspace(0, input_unseen_feature.size(0) - 1, steps=input_unseen_feature.size(0))
    tem1 = tem1.long()
    inefewdex = np.setdiff1d(tem1, inefewdn)
    inefewdex = torch.from_numpy(inefewdex)
    inefewdex = inefewdex.long()

    inputunseen1 = input_unseen_feature[inefewdex, :]
    input_unseen_labelmapunsee1 = input_unseen_labelmap[inefewdex]
    input_unseen_labelmapreal1 = input_unseen_labelreal[inefewdex]

    loss111 = pairwise_distances(input_unseen_feature, Rall)
    predprod = chuanboyici(loss111)
    predicted_labelprostdist, predicted_labelsdis = torch.max(predprod, 1)
    qfda = torch.sum(input_unseen_labelreal == predicted_labelsdis)
    saccdis = qfda / input_unseen_labelreal.size(0)
    print('distance 正确率',
          saccdis * 100)
    qfda = torch.sum(input_unseen_labelreal == predicted_labels)
    saccdis = qfda / input_unseen_labelreal.size(0)
    print('lr 正确率',
          saccdis * 100)

    indx = predicted_labelsdis == predicted_labels
    ccaa = torch.nonzero(indx).squeeze()
    fwew1 = int(0.0001 * nall)
    inefewdn = torch.Tensor([0.5])
    while inefewdn.size(0) < fwew1:
        lamdset1 = lamdset1 * 0.5
        mujiadeta = lamdset1 * mu0

        ind = torch.ge(predicted_labelprost, mujiadeta)
        inefewd = np.nonzero(ind).long()
        inefewd = inefewd.cpu()
        inefewd = inefewd.squeeze()
        nb = lamdset1 * mu1

        ind = torch.ge(maxmixsecon, nb)
        inefewdmsm = np.nonzero(ind).long()
        inefewdmsm = inefewdmsm.squeeze()
        inefewdn = set(inefewd.numpy()).intersection(set(inefewdmsm.numpy()))
        inefewdn = list(inefewdn)
        inefewdn = np.array(inefewdn)
        inefewdn = torch.from_numpy(inefewdn)
        inefewdn = inefewdn.long()

    cc = set(inefewdn.numpy()).intersection(set(ccaa.numpy()))
    cc = list(cc)
    cc = np.array(cc)
    cc = torch.from_numpy(cc)
    cc = cc.long()

    input_unseen_featureconfidence = torch.index_select(input_unseen_feature, 0, cc)
    input_unseen_labelconfidence = torch.index_select(predicted_labels, 0, cc)
    input_unseen_labelconfidencereal = torch.index_select(input_unseen_labelreal, 0, cc)

    input_unseen_featurezq = torch.cat((input_unseen_feature, input_unseen_featureconfidence), 0)
    input_unseen_labelmapupdata = torch.cat((input_unseen_labelmap, input_unseen_labelconfidence), 0)

    seenlabelone_hot = torch.zeros(input_unseen_labelmapupdata.size(0), ct + old_classes.size(0))
    seenlabelone_hot = seenlabelone_hot.scatter_(1, input_unseen_labelmapupdata.unsqueeze(1), 1)
    Wlr = compute_lrW(input_unseen_featurezq, seenlabelone_hot, 0.1)
    predictst = torch.mm(input_unseen_feature, Wlr)
    predictst = guiyihua(predictst)
    predicted_labelprost, predicted_labelsupdatae = torch.max(predictst, 1)

    saccdis = accpred(predicted_labelsupdatae, input_unseen_labelreal)
    print('增加confident样本后所有类变好了吗',
          saccdis * 100)
    input_unseen_labelmapunsee1lrdist = predicted_labelsupdatae[inefewdex]

    prddis = predicted_labelsdis[inefewdex]
    prdlr = predicted_labels[inefewdex]
    print('增加confident样本后新类变好了吗')
    saccdis = accpred(prddis, input_unseen_labelmapreal1)
    sacclr = accpred(prdlr, input_unseen_labelmapreal1)
    saccdislr = accpred(input_unseen_labelmapunsee1lrdist, input_unseen_labelmapreal1)

    return inputseend1, input_unseen_labelmapsee1, inputunseen1, input_unseen_labelmapunsee1lrdist


def compute_lrWgraphtwolabel(X, train_Ya, beta, gama, train_Xaselectleft, train_Yaselectleft):
    seenlabelone_hot = torch.zeros(train_Ya.size(0), ct + cs)
    seenlabelone_hot = seenlabelone_hot.scatter_(1, train_Ya.unsqueeze(1), 1)

    Xtt = X.transpose(0, 1)
    temp1 = torch.mm(Xtt, X)
    d = X.size(1)
    temp3 = temp1 + beta * torch.eye(d)

    train_Xaselectlefttt = train_Xaselectleft.transpose(0, 1)
    G, L = cal_rbf_dist(train_Xaselectleft.numpy(), n_neighbors=10, t=1)
    L = torch.from_numpy(L)
    L = L.float()
    temp1w = torch.mm(train_Xaselectlefttt, L)
    tempXLX = torch.mm(temp1w, train_Xaselectleft)

    temp2 = torch.mm(Xtt, seenlabelone_hot)

    temp31 = temp3 + gama * tempXLX
    W1 = torch.inverse(temp31)
    W = torch.mm(W1, temp2)
    return W


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def consensus_scorefx(t_feats, Yt, t_codes, t_centers, s_feats, s_labels, s_centers):
    # Calculate the consensus score of cross-domain matching
    setp = 0

    s_centers = F.normalize(s_centers, p=2, dim=-1)
    t_centers = F.normalize(t_centers, p=2, dim=-1)
    simis = torch.matmul(s_centers, t_centers.transpose(0, 1))
    s_index = simis.argmax(dim=1)
    t_index = simis.argmax(dim=0)
    map_s2t = [(i, s_index[i].item()) for i in range(len(s_index))]
    map_t2s = [(t_index[i].item(), i) for i in range(len(t_index))]
    inter = [a for a in map_s2t if a in map_t2s]
    Xsc = torch.FloatTensor([])
    Ysc = torch.LongTensor([])
    Xtc = torch.FloatTensor([])
    Ytc = torch.LongTensor([])
    Ytreal = torch.LongTensor([])
    cscom = len(inter)
    filtered_inter = []

    for i, j in inter:
        si_index = (s_labels == i).squeeze().nonzero(as_tuple=False)
        si_index = si_index.squeeze()
        tj_index = (t_codes == j).squeeze().nonzero(as_tuple=False)
        tj_index = tj_index.squeeze()
        si_feat = s_feats[si_index, :]
        Xsc = torch.cat((Xsc, si_feat), 0)
        filtered_cluster_label = torch.ones(si_feat.size(0)) + setp - 1
        Ysc = torch.cat((Ysc, filtered_cluster_label), 0)

        tj_feat = t_feats[tj_index, :]
        tjy = Yt[tj_index]
        Ytreal = torch.cat((Ytreal, tjy), 0)
        Xtc = torch.cat((Xtc, tj_feat), 0)
        filtered_cluster_label = torch.ones(tj_feat.size(0)) + setp - 1
        Ytc = torch.cat((Ytc, filtered_cluster_label), 0)
        setp = setp + 1
        filtered_inter.append((i, j))

    Ytc = Ytc.long()
    Ysc = Ysc.long()
    Ytreal = Ytreal.long()
    new_acct = cluster_acc(Ytreal.numpy(), Ytc.numpy())
    indcs = torch.le(Ytreal, cs)
    inefewdext = np.nonzero(indcs).long()
    inefewdext = inefewdext.squeeze()
    acctseen = inefewdext.size(0) / Ytreal.size(0)

    seenlabelone_hot = torch.zeros(Ytc.size(0), cscom)
    seenlabelone_hot = seenlabelone_hot.scatter_(1, Ytc.unsqueeze(1), 1)
    Wlr = compute_lrW(Xtc, seenlabelone_hot, 0.1)
    predictst = torch.mm(Xsc, Wlr)
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    qfda = torch.sum(Ysc == predicted_labels)
    scores = qfda / Ysc.size(0)
    #
    seenlabelone_hot = torch.zeros(Ysc.size(0), cscom)
    seenlabelone_hot = seenlabelone_hot.scatter_(1, Ysc.unsqueeze(1), 1)
    Wlr = compute_lrW(Xsc, seenlabelone_hot, 0.1)
    predictst = torch.mm(Xtc, Wlr)
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    qfda = torch.sum(Ytc == predicted_labels)
    scoret = qfda / Ytc.size(0)

    score = (scores + scoret) / 2

    seenRs = torch.FloatTensor([])
    seenRt = torch.FloatTensor([])
    ij = 0
    for i, j in inter:
        temp2 = torch.nonzero(Ysc == ij)
        indx = temp2.squeeze().long()
        Rssel = Xsc[indx, :]
        Rssel1 = torch.mean(Rssel, 0)
        Rssel1 = Rssel1.unsqueeze(1)
        Rssel1 = Rssel1.transpose(0, 1)
        seenRs = torch.cat((seenRs, Rssel1), 0)
        temp2 = torch.nonzero(Ytc == ij)
        indx = temp2.squeeze().long()
        Rssel = Xtc[indx, :]
        Rssel1 = torch.mean(Rssel, 0)
        Rssel1 = Rssel1.unsqueeze(1)
        Rssel1 = Rssel1.transpose(0, 1)
        seenRt = torch.cat((seenRt, Rssel1), 0)
        ij = ij + 1
    # dist1 = pairwise_distances(seenRs, seenRt)

    return score, scores, scoret, filtered_inter, new_acct, acctseen, seenRs, seenRt



def clus_acc(t_codes, t_gt, mapping, gt_freq, sub_score):
    # Print the status of matching
    print(mapping, len(sub_score))
    for i, (src, tgt) in enumerate(mapping):
        mask = t_codes == tgt
        i_gt = torch.masked_select(t_gt, mask)
        i_acc = ((i_gt == src).sum().float()) / len(i_gt)
        if src in gt_freq:
            gt_cnt = gt_freq[src]
        else:
            gt_cnt = 1.0
        recall = i_acc * len(i_gt) / gt_cnt
        print(
            '{:0>2d}th Cluster ACC:{:.2f} Correct/Total/GT {:0>2d}/{:0>2d}/{:0>2d} Precision:{:.3f} Recall:{:.3f} Score:{:.2f}'.format(
                src, i_acc.item(), (i_gt == src).sum().item(), len(i_gt), int(gt_cnt), i_acc, recall, sub_score[i]))


def fix_k(scores, n=3):
    # Stopping critetion: stop searching if K holds a certain value for n times.
    if len(scores) < n:
        return False
    scores = scores[-n:]
    flag = 0.0
    for i in scores:
        if i == scores[-n]:
            flag += 1
    if flag == n:
        return True
    else:
        return False


def detect_continuous_machpairsstable(scores, n, qting):
    if len(scores) < n:
        return False
    nas = len(scores) - 1
    A = scores[nas]
    A1 = scores[nas - 1]
    A2 = scores[nas - 2]

    AA1 = abs(A - A1)
    A1A2 = abs(A2 - A1)
    Am = abs(AA1 - A1A2)
    if Am <= qting:
        return True
    else:
        return False


def detect_continuous_drop(scores, n, con=False):
    # Stopping Criterion: stop searching in a round if the score drops continuously for n times.
    if len(scores) < n:
        return False
    scores = scores[-n:]
    flag = 0.0
    if con:
        for i in range(1, n):
            if scores[-i] <= scores[-(i + 1)]:
                flag += 1
    else:
        flag = 0.0
        for i in scores:
            if i <= scores[-n]:
                flag += 1
    if flag >= n - 1:
        return True
    else:
        return False


def sklearn_kmeans(feat, num_centers, init=None):
    if init is not None:
        kmeans = KMeans(n_clusters=num_centers, init=init, random_state=0).fit(feat.detach().cpu().numpy())
    else:
        kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(feat.detach().cpu().numpy())
    center, t_codes = kmeans.cluster_centers_, kmeans.labels_
    # score = sklearn.metrics.silhouette_score(feat.cpu().numpy(), t_codes)
    score = 0
    return torch.from_numpy(center), torch.from_numpy(t_codes), score


def map_labelrev(mapped_label, classes):  # 0-40 转换为0-50
    mapped_labelrev = torch.LongTensor(mapped_label.size())
    for i in range(classes.size(0)):
        mapped_labelrev[mapped_label == i] = classes[i]
    return mapped_labelrev


def zuizhongfenlei(Rall, train_X, train_Y, new_class, old_classes, alpha):
    ct = new_class.size(0)
    #
    # loss111 = pairwise_distances(Xtest, Rall)
    # predprod = chuanboyici(loss111)
    # predicted_labelprost1, predicted_labels1 = torch.max(predprod, 1)
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Ytest.numpy()]))
    # all_accd, old_accd, new_accd = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels1.numpy(), mask=mask,
    #                                                 eval_funcs=args.eval_funcs, save_name=None,
    #                                                 writer=None)
    # print('distance test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_accd, old_accd, new_accd))

    seenlabelone_hot = torch.zeros(train_Y.size(0), ct + old_classes.size(0))
    seenlabelone_hot = seenlabelone_hot.scatter_(1, train_Y.unsqueeze(1), 1)
    Wlr = compute_lrW(train_X, seenlabelone_hot, alpha)
    predictst = torch.mm(Xtest, Wlr)
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels.numpy(), mask=mask,
                                                    eval_funcs=args.eval_funcs, save_name=None,
                                                    writer=None)
    print('lr test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    #
    #
    # loss111 = pairwise_distances(Xunlabeled, Rall)
    # predprod = chuanboyici(loss111)
    # predicted_labelprost1, predicted_labels2 = torch.max(predprod, 1)
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Yunlabeled.numpy()]))
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels2.numpy(),
    #                                                 mask=mask,
    #                                                 eval_funcs=args.eval_funcs, save_name=None,
    #                                                 writer=None)
    # print(
    #     'distance trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    predictst = torch.mm(Xunlabeled, Wlr)
    predicted_labelprost, predicted_labels1 = torch.max(predictst, 1)

    all_acc1, old_acc1, new_acc1 = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels1.numpy(),
                                                       mask=mask,
                                                       eval_funcs=args.eval_funcs, save_name=None,
                                                       writer=None)
    print(
        'lr trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc1, old_acc1, new_acc1))

    #
    #
    # pretrain_clst = classifier.CLASSIFIER(train_X.cuda(), train_Y.cuda(), old_classes.size(0) + ct, train_X.size(1),
    #                                       device,
    #                                       0.001, 0.5, 20, 100, '')
    # predictst = pretrain_clst.model(Xtest.cuda())
    # predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    # predicted_labels = predicted_labels.cpu().numpy()
    # mask = np.array([])
    # mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
    #                                  else False for x in Ytest.numpy()]))
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels, mask=mask,
    #                                                 eval_funcs=args.eval_funcs, save_name=None,
    #                                                 writer=None)
    # print('svm test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
    #
    # predictst = pretrain_clst.model(Xunlabeled.cuda())
    # predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    # predicted_labels = predicted_labels.cpu()
    # mask = np.array([])
    # mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
    #                                  else False for x in Yunlabeled.numpy()]))
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels.numpy(),
    #                                                 mask=mask,
    #                                                 eval_funcs=args.eval_funcs, save_name=None,
    #                                                 writer=None)
    # print(' svm trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    return all_acc, old_acc, new_acc, all_acc1, old_acc1, new_acc1,predicted_labels


def zuizhongfenleiphionlyfenlei(predictsttest, predictsttrain,   old_classes):
    predicted_labelprost, predicted_labels = torch.max(predictsttest, 1)
    predicted_labels = predicted_labels.cpu().numpy()
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Ytest.numpy()]))
    all_acc1, old_acc1, new_acc1 = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels, mask=mask,
                                                       eval_funcs=args.eval_funcs, save_name=None,
                                                       writer=None)
    print('ce test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc1, old_acc1, new_acc1))

    predicted_labelprost, predicted_labels = torch.max(predictsttrain, 1)
    predicted_labels = predicted_labels.cpu()
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Yunlabeled.numpy()]))
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels.numpy(),
                                                    mask=mask,
                                                    eval_funcs=args.eval_funcs, save_name=None,
                                                    writer=None)
    print(
        'ce  trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    return all_acc1, old_acc1, new_acc1, all_acc, old_acc, new_acc


def zuizhongfenleiphi(Rall, train_X, train_Y, new_class, old_classes, Xtest, Xunlabeled,ct):
    # loss111 = pairwise_distances(Xtest, Rall)
    # predprod = chuanboyici(loss111)
    # predicted_labelprost1, predicted_labels1 = torch.max(predprod, 1)
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Ytest.numpy()]))
    # all_accd, old_accd, new_accd = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels1.numpy(),
    #                                                    mask=mask,
    #                                                    eval_funcs=args.eval_funcs, save_name=None,
    #                                                    writer=None)
    # print('distance test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_accd, old_accd, new_accd))

    seenlabelone_hot = torch.zeros(train_Y.size(0), ct + old_classes.size(0))
    seenlabelone_hot = seenlabelone_hot.scatter_(1, train_Y.unsqueeze(1), 1)
    Wlr = compute_lrW(train_X, seenlabelone_hot, 0.1)
    predictst = torch.mm(Xtest, Wlr)
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)

    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels.numpy(), mask=mask,
                                                    eval_funcs=args.eval_funcs, save_name=None,
                                                    writer=None)
    print('lr test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    # loss111 = pairwise_distances(Xunlabeled, Rall)
    # predprod = chuanboyici(loss111)
    # predicted_labelprost1, predicted_labels2 = torch.max(predprod, 1)
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Yunlabeled.numpy()]))
    # all_acc, old_acc, new_acc = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels2.numpy(),
    #                                                 mask=mask,
    #                                                 eval_funcs=args.eval_funcs, save_name=None,
    #                                                 writer=None)
    # print(
    #     'distance trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

    predictst = torch.mm(Xunlabeled, Wlr)
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)

    all_accd, old_accd, new_accd = log_accs_from_preds(y_true=Yunlabeled.numpy(), y_pred=predicted_labels.numpy(),
                                                       mask=mask,
                                                       eval_funcs=args.eval_funcs, save_name=None,
                                                       writer=None)
    print('lr trainunlabeled Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_accd, old_accd, new_accd))
    return all_acc, old_acc, new_acc, all_accd, old_accd, new_accd


def detect_continuous_raisze(scores, n, con=False):
    # Stopping Criterion: stop searching in a round if the score drops continuously for n times.
    if len(scores) < n:
        return False
    scores = scores[-n:]
    flag = 0.0
    if con:
        for i in range(1, n):
            if scores[-i] >= scores[-(i + 1)]:
                flag += 1
    else:
        flag = 0.0
        for i in scores:
            if i >= scores[-n]:
                flag += 1
    if flag >= n - 1:
        return True
    else:
        return False


def Xslevave(Xseen, Yseen, inter):
    cshar = len(inter)
    indxs = torch.zeros(cshar, 1)
    for i, (src, tgt) in enumerate(inter):
        indxs[i] = src
    indxs = indxs.long()
    indxs = indxs.squeeze()
    indx1a = torch.linspace(0, cs - 1, steps=cs)
    indx1a = indx1a.long()
    index3leve = np.setdiff1d(indx1a, indxs)
    index3leve = torch.from_numpy(index3leve)
    index3leve = index3leve.long()
    Xslev = torch.FloatTensor([])
    Yslevmap = torch.LongTensor([])
    seenRslev = torch.FloatTensor([])
    for i in range(index3leve.size(0)):
        ij = index3leve[i]
        temp2 = torch.nonzero(Yseen == ij)
        indx = temp2.squeeze().long()
        Rssel = Xseen[indx, :]
        Xslev = torch.cat((Xslev, Rssel), 0)
        filtered_cluster_label = torch.ones(Rssel.size(0)) + ij - 1
        Yslevmap = torch.cat((Yslevmap, filtered_cluster_label), 0)
        Rssel1 = torch.mean(Rssel, 0)
        Rssel1 = Rssel1.unsqueeze(1)
        Rssel1 = Rssel1.transpose(0, 1)
        seenRslev = torch.cat((seenRslev, Rssel1), 0)
    Yslevmap = Yslevmap.long()
    return Xslev, Yslevmap, seenRslev, index3leve


def get_tgt_centers(step, init_center):
    best_disloss = 1000000
    drop = 2
    drop_con = True
    drop_stop = True
    best_score = 0.0
    final_n_center = None
    inter_memo = {}
    search = True
    warmup_steps = 400
    warmup = False
    cs = old_classes.size(0)
    interval = cs * 0.1
    interval = int(interval)

    max_center = cs * 4
    n_center = init_center
    num_centers = n_center
    score_his = []
    t_codes_dic = {}
    t_centers_dic = {}
    disloss_dic = []
    seenRs_dic = {}
    seenRt_dic = {}
    seffewall = []
    if search:
        while search and n_center <= max_center:
            t_centers, t_codes, sh_score = sklearn_kmeans(Xunlabeled, n_center)
            score, scores, scoret, inter, new_acc, accxtseen, seenRs11, seenRt11 = consensus_scorefx(Xunlabeled,
                                                                                                     Yunlabeled,
                                                                                                     t_codes, t_centers,
                                                                                                     Xseen, Yseen,
                                                                                                     seenRs)  # 目标域特征，聚类结果，聚类中心，源域特征，标记，中心
            disloss = loss_func(seenRs11, seenRt11)
            inter_memo[n_center] = inter
            t_codes_dic[n_center] = t_codes
            t_centers_dic[n_center] = t_centers
            seenRs_dic[n_center] = seenRs11
            seenRt_dic[n_center] = seenRt11

            if score > best_score:
                final_n_center = n_center
                best_score = score
            #
            # if disloss < best_disloss:
            #     final_n_center = n_center
            #     best_disloss = disloss
            seffew = len(inter)
            print('总类数k，源码目标域共享类别数c,中心差异，本轮分数，源域分数，目标域分数，目标域样本聚类正确率,选出的目标域已知类样本正确率', t_centers.size(0), seffew, disloss,
                  score, scores, scoret,
                  new_acc, accxtseen)

            score_his.append(score)
            seffewall.append(seffew)
            disloss_dic.append(disloss)
            if t_centers.size(0)== cs+ ct:  # 寻找k
                search = False
            # if detect_continuous_machpairsstable(seffewall, n=2, qting=10) and drop_stop and detect_continuous_drop(
            #         score_his, n=drop, con=drop_con):  # 寻找k
            #     search = False
            # if detect_continuous_raisze(disloss_dic, 2, con=drop_con) and drop_stop and detect_continuous_drop(score_his, 2,  con=drop_con):  # 寻找k
            #     search = False

            n_center += interval

        inter = inter_memo[final_n_center]
        n_center = final_n_center
        t_centers = t_centers_dic[final_n_center]
        t_codes = t_codes_dic[final_n_center]
        seenRs11 = seenRs_dic[n_center]
        seenRt11 = seenRt_dic[n_center]
    return Xunlabeled, t_codes, t_centers, inter, seenRs11, seenRt11



def XtsXtuseparatethesecondstep(Xunlabeled, Yunlabeled, t_codes, mapping, csold):
    Xts = torch.FloatTensor([])

    Ytspre = torch.LongTensor([])

    maskxs = torch.LongTensor([])
    Ytsreal = torch.LongTensor([])
    Ytureal = torch.LongTensor([])
    tstu = np.unique(t_codes)
    tstu = torch.from_numpy(tstu).long()
    tscommon = len(mapping)
    jiq = 0
    ts1 = torch.zeros(tscommon).long()
    for s_index, t_index in mapping:
        ts1[jiq] = t_index
        jiq = jiq + 1

    index3 = np.setdiff1d(tstu, ts1)
    index3 = torch.from_numpy(index3)
    tu1 = index3.long()

    Xnew = torch.FloatTensor([])
    Ynew = torch.LongTensor([])
    Xnew = torch.FloatTensor([])
    Ynew = torch.LongTensor([])
    j = 0
    for i in range(tu1.size(0)):
        ij = tu1[i]
        temp2 = torch.nonzero(t_codes == ij)
        indx = temp2.squeeze().long()
        # print(indx.size())
        if indx.size() == torch.Size([]):
            j = i - 1
        else:
            j = j + 1
            Rssel = Xunlabeled[indx, :]
            Xnew = torch.cat((Xnew, Rssel), 0)
            filtered_cluster_label = csold * torch.ones(Rssel.size(0)) + j - 1
            Ynew = torch.cat((Ynew, filtered_cluster_label), 0)
            Rytel1 = Yunlabeled[indx]
            Ytureal = torch.cat((Ytureal, Rytel1), 0)

    new_classes11 = np.unique(Ynew)
    new_classes11 = torch.from_numpy(new_classes11)

    Ynew = Ynew.long()
    new_classes11 = new_classes11.long()

    for s_index, t_index in mapping:
        t_mask = t_codes == t_index
        i_index = t_mask.squeeze().nonzero(as_tuple=False)
        i_index = i_index.squeeze()
        Rstel = Xunlabeled[i_index, :]
        Xts = torch.cat((Xts, Rstel), 0)
        Rytel = Yunlabeled[i_index]
        Ytsreal = torch.cat((Ytsreal, Rytel), 0)
        filtered_cluster_label = s_index * torch.ones(Rstel.size(0))
        Ytspre = torch.cat((Ytspre, filtered_cluster_label), 0)
    Ytspre = Ytspre.long()

    return Xts, Ytspre, Xnew, Ynew, new_classes11, Ytsreal, Ytureal


def seenlabelone_hotmmR(Yseen, phiRs):
    Yseen = Yseen.long()
    csuqi = np.unique(Yseen)
    cs1 = len(csuqi)
    seenlabelone_hot = torch.zeros(Yseen.size(0), cs1)
    seenlabelone_hot = seenlabelone_hot.scatter_(1, Yseen.unsqueeze(1), 1)
    YF = torch.mm(seenlabelone_hot, phiRs)
    return YF







def farawayRs(Xnew, seenRs, Ynew, Ytureal, cs,ratio,ntest_class):
    predprod = pairwise_distances(Xnew, seenRs)
    predicted_labelprost1, predicted_labels1 = torch.max(predprod, 1)
    ewfe = torch.max(predicted_labelprost1)
    fewe = torch.min(predicted_labelprost1)
    fewfew1 = 0.1 * (ewfe - fewe)
    mu0 = fewe + ratio * fewfew1
    print(mu0)
    ind = torch.ge(predicted_labelprost1, mu0)
    inefewd = np.nonzero(ind).long()
    inefewd = inefewd.cpu()
    inefewd = inefewd.squeeze()

    Xnewselect = Xnew[inefewd, :]
    Ynewselect = Ynew[inefewd]
    Ynewselectreal = Ytureal[inefewd]
    qfda = torch.ge(Ynewselectreal, cs)
    qfda = torch.sum(qfda)
    sacc11 = qfda / Ynewselect.size(0)
    print(sacc11)
    sacc11new_acct = cluster_acc(Ynewselectreal.numpy(), Ynewselect.numpy())
    print(sacc11new_acct)

    # t_centers11, Ynewt_codes, sh_score1 = sklearn_kmeans(Xnew, ntest_class)
    # sacc11new_acct = cluster_acc(Ytureal.numpy(), Ynewt_codes.numpy())
    # print(sacc11new_acct)
    #
    # t_centers1, Ynewt_codes1, sh_score1 = sklearn_kmeans(Xnewselect, ntest_class)
    # sacc11new_acct = cluster_acc(Ynewselectreal.numpy(), Ynewt_codes1.numpy())
    # print(sacc11new_acct)
    return  Xnewselect, Ynewselect, Ynewselectreal


def L2Norm(trainFeatures_proj):
    aaa = torch.sqrt(torch.sum(torch.mul(trainFeatures_proj, trainFeatures_proj), 1))
    aaa = aaa.unsqueeze(1)
    aaa = aaa.repeat(1, trainFeatures_proj.size(1))
    y = trainFeatures_proj / aaa
    return y


def chuanboyici(target_distances):
    expMatrix = torch.exp(-1 * target_distances)
    a = torch.sum(expMatrix, 1).unsqueeze(dim=1)
    be = target_distances.size(1)
    fewf = a.repeat(1, be)
    fewf = fewf + 0.0001
    probMatrix = expMatrix / fewf
    probMatrix1 = L2Norm(probMatrix)
    return probMatrix1

def seenRtqiu(Xsc, Ysc):
    seenRs = torch.FloatTensor([])

    tstu = np.unique(Ysc)
    tstu = torch.from_numpy(tstu)
    for i in range(tstu.size(0)):
        ij = tstu[i]
        temp2 = torch.nonzero(Ysc == ij)
        indx = temp2.squeeze().long()
        Rssel = Xsc[indx, :]
        if Rssel.size(0) !=  0 :
            if Rssel.size(0) == Xsc.size(1):
                Rssel1 = Rssel
                Rssel1 = Rssel1.unsqueeze(1)
                Rssel1 = Rssel1.transpose(0, 1)
            else:
                Rssel1 = torch.mean(Rssel, 0)
                Rssel1 = Rssel1.unsqueeze(1)
                Rssel1 = Rssel1.transpose(0, 1)


        seenRs = torch.cat((seenRs, Rssel1), 0)

    return  seenRs


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot


class AE(torch.nn.Module):
    def __init__(self, dim, latent_size):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dim, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, latent_size),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z


def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x.view(-1, 768), x.view(-1, 768), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return (BCE + KLD) / x.size(0)



def data_iterator(train_x,  Yseen, batch_size,isSeen):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(train_x))
        np.random.shuffle(idxs)
        shuf_visual = train_x[idxs]

        shuf_label = Yseen[idxs]
        shuf_isSeen = isSeen[idxs]

        afew =  int(len(train_x)) / batch_size
        afe = int(afew)

        for batch_idx in range(0, afe *batch_size, batch_size):
            visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]

            shuf_label_batch = shuf_label[batch_idx:batch_idx + batch_size]
            shuf_isSeen_batch = shuf_isSeen[batch_idx:batch_idx + batch_size]


            visual_batch = visual_batch.float().cuda()

            shuf_label_batch = shuf_label_batch.long().cuda()
            shuf_isSeen_batch = shuf_isSeen_batch.cuda()
            yield visual_batch, shuf_label_batch,shuf_isSeen_batch

def cross_entropy_loss(predictions, targets):
    loss = F.cross_entropy(predictions, targets)
    return loss
def loss_funckl(feat,cluster_centers,alpha):
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    # p_clone = p.data.clone()
    # p = torch.autograd.Variable(p_clone, requires_grad=True)
    p = p.detach()
    q_clone = q.data.clone()
    q = torch.autograd.Variable(q_clone, requires_grad=True)
    log_q = torch.log(q)


    loss = F.kl_div(log_q, p)
    return loss, p
def prototypical_probability(zi, muk, temperature=1.0):
    distances = pairwise_distances(zi, muk)  # Calculate Euclidean distance between zi and µk
    logits = -distances / temperature  # Divide distances by temperature and negate them
    probabilities = F.softmax(logits, dim=0)  # Apply softmax to obtain probabilities
    probabilities = L2Norm( probabilities)
    return probabilities

def  separateXsXu(Xs, Ys, Xu, Yu, allclass):
    seenRs = seenRtqiu(Xs, Ys)

    probabilities = prototypical_probability(Xu, seenRs, temperature=1.0)
    predicted_labelprost1, predicted_labels = torch.max(probabilities, 1)
    predicted_labels = predicted_labels.cpu()

    mu0 = torch.mean(predicted_labelprost1)
    std = torch.std(predicted_labelprost1)
    ind = torch.ge(predicted_labelprost1, mu0 - 0.5 * std)
    inefewd = np.nonzero(ind).long()
    inefewd = inefewd.cpu()
    cc = inefewd.squeeze()
    Xts = torch.index_select(Xu, 0, cc)
    Yts = torch.index_select(predicted_labels, 0, cc)
    Ytsreal = torch.index_select(Yu, 0, cc)
    acc1 = torch.sum(Yts == Ytsreal) / Ytsreal.size(0)
    print('预分类已知类样本的分类正确率', acc1)
    Rts = seenRtqiu(Xts, Yts)

    ind = torch.le(predicted_labelprost1, mu0 - std)
    inefewd = np.nonzero(ind).long()
    inefewd = inefewd.cpu()
    cc1 = inefewd.squeeze()
    Xtu = torch.index_select(Xu, 0, cc1)
    Ytureal = torch.index_select(Yu, 0, cc1)
    Rtu, Ytu, st_scoreper = sklearn_kmeans(Xtu, ct)
    Ytu = Ytu + cs
    new_acctunseen1 = cluster_acc(Ytureal.numpy(), Ytu.numpy())
    print('z xu xu clusteracc正确率', new_acctunseen1 * 100)

    return Xts, Yts, Rts, Xtu, Ytu, Rtu

from sklearn.neighbors import NearestNeighbors
def find_k_nearest_neighbors_with_pseudo_labels(unlabeled_samples, pseudo_labels, Yreal, Xsee, Ysee, k,delat):
    # Create the NearestNeighbors object and fit it with the unlabeled samples
    new_acctunseen1 = accpred(torch.from_numpy(Yreal), torch.from_numpy(pseudo_labels))
    print("选择前的样本分类正确率", new_acctunseen1)
    new_acctunseen1 = cluster_acc(Yreal, pseudo_labels)
    #print("选择前的样本聚类正确率", new_acctunseen1)

    # 初始化 NearestNeighbors 模型
    n_neighbors = k # 要找的邻居数量
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn_model.fit(Xsee)

    # 找到 unlabeled_samples 的邻居索引
    distances, indices = nn_model.kneighbors(unlabeled_samples)

    # 这里的 indices 就是 unlabeled_samples 的邻居索引，distances 是相应的距离
    # 可以根据 indices 在 Xsee 中获取实际的邻居样本

    # 将 indices 转换为 PyTorch tensor
    #indices_tensor = torch.tensor(indices)
    #
    # nbrs = NearestNeighbors(n_neighbors=k).fit(unlabeled_samples)
    #
    # # Find the indices of the k-nearest neighbors for all unlabeled samples
    # distances, indices = nbrs.kneighbors(unlabeled_samples)
    #
    # # Create a list to store the results

    num_neighbors_with_same_labelpercentage = torch.zeros(unlabeled_samples.shape[0])
    # Iterate through each unlabeled sample
    for i in range(unlabeled_samples.shape[0]):
        neighbors_indices = indices[i]
        neighbors_labels = Ysee[neighbors_indices]
        sample_label = pseudo_labels [i] # Pseudo-label of the sample itself
        num_neighbors_with_same_label = (neighbors_labels == sample_label).sum()
        # Count the number of neighbors with the same pseudo-label as the sample
        num_neighbors_with_same_labelpercentage[i] = num_neighbors_with_same_label / k

    indices_above_threshold = num_neighbors_with_same_labelpercentage > delat
    Yrealneigb = Yreal[indices_above_threshold]
    unlabeled_samplesselect = unlabeled_samples[indices_above_threshold]
    pseudo_labelssselect = pseudo_labels[indices_above_threshold]
    unlabeled_samplesselect = torch.from_numpy(unlabeled_samplesselect)
    pseudo_labelssselect = torch.from_numpy(pseudo_labelssselect)
    Yrealneigb = torch.from_numpy(Yrealneigb)
    new_acctunseen1 = cluster_acc(Yrealneigb.numpy(), pseudo_labelssselect.numpy())
   # print("选择后的样本聚类正确率", new_acctunseen1)
    new_acctunseen1 = accpred(Yrealneigb, pseudo_labelssselect)
    print("选择后的已知类样本分类正确率选择后的已知类样本分类正确率选择后的已知类样本分类正确率", new_acctunseen1)
    ab1 = Yrealneigb.size(0)
    ab2 = len(Yreal)
    if ab1< ab2:
        indices_above_threshold = num_neighbors_with_same_labelpercentage <= delat
        Yrealneigb1 = Yreal[indices_above_threshold]
        unlabeled_samplesselect1 = unlabeled_samples[indices_above_threshold]
        pseudo_labelssselect1 = pseudo_labels[indices_above_threshold]
        unlabeled_samplesselect1 = torch.from_numpy(unlabeled_samplesselect1)
        pseudo_labelssselect1 = torch.from_numpy(pseudo_labelssselect1)
        Yrealneigb1 = torch.from_numpy(Yrealneigb1)
        new_acctunseen11 = cluster_acc(Yrealneigb1.numpy(), pseudo_labelssselect1.numpy())
        print("选择后的未知类样本聚类正确率", new_acctunseen11)
        qfda = torch.ge(Yrealneigb1, cs)
        qfda = torch.sum(qfda)
        sacc11 = qfda / Yrealneigb1.size(0)
        print("选择的未知知类detection正确率选择的未知知类detection正确率选择的未知知类detection正确率选择的未知知类detection正确率选择的未知知类detection正确率")
        print(sacc11)
        return unlabeled_samplesselect, pseudo_labelssselect, Yrealneigb, num_neighbors_with_same_labelpercentage, unlabeled_samplesselect1, pseudo_labelssselect1, Yrealneigb1
    else:
        return unlabeled_samplesselect, pseudo_labelssselect, Yrealneigb, num_neighbors_with_same_labelpercentage, torch.FloatTensor([]),torch.FloatTensor([]), torch.FloatTensor([])



import torch.nn as nn
class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 768)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
    def inference(self, x, c):

        if x.dim() > 2:
            x = x.view(-1, 768)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z,c)

        return recon_x, means, log_var, z
    # def inference(self, z, c=None):
    #
    #     recon_x = self.decoder(z, c)
    #
    #     return recon_x



class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=cs)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=cs)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
def quedingkfu(Xseen,Xunlabeled, Yseen, Yunlabeled, cs):
    train_X = torch.cat((Xseen, Xunlabeled), 0)
    train_Yreal = torch.cat((Yseen, Yunlabeled), 0)
    ns = Xseen.size(0)
    cuestimate = int(3 * cs)
    class_counts = torch.bincount(Yseen)
    #numberrr = torch.min(class_counts)
    while (True):
        max_class_matrix = [None] * (cs + cuestimate)
        classindx = [None] * (cs + cuestimate)
        z_centers, Ysclu, st_score = sklearn_kmeans(train_X, cs + cuestimate)
        cluster_stats = []
        tem1 = torch.linspace(0, cs + cuestimate - 1, steps=cs + cuestimate)
        tem1 = tem1.long()
        for i in range(tem1.size(0)):
            print("Cluster:", i)
            cluster_idx = i
            idn1 = Ysclu == cluster_idx
            geshupercluster = torch.nonzero(idn1)
            geshuperclustersize = geshupercluster.size(0)
            cluster_known_classes = Yseen[idn1[0:ns]]
            known_class_counts = torch.zeros(cs, dtype=torch.int32)
            for class_idx in range(cs):
                known_class_counts[class_idx] = (cluster_known_classes == class_idx).sum().item()
            cluster_stats.append((cluster_idx, known_class_counts))
            seenduo = torch.nonzero(known_class_counts)
            print("Class:", seenduo)
            abns1 = seenduo.size(0)
            seenduo = torch.nonzero(known_class_counts)
            print("Class:", seenduo)
            abns1 = seenduo.size(0)
            geshu = 0 * seenduo
            meiper = 0 * seenduo
            meipergeshu = 0 * seenduo
            for jjj in range(seenduo.size(0)):
                iia = seenduo[jjj]
                geshu[jjj] = torch.sum(cluster_known_classes == iia)
                meipergeshu[jjj] = class_counts[iia]
                meiper[jjj] = (0.5 / abns1) * geshuperclustersize

            if (abns1 > 1):
                greater_than_or_equal = geshu > 8  #3越大，满足条件的越少，dayudlei 越小  越不会增加类别
                dayudlei = torch.sum(greater_than_or_equal)
                if (dayudlei > 1):
                    cuestimate = cuestimate + dayudlei
                    break
                elif (dayudlei == 1):
                    # 获取簇中类别数最多的类
                    max_class_idx = torch.nonzero(known_class_counts == known_class_counts.max()).squeeze()
                    classindx[cluster_idx] =  torch.argmax(known_class_counts).item()
                    # 将类别数最多的类存储在矩阵中
                    max_class_matrix[cluster_idx] = max_class_idx.item()
                elif (dayudlei == 0):
                    continue

            elif (abns1 == 1  ):
                if geshu > 2: #3越大，满足条件的越少，未知类越多
                    # 获取簇中类别数最多的类
                    max_class_idx = torch.nonzero(known_class_counts == known_class_counts.max()).squeeze()
                    classindx[cluster_idx] = torch.argmax(known_class_counts).item()
                    # 将类别数最多的类存储在矩阵中.item()
                    max_class_matrix[cluster_idx] = max_class_idx.item()

            elif (abns1 == 0):
                continue

        if i == tem1.size(0) - 1:
            break

    non_none_indices = [idx for idx, value in enumerate(max_class_matrix) if value is not None]
    seenclasses1 = [classindx[idx] for idx in non_none_indices]
    print("the clusters of Indices with seen classes:", non_none_indices)
    print(len(non_none_indices))
    print("Indices with seen classes:", seenclasses1)
    gejee = np.unique(seenclasses1)
    print(gejee)
    Xtallseen = torch.FloatTensor([])
    Ytallseen = torch.LongTensor([])
    Ytallrseen = torch.LongTensor([])
    Ysclus = Ysclu[0:ns]
    Yscluu = Ysclu[ns:Ysclu.size(0)]
    Yscluumapseen = torch.LongTensor([])
    for j in range(len(non_none_indices)):
        seen1 = seenclasses1[j]
        idnxper = Yscluu == torch.tensor(non_none_indices[j])
        Xt = Xunlabeled[idnxper]
        Yt = Yscluu[idnxper]
        Ytrseen = seen1 * torch.ones_like(Yt)
        Yscluumapseen = torch.cat((Yscluumapseen, Ytrseen), 0)  # seen lei
        Xtallseen = torch.cat((Xtallseen, Xt), 0)
        Ytallseen = torch.cat((Ytallseen, Yt), 0)  # juleibiaojie
        Ytr = Yunlabeled[idnxper]
        Ytallrseen = torch.cat((Ytallrseen, Ytr), 0)  # 真正标记

    acctselect11 = torch.sum(Ytallrseen <= cs) / Ytallrseen.size(0)
    new_acctunseen111 = cluster_acc(Ytallrseen.numpy(), Ytallseen.numpy())
    new_acctunseen1111 = torch.sum(Yscluumapseen == Ytallrseen) / Ytallrseen.size(0)
    print("选择的已知类是已知类的正确率，伪标记的正确率")
    print(acctselect11, new_acctunseen111, new_acctunseen1111)


    #
    # for j in range(len(non_none_indices)):
    #     seen1 = seenclasses1[j]
    #     idnxperseen = Ysclus == torch.tensor(non_none_indices[j])
    #     geshujuleiper = torch.nonzero(idnxperseen).size(0)
    #     geshuperi = class_counts[seen1]
    #     idnxper = Yscluu == torch.tensor(non_none_indices[j])
    #     Xt = Xunlabeled[idnxper]
    #     Yt = Yscluu[idnxper]
    #     Ytrseen = seen1 * torch.ones_like(Yt)
    #     Yscluumapseen = torch.cat((Yscluumapseen, Ytrseen), 0)  # seen lei
    #     Xtallseen = torch.cat((Xtallseen, Xt), 0)
    #     Ytallseen = torch.cat((Ytallseen, Yt), 0)  # juleibiaojie
    #     Ytr = Yunlabeled[idnxper]
    #     Ytallrseen = torch.cat((Ytallrseen, Ytr), 0)  # 真正标记
    #
    #     #if geshujuleiper > geshuperi * 0.3:
    #
    #
    #
    # acctselect11 = torch.sum(Ytallrseen <= cs) / Ytallrseen.size(0)
    # new_acctunseen111 = cluster_acc(Ytallrseen.numpy(), Ytallseen.numpy())
    # new_acctunseen1111 = torch.sum(Yscluumapseen == Ytallrseen) / Ytallrseen.size(0)
    # print("选择的已知类是已知类的正确率，伪标记的正确率")
    # print(acctselect11, new_acctunseen111, new_acctunseen1111)







    none_indices = [idx for idx, value in enumerate(max_class_matrix) if value is None]
    Xtall = torch.FloatTensor([])
    Ytall = torch.LongTensor([])
    Ytallr = torch.LongTensor([])
    Yscluumapunseen = torch.LongTensor([])
    print("Indices with None:", none_indices)
    print(len(none_indices))
    for j in range(len(none_indices)):
        Xt = Xunlabeled[Yscluu == torch.tensor(none_indices[j])]
        Yt = Yscluu[Yscluu == torch.tensor(none_indices[j])]
        Ytrseen = j * torch.ones_like(Yt) + cs
        Yscluumapunseen = torch.cat((Yscluumapunseen, Ytrseen), 0)
        Xtall = torch.cat((Xtall, Xt), 0)
        Ytall = torch.cat((Ytall, Yt), 0)
        Ytr = Yunlabeled[Yscluu == torch.tensor(none_indices[j])]
        Ytallr = torch.cat((Ytallr, Ytr), 0)

    acctselect = torch.sum(Ytallr >= cs) / Ytallr.size(0)
    new_acctunseen1222 = cluster_acc(Ytallr.numpy(), Ytall.numpy())
    new_acctunseen11112 = cluster_acc(Ytallr.numpy(), Yscluumapunseen.numpy())
    print("选择的未知类是未知类的正确率，伪标记的正确率")
    print(acctselect, new_acctunseen1222, new_acctunseen11112)
    cuestimate = len(none_indices)
    print(cuestimate)
    # t_centers1aa1, Ynewt_codesaa, sh_score1aa = sklearn_kmeans(Xtall, cuestimate)
    # sacc11new_acct = cluster_acc(Ytallr.numpy(), Ynewt_codesaa.numpy())
    # print("选择的未知类准确率")
    # print(sacc11new_acct)
    # Ynewt_codesaa = Ynewt_codesaa + cs
    #

    return Xtallseen, Xtall, Yscluumapseen, Yscluumapunseen, Ytallrseen, Ytallr, cuestimate


import torch.nn.functional as F

def cosine_similarity(ziu, zju):

    ziu_norm = torch.norm(ziu)
    zju_norm = torch.norm(zju)
    similarity = torch.dot(ziu, zju) / (ziu_norm * zju_norm)
    return similarity

def pairwise_pseudo_labels(ziu_list, zju_list,simil):
    """
    Compute the pairwise pseudo-labels for each pair of samples based on cosine similarity.

    Args:
        ziu_list (List[torch.Tensor]): List of n embedding vectors.
        zju_list (List[torch.Tensor]): List of m embedding vectors.

    Returns:
        torch.Tensor: Pseudo-labels tensor of shape (n, m).
    """
    pseudo_labels = torch.zeros(len(ziu_list), len(zju_list))
    similarityall = torch.zeros(len(ziu_list), len(zju_list))
    for i, ziu in enumerate(ziu_list):
        for j, zju in enumerate(zju_list):
            # Compute the cosine similarity between ziu and zju
            similarity = cosine_similarity(ziu, zju)

            # Assign pseudo-label based on cosine similarity
            pseudo_labels[i, j] = 1 if similarity >= simil else 0
            similarityall[i, j] = similarity
    return pseudo_labels,similarityall
def pairwise_bce_loss(features1, features2, labels):
    criterion = nn.BCELoss()
    num_samples1 = features1.shape[0]
    num_samples2 = features2.shape[0]

    # Expand features and labels to match the pairwisel shape (num_samples1, num_samples2)
    expanded_features1 = features1.view(num_samples1, 1, -1).expand(num_samples1, num_samples2, -1)
    expanded_features2 = features2.view(1, num_samples2, -1).expand(num_samples1, num_samples2, -1)
    expanded_labels = labels.view(num_samples1, num_samples2)

    # Compute the pairwise BCE loss
    predictions = torch.sigmoid(torch.sum(expanded_features1 * expanded_features2, dim=-1))
    loss = criterion(predictions, expanded_labels.to(torch.float))
    return  loss




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

import torch.optim as optim


def mmd_loss(source_features, target_features, sigma=1.0):
    # Compute the mean feature representations for the source and target domains
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)

    # Calculate the pairwise squared Euclidean distance between the means
    distance = torch.sum((source_mean - target_mean) ** 2)

    # Apply the kernel function to obtain the MMD loss
    mmd_loss = torch.exp(-distance / (2 * sigma ** 2))

    return mmd_loss






def CACLoss(axh1, phiRall, train_Y):
    distances = pairwise_distances(axh1, phiRall)
    distances = L2Norm(distances)
    ns1 = axh1.size(0)
    c = phiRall.size(0)
    others = torch.zeros(ns1, c).cuda()
    true = torch.gather(distances, 1, train_Y.view(-1, 1)).view(-1).cuda()
    true1 = torch.zeros(ns1, c).float().cuda()
    for i in range(ns1):
        indx = train_Y[i]
        true1[i, indx] = true[i]
    others = distances - true1
    others1 = torch.mean(others, 1)
    anchor = torch.mean(true)

    tuplet = torch.exp(-others1.unsqueeze(1) + true)
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))
    total = anchor + tuplet
    return total, anchor, tuplet


def CACLossjian(pred, train_Y, ncs):
    ns1 = len(pred)

    c = ncs
    others = torch.zeros(ns1, c).cuda()
    true = torch.gather(pred, 1, train_Y.view(-1, 1)).view(-1).cuda()
    true1 = torch.zeros(ns1, c).float().cuda()
    for i in range(ns1):
        indx = train_Y[i]
        true1[i, indx] = true[i]
    others = pred - true1
    others1 = torch.mean(others, 1)
    anchor = torch.mean(true)
    tuplet = torch.exp(true - others1.unsqueeze(1))
    tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))
    total1 = anchor + tuplet
    total = torch.exp(- 1 * total1)
    return total, anchor, tuplet


def pairwise_kl_loss(neighbourhood_logits, logits,
                     knn_indices,
                     knn_similarities,
                     temperature,
                     epsilon=1e-16,
                     example_weights=None):
    """KL Divergence loss weighted by similarity within a neighbourhood.

    Args:
      logits: [n, d] array.
      neighbourhood_logits: [m, d] array.
      knn_indices: [n, k] array of nearest neighbours for each logit. Indexes
        into "neighbourhood_logits". Therefore, in each row, each of the k values
        are integers in the interval [0, m).
      knn_similarities: [n, k] array of each of the similarity scores
        corresponding to the knn_indices.
      temperature: A float.
      epsilon: Small constant for numerical stability.
      example_weights: The weight for each example in the batch. Can be used
        to account for TPU padding.

    Returns:
      The KL divergence of each logit to its neighbourhood, weighted by the
        similarity.
    """
    n, d = logits.shape
    k = knn_indices.shape[1]

    knn_logits = neighbourhood_logits[knn_indices.reshape(-1)].reshape([n, k, d])

    t_softmax_temp = torch.softmax(knn_logits / temperature, dim=1) + epsilon
    s_softmax_temp = torch.log_softmax(logits / temperature, dim=1)  # [n, d]

    # Normalize the sum of similarities by their sum, so that the new labels
    # will sum up to 1.
    normalized_sim = knn_similarities / torch.reshape(
        torch.sum(knn_similarities, axis=-1), (-1, 1))  # [n, k]
    afew = normalized_sim.unsqueeze(1)

    weighted_t_softmax = torch.squeeze(
        torch.matmul(afew, t_softmax_temp))  # [n, d]

    kldiv_loss_per_pair = weighted_t_softmax * (
            torch.log(weighted_t_softmax) - s_softmax_temp)  # [n, m]
    qfew = pow(temperature, 2)
    kldiv_loss_per_example = (qfew * torch.sum(kldiv_loss_per_pair, 1))  # [n, 1]
    kldiv_loss_per_example[kldiv_loss_per_example.isnan()] = 0
    if example_weights is not None:
        normalization = example_weights.sum()
    else:
        normalization = n
    qef = torch.sum(kldiv_loss_per_example) / (normalization + epsilon)
    return qef


def get_knn( queries, dataset, k):
    """Return k nearest neighbours from dataset given queries.

    Args:
      queries: An [q, d] array where q is the number of queries, and d the
        dimensionality of each query-vector.
      dataset: An [n, d] array where n is the number of examples.
      k: The number of nearest neighbours to retrieve.
      zero_negative_similarities: If true, negative similarities are set to 0.

    Returns:
      indices: A [q, k] dimensional array. For each query, the k indices of the
        nearest neighbours are returned.
      similarities: A [q, k] dimensional array with the similarities to each
        query. Similarities and corresponding indices are sorted, in descending
        order.
    """
    if k <= 0:
        k = dataset.shape[0]
    if k > dataset.shape[0]:
        k = dataset.shape[0]
    queries = F.normalize(queries, p=2, dim=-1)
    dataset = F.normalize(dataset, p=2, dim=-1)

    all_similarities = torch.mm(queries, dataset.transpose(0, 1))  # [q, n]
    similarities, indices = all_similarities.topk(k)

    return indices, similarities


def normalize_to_01(tensor):
    # Calculate the minimum and maximum values in the tensor
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)

    # Normalize the tensor to the range [0, 1]
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    return normalized_tensor
def ncr(logits,
        features,
        batch_logits,
        batch_features,
        number_neighbours,
        smoothing_gamma,
        temperature=1.0,
        example_weights=None):
    indices, similarities = get_knn(batch_features, features,
                                         number_neighbours + 1)
    # Remove the example itself from the list of nearest neighbours.
    indices = indices[:, 1:]
    similarities = similarities[:, 1:]

    similarities = torch.pow(similarities, smoothing_gamma)
    loss = pairwise_kl_loss(logits, batch_logits, indices, similarities,
                                 temperature, example_weights=example_weights)
    return loss

from torch.autograd import Variable
class subspacemlpclass:
    def __init__(self, b,train_seen, Rall, ppp, alpha2, ahpha3, _train_X, _train_Y, knnq, _input_dim, _cuda, _lr, _beta1, _nepoch,
                 _batch_size, pretrain_classifer=''):
        self.b = b
        self.train_seen = train_seen
        self.train_X = _train_X
        self.train_Y = _train_Y
        self.ahpha3 = ahpha3
        self.Rall = Rall
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = ppp  # 子空间维度
        self.knnq = knnq
        self.nclassc = Rall.size(0)
        self.input_dim = _input_dim
        self.cuda = _cuda
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
       # self.model.apply(util.weights_init)

        self.alpha2 = alpha2
        self.modellr = LINEAR_LOGSOFTMAXlr(self.nclass, self.nclassc)
       # self.modellr.apply(util.weights_init)

        self.criterion = nn.NLLLoss()
        self.loss_func = torch.nn.MSELoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.aaa = torch.nn.ModuleList([self.model, self.modellr])
        self.optimizer = optim.Adam(self.aaa.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.cuda()
            self.modellr.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if pretrain_classifer == '':
            self.fit()


    def fit(self):
        seenlabelone_hot = torch.zeros(self.train_Y.size(0), self.nclassc).cuda()
        seenlabelone_hot = seenlabelone_hot.scatter_(1, self.train_Y.unsqueeze(1), 1)
        iter_ = data_iterator(self.train_X, self.train_Y, self.batch_size, self.train_seen)
        ns = torch.sum(train_seen)
        Rs = seenRtqiu(self.train_X[0:ns].cpu(), self.train_Y[0:ns].cpu())
        Rsv = Variable(Rs.cuda())
        cs = Rs.size(0)
        ct1 = self.nclassc - cs
        momentum = 0.9
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):
                # x, y, seen = next(iter_)
                # x, y, seen = x.to(device), y.to(device), seen.bool().to(device)

                self.model.zero_grad()
                self.modellr.zero_grad()
                batch_input, batch_label, seen = next(iter_)

                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                seenv = seen.bool()
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                seenlabelone_hotlabelv = torch.zeros(labelv.size(0), self.nclassc).cuda()
                seenlabelone_hotlabelv = seenlabelone_hotlabelv.scatter_(1, labelv.unsqueeze(1), 1)
                # #seen labels
                # seenlabelone_hotlabelvs = torch.zeros(labelv[seenv].size(0), cs).cuda()
                # seenlabelone_hotlabelvs = seenlabelone_hotlabelvs.scatter_(1, labelv[seenv].unsqueeze(1), 1)
                # # unseenlabels
                # seenlabelone_hotlabelvs = torch.zeros(labelv[~seenv].size(0), ct1).cuda()
                # seenlabelone_hotlabelvs = seenlabelone_hotlabelvs.scatter_(1, labelv[~seenv].unsqueeze(1), 1)

                Rallv = Variable(self.Rall)

                # 子空间
                output = self.model(inputv)
                phiRall = self.model(Rallv)
                phiRsv = self.model(Rsv)

                #   cacLoss, anchorLoss, tupletLoss = self.CACLoss(output, phiRall, labelv)
                # phiRallRs = phiRall[labelv, :]
                # phiRsvYs =  torch.mm(seenlabelone_hotlabelvs, phiRsv)
                # aaa = output[seenv, :]
                #
                # bbb = phiRsvYs
                # lossz = self.loss_func(aaa, bbb)
                # distances = self.pairwise_distances(output, phiRall)
                # predprod = self.chuanboyici(distances)
                # predicted_labelprostdist, predicted_labelsdis = torch.max(predprod, 1)
                # lossz = torch.sum((predicted_labelsdis - labelv)^2)
                # 分类器
                trainXphi = self.model(self.train_X)
                outputlr = self.modellr(output)
                trainXy = self.modellr(trainXphi)
                loss = self.criterion(outputlr[seenv], labelv[seenv])
                if epoch  > 10:

                    logits = outputlr[~seenv]
                    # obtain prob, then update running avg
                    prob = F.softmax(logits, dim=1)
                    soft_labels = seenlabelone_hotlabelv[~seenv]
                    soft_labels = self.b * soft_labels + (1 - self.b) * prob.detach()

                    # obtain weights based largest and second largest prob
                    weights, _ = soft_labels.max(dim=1)
                    weights *= logits.shape[0] / weights.sum()

                    # use symmetric cross entropy loss, without reduction
                    loss_weighted = torch.sum(-F.log_softmax(logits, dim=1) * soft_labels, dim=1)


                    loss_weighted = (loss_weighted * weights).mean()
                else:
                    loss_weighted = 0




                # logits = outputlr
                # # obtain prob, then update running avg
                # prob = F.softmax(logits, dim=1)
                # soft_labels = seenlabelone_hotlabelv
                # soft_labels = momentum * soft_labels + (1 - momentum) * prob.detach()
                #
                # # obtain weights based largest and second largest prob
                # weights, _ = soft_labels.max(dim=1)
                # weights *= logits.shape[0] / weights.sum()
                #
                # # use symmetric cross entropy loss, without reduction
                # loss_weighted = - 1 * torch.sum(soft_labels * torch.log(prob), dim=-1) \
                #                 - 0.3 * torch.sum(prob * torch.log(soft_labels), dim=-1)
                #
                # # sample weighted mean
                # loss_weighted = (loss_weighted * weights).mean()

                # lossnrc1 = ncr(trainXy, self.train_X, outputlr, inputv, number_neighbours=self.knnq,
                #                smoothing_gamma=1, temperature=2.0, example_weights=None)
                # 对 子空间的约束
                lossnrc = ncr(trainXphi, self.train_X, output, inputv, number_neighbours=self.knnq,
                              smoothing_gamma=1, temperature=2.0, example_weights=None)
               # loss1 = loss + self.ahpha3 * (lossnrc + lossnrc1) + self.alpha2 * loss_weighted
                #loss1 = loss +  self.alpha2 * loss_weighted
                loss1 = loss +  self.alpha2 * loss_weighted + self.ahpha3 * lossnrc
                loss1.backward(retain_graph=True)
                self.optimizer.step()





                #loss22 = self.criterion(outputlr[~seenv], labelv[~seenv])
                #
                # pseudo_labelsst, similarityst = pairwise_pseudo_labels(outputlr[~seenv], outputlr[~seenv], 0.5)
                # pseudo_labelsstf, similaritystf = pairwise_pseudo_labels(inputv[~seenv], inputv[~seenv], 0.5)
                # losssimsim = self.loss_func(similarityst, similaritystf)
                # lossz1 = self.loss_func(outputlr, seenlabelone_hotlabelv)
                # 对 分类空间的约束


                # # 子空间近
                # lossnrc = self.ncr(seenlabelone_hot, trainXphi, seenlabelone_hotlabelv, output, number_neighbours=40,
                #                    smoothing_gamma=1, temperature=2.0, example_weights=None)

                # # cacLoss1, anchorLoss1, tupletLoss1 = self.CACLossjian(output, labelv, self.nclass)


                # lossmmd = mmd_loss(output[seenv], output[~seenv], sigma=1.0)
                #
                # pairwisel = pairwise_pseudo_labels(output[~seenv], output[~seenv], 0.5)
                # losspbc = pairwise_bce_loss(output[~seenv], output[~seenv], pairwisel.to(device))
                 # loss1 =  loss + self.alpha2 * losspbc - self.ahpha3 * lossmmd
                # unseenpl = outputlr[~seenv]
                # unseenpl =  unseenpl[:,cs:self.nclassc]
                # unseenpl = normalize_to_01(outputlr)
                # average_probabilities = torch.mean(unseenpl, dim=0)
                # entropy = -torch.sum(average_probabilities * torch.log(average_probabilities + 1e-8))
                # lossem = entropy / self.batch_size +self.alpha2 * lossnrc1




class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)

    def forward(self, x):
        x = self.fc(x)
        return x


#
class LINEAR_LOGSOFTMAXlr(nn.Module):
    def __init__(self, input_dim, nclassc):
        super(LINEAR_LOGSOFTMAXlr, self).__init__()
        self.fc = nn.Linear(input_dim, nclassc)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


def zikongjianfenlei(Rall,ppp, alpha2, ahpha3,train_Xa,train_Ya,Xtest,Xunlabeled,Xseen,new_classes2,trainseen,b):

    print('loss + lossz + self.alpha1 * lossnrc + self.alpha2 * lossnrc1',  alpha2, ahpha3)
    pretrain_clst = subspacemlpclass(b,trainseen.cuda(), Rall.cuda(), ppp, alpha2, ahpha3, train_Xa.cuda(),
                                     train_Ya.cuda(),
                                     100,
                                     train_Xa.size(1), True, 0.001, 0.5, 20, 100,
                                     '')

    phiXtest = pretrain_clst.model(Xtest.cuda())
    phitrain_Xa = pretrain_clst.model(train_Xa.cuda())
    phiXunlabeled = pretrain_clst.model(Xunlabeled.cuda())
    phiXseen = pretrain_clst.model(Xseen.cuda())
    phiRall = pretrain_clst.model(Rall.cuda())
    print(' 子空间，然后lr分类 ')
    ct = new_classes2.size(0)
    # all_acc1, old_acc1, new_acc1, all_acc, old_acc, new_acc = zuizhongfenleiphi(Rall, phitrain_Xa.cpu(),
    #                                                                             train_Ya,
    #                                                                             new_classes2,
    #                                                                             old_classes,
    #                                                                             phiXtest.cpu(),
    #                                                                             phiXunlabeled.cpu(),ct)

    predictst = pretrain_clst.modellr(phiXtest.cuda())
    predicted_labelprost, predicted_labels = torch.max(predictst, 1)
    predicted_labels = predicted_labels.cpu().numpy()
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Ytest.numpy()]))
    all_acc4, old_acc4, new_acc4 = log_accs_from_preds(y_true=Ytest.numpy(), y_pred=predicted_labels,
                                                       mask=mask,
                                                       eval_funcs=args.eval_funcs, save_name=None,
                                                       writer=None)
    print('子空间同时求的svm  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc4, old_acc4, new_acc4))
    all_acc1 = 0
    old_acc1 = 0
    new_acc1 = 0
    return all_acc4, old_acc4, new_acc4, all_acc1, old_acc1, new_acc1,phiXtest,phitrain_Xa, phiXunlabeled, phiXseen, phiRall

def calculate_accuracy(Yall1ssselect1, Yrealselect1):
    """
    Calculate accuracy of the predictions.

    Parameters:
        Yall1ssselect1 (torch.Tensor): Tensor containing predicted labels.
        Yrealselect1 (torch.Tensor): Tensor containing true labels.

    Returns:
        float: Accuracy of the predictions.
    """
    correct_predictions = torch.sum(Yall1ssselect1 == Yrealselect1)
    total_samples = Yrealselect1.size(0)
    accuracy = correct_predictions / total_samples
    return accuracy.item()


def run_clustering_algorithm(data, num_clusters):
    # Implement your clustering algorithm here (e.g., K-means)
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(data)
    return labels

def compute_similarity_matrix(labels_list):
    # Compute similarity matrix using pairwise distances or other similarity measures
    similarity_matrix = torch.zeros(len(labels_list), len(labels_list))
    for i, labels_i in enumerate(labels_list):
        for j, labels_j in enumerate(labels_list):
            similarity_matrix[i, j] = torch.tensor(np.mean(labels_i == labels_j))
    return similarity_matrix

def consensus_clustering(data, num_clusters, num_iterations):
    labels_list = []
    for _ in range(num_iterations):
        # Run clustering algorithm and store the labels in a list
        labels = run_clustering_algorithm(data, num_clusters)
        labels_list.append(labels)

    similarity_matrix = compute_similarity_matrix(labels_list)

    # Implement consensus function (e.g., average linkage or K-means on consensus matrix)
    consensus_labels = run_clustering_algorithm(similarity_matrix, num_clusters)

    return consensus_labels

if __name__ == "__main__":

    parser = ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action="store_true", default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', action="store_true", default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()

    print(args)
    device = torch.device('cuda:0')
    # 取出训练的类别 和测试的类别
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    args.interpolation = 3
    args.crop_pct = 0.875


 
    dill.load_session('faircraftea.pkl')



    cs = old_classes.size(0)
    ct = new_classes2.size(0)
    allclass = torch.cat((old_classes, new_classes2), 0)
    ntrain = Xseen.size(0)
    dim =  Xseen.size(1)
    cuestimate= ct
   
    ahpha1 = 0.1
    alpha = 0.1
    batch_size = 100
    train_seen = torch.ones_like(Yunlabeled)
    iter_ = data_iterator(Xunlabeled, Yunlabeled, batch_size, train_seen)

    seenRs = seenRtqiu(Xseen, Yseen)
    latent_size = cs + 1
    model = AE(dim, latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ntrain = Xunlabeled.size(0)
    epochs = 20
    Xseen = Xseen.to(device)
    Yseen = Yseen.to(device)
    lda = LinearDiscriminantAnalysis(n_components=400)

    for epoch in range(epochs):

        for iteration in range(0, ntrain, batch_size):
            x, y, unseen = next(iter_)

            x, y, unseen = x.to(device), y.to(device), unseen.bool().to(device)
            Xsxtb = torch.cat((Xseen, x), 0)

            U, s, VT = torch.svd(Xsxtb)

            Sigma = torch.diag(s)
            P = VT.t()
            Wlda = P
            eig_vec_picked = Wlda
            phix = torch.mm(x, eig_vec_picked)
            Rsen = torch.mm(seenRs.to(device), eig_vec_picked)
            probabilities = prototypical_probability(phix, Rsen, temperature=1.0)
            predicted_labelprost1, predicted_labels = torch.max(probabilities, 1)

            ewfe = torch.max(predicted_labelprost1)
            fewe = torch.min(predicted_labelprost1)
            fewfew1 = 0.1 * (ewfe - fewe)
            mu0 = fewe + 0.7 * fewfew1
            mu0 =  0.4
            print(mu0)
            ind = torch.ge(predicted_labelprost1, mu0)
            inefewd = np.nonzero(ind).long()
            inefewd = inefewd.cpu()
            cc = inefewd.squeeze()
            cc = cc.to(device)
            Xtsconfidence = torch.index_select(x, 0, cc)
            Ytsconfidence = torch.index_select(predicted_labels,  0, cc)
            Ytsconfidencereal = torch.index_select(y, 0, cc)
            acc = accpred(Ytsconfidence,Ytsconfidencereal)
         

            deta = 0.6
            Xsselecthighconfi, Ysselecthighconfi, Ysrealhighconfi, num_neighbors_with_same_labelpercentage1, Xuselecthighconfi, Yuselecthighconfi, Yurealhighconfi= find_k_nearest_neighbors_with_pseudo_labels(
                Xtsconfidence.cpu().numpy(), Ytsconfidence.cpu().numpy(), Ytsconfidencereal.cpu().numpy(),Xseen.cpu().numpy(), Yseen.cpu().numpy(), 10, deta)


            tem1 = torch.linspace(0, x.size(0) - 1, steps= x.size(0))
            tem1 = tem1.long()
            inefewdex = np.setdiff1d(tem1, inefewd)
            inefewdex = torch.from_numpy(inefewdex)
            inefewdex = inefewdex.long()
            cc1 = inefewdex.squeeze()
            cc1 = cc1.to(device)
            Xtuconfidence = torch.index_select(x, 0, cc1)
            Ytuconfidence = torch.index_select(predicted_labels, 0, cc1)
            Ytuconfidencereal = torch.index_select(y, 0, cc1)
            #acc1 = cluster_acc(Ytuconfidencereal.cpu().numpy(), Ytuconfidence.cpu().numpy())

            qfda = torch.ge(Ytuconfidencereal, cs)
            qfda = torch.sum(qfda)
            sacc11 = qfda / Ytuconfidencereal.size(0)
         
            Xsxtbs = torch.cat((Xseen, Xsselecthighconfi.to(device)), 0)
            Ysxtbs = torch.cat((Yseen, Ysselecthighconfi.to(device)), 0)

            Xtuconfidence = torch.cat((Xtuconfidence, Xuselecthighconfi.to(device)), 0)
            Ytuconfidence = torch.cat((Ytuconfidence, Yuselecthighconfi.to(device)), 0)
           
            pairwisel, similaritystf = pairwise_pseudo_labels(Xtuconfidence, Xtuconfidence, 0.5)
            losspbc = pairwise_bce_loss(Xtuconfidence, Xtuconfidence, pairwisel.to(device))

            Ytuconfidence = cs * torch.ones_like(Ytuconfidence)

            Xsu = torch.cat((Xsxtbs, Xtuconfidence), 0)
            Ysu = torch.cat((Ysxtbs,  Ytuconfidence), 0)

            Xsbsraec1, Xsxtbbatch1 = model(Xsu)
            lossseen = cross_entropy_loss(Xsxtbbatch1, Ysu.long())

            loss = lossseen + losspbc

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    Yseen = Yseen.cpu()
    feat1, Ysepre = model(Xseen.to(device))
    feat2, Ytuspre = model(Xunlabeled.to(device))
    predicted_labelprost1, predicted_labels = torch.max(Ytuspre.cpu(), 1)
    
    x_centers, Ysclu1, st_score = sklearn_kmeans(feat1.cpu(), cs)
    new_acctunseen1 = cluster_acc(Yseen.numpy(), Ysclu1.numpy())
    print(' xencoder xs clusteracc正确率', new_acctunseen1 * 100)
    x_centers, Ysclu1, st_score = sklearn_kmeans(feat2.cpu(), cs + cuestimate)
    new_acctunseen1 = cluster_acc(Yunlabeled.numpy(), Ysclu1.numpy())
    print('x xencoder clusteracc正确率', new_acctunseen1 * 100)

    Xtestmap, Ytestspre = model(Xtest.to(device))
    predicted_labelprost1, predicted_labels = torch.max(Ytestspre.cpu(), 1)

    temp2 = torch.nonzero(predicted_labels == cs)
    indx = temp2.squeeze().long()
    Xtestu = Xtest[indx,:]
    x_centers3z, Ysclu13z, st_score3z = sklearn_kmeans(Xtestu.cpu(),  cuestimate)
    predicted_labels = predicted_labels.long()
    Ysclu13z =  Ysclu13z.long()
    predicted_labels[indx] = Ysclu13z + cs
    mask = np.array([])
    mask = np.append(mask, np.array([True if x.item() in range(len(old_classes))
                                     else False for x in Ytest.numpy()]))
    all_acc1, old_acc1, new_acc1 = log_accs_from_preds(y_true=Ytest.numpy(), y_pred= predicted_labels.numpy(), mask=mask,
                                                       eval_funcs=args.eval_funcs, save_name=None,
                                                       writer=None)
    print('test Accuracies:  | All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc1, old_acc1, new_acc1))





    x_centers3, Ysclu13, st_score3 = sklearn_kmeans(Xtestmap.cpu(), cs + cuestimate)
    new_acctunseen1 = cluster_acc(Ytest.numpy(), Ysclu13.numpy())
    print('x xencoder  Xtest clusteracc正确率', new_acctunseen1 * 100)

    train_Xmap, Ytuspref = model(train_X.to(device))
    Rtssen = model(Rsu.to(device))
    print("Xseen  Xtu map ")
    all_acc1, old_acc1, new_acc1, all_acc, old_acc, new_acc = zuizhongsvm(Rtssen, train_Xmap, train_Y,
                                                                          old_classes, Xtestmap, feat2, alpha,  ahpha1,cuestimate)
   




    #
    #





