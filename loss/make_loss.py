# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss_MC import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss_MC import TripletLoss
from .center_loss_MC import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    # feat_dim = 2048
    feat_dim = cfg.MODEL.FEAT_DIM if 'FEAT_DIM' in cfg.MODEL else 2048
    if_with_center = cfg.MODEL.IF_WITH_CENTER == 'yes'
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            if if_with_center:
                if isinstance(feat, list):
                    CENTER_LOSS = [center_criterion(feats, target) for feats in feat[1:]]
                    CENTER_LOSS = sum(CENTER_LOSS) / len(CENTER_LOSS)
                    CENTER_LOSS = 0.5 * CENTER_LOSS + 0.5 * center_criterion(feat[0], target)
                else:
                    CENTER_LOSS = center_criterion(feat, target)
                return F.cross_entropy(score, target) + CENTER_LOSS
            else:
                return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    if if_with_center:
                        if isinstance(feat, list):
                            CENTER_LOSS = [center_criterion(feats, target) for feats in feat[1:]]
                            CENTER_LOSS = sum(CENTER_LOSS) / len(CENTER_LOSS)
                            CENTER_LOSS = 0.5 * CENTER_LOSS + 0.5 * center_criterion(feat[0], target)
                        else:
                            CENTER_LOSS = center_criterion(feat, target)

                    if if_with_center:
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                   cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                                   cfg.SOLVER.CENTER_LOSS_WEIGHT * CENTER_LOSS
                    else:
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    if if_with_center:
                        if isinstance(feat, list):
                            CENTER_LOSS = [center_criterion(feats, target) for feats in feat[1:]]
                            CENTER_LOSS = sum(CENTER_LOSS) / len(CENTER_LOSS)
                            CENTER_LOSS = 0.5 * CENTER_LOSS + 0.5 * center_criterion(feat[0], target)
                        else:
                            CENTER_LOSS = center_criterion(feat, target)

                    if if_with_center:
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                                   cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                                   cfg.SOLVER.CENTER_LOSS_WEIGHT * CENTER_LOSS
                    else:
                        return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                            cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


