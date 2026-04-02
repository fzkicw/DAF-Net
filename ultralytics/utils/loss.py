# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training.
       训练期间计算训练损失的准则类 """

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.
        初始化 BboxLoss模块，设置正则化最大值和DFL设置
        参数   reg_max(int)回归的最大值,如果大于1则使用DFL"""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None  # 如果 reg_max > 1，则初始化 DFLoss

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                difficulty_weights=None):
        """IoU loss."""
        """
                计算 IoU 损失和可选的 DFL 损失。
                参数:
                    pred_dist (Tensor): 预测的边界框分布。
                    pred_bboxes (Tensor): 预测的边界框。
                    anchor_points (Tensor): 锚点。
                    target_bboxes (Tensor): 目标（真实）边界框。
                    target_scores (Tensor): 每个锚点的目标（真实）分数。
                    target_scores_sum (Tensor): 目标分数的总和。
                    fg_mask (Tensor): 指示正样本的前景掩码。
                    difficulty_weights (Tensor, optional): 难度权重，形状为 [num_fg]。
                返回:
                    Tuple[Tensor, Tensor]: IoU 损失和 DFL 损失。
                """
        # 计算前景样本（正样本）的权重
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # 如果提供了难度权重，应用到权重上
        if difficulty_weights is not None:
            difficulty_weights = difficulty_weights.unsqueeze(-1)
            weight = weight * difficulty_weights

        # 计算IoU损失
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 似乎是只计算了确认正确下的损失函数， fg_mask

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        # iou的计算最终与 正样本部分的 类别权重相乘， 类似于v5里面的 objective*conf吧，最终在求和并且标准化

        # DFL loss  如果使用 DFL，则计算 DFL 损失
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                difficulty_weights=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # 如果提供了难度权重，应用到权重上
        if difficulty_weights is not None:
            difficulty_weights = difficulty_weights.unsqueeze(-1)
            weight = weight * difficulty_weights

        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    # YOLOv8目标检测部分损失函数

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters cfg/default.yaml 下的所有超参获取

        m = model.model[-1]  # Detect() module 获取模型的检测头，获取其中参数
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # 二分类交叉熵损失函数
        self.hyp = h
        self.stride = m.stride  # model strides   对应 640即== 20（32）， 40（16）， 80（8）
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4  # reg_max为每个位置信息的预测数量 4代表四个位置 即xywh
        self.reg_max = m.reg_max  # DFL通道数量，每个预测框的回归输出通道数
        self.device = device

        self.use_dfl = m.reg_max > 1
        # 任务对齐分配器
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)  # 分配GT信息到锚点矩阵
        self.bbox_loss = BboxLoss(m.reg_max).to(device)  # 边界框损失计算函数
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)  # 投影张量，从1取一直到reg_max

        # 难度权重配置
        self.difficulty_weights = torch.tensor([0.6, 1.0, 1.4], device=self.device)  # 简单:0.8, 中等:1.0, 困难:1.2

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:  # 没有目标的情况
            # 根据是否有难度信息返回不同维度的张量
            if ne == 7:  # batch_idx + cls + bbox(4) + difficulty
                out = torch.zeros(batch_size, 0, 6, device=self.device)  # cls(1) + bbox(4) + difficulty(1)
            else:  # batch_idx + cls + bbox(4) 或 batch_idx + cls + bbox(4) + ??
                out = torch.zeros(batch_size, 0, 5, device=self.device)  # cls(1) + bbox(4)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)  # 获取每个图像的目标计数
            counts = counts.to(dtype=torch.int32)

            # 根据输入维度确定输出维度
            if ne == 7:  # 包含难度信息
                out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            else:  # 不包含难度信息
                out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

            for j in range(batch_size):  # 遍历每个批次
                matches = i == j  # 匹配当前批次的目标
                n = matches.sum()  # 目标数量
                if n:
                    out[j, :n] = targets[matches, 1:]  # 填充目标

            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))  # 转换坐标格式
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution.
           从锚点和分布预测中解码出越策的目标边界框坐标
           参数:
            anchor_points (torch.Tensor): 锚点坐标，形状为 [num_anchors, 2]。 640一般是8400，即（8400,2）
            pred_dist (torch.Tensor): 预测的边界框分布，形状为 [batch_size, num_anchors, num_channels]。 （4,8400,64）
        返回:
            torch.Tensor: 解码后的边界框坐标，形状为 [batch_size, num_anchors, 4]。   （4,8400,4）
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels  输入的pred_dist == 4.8400,64
            # 将预测分布变形为[batch_size, num_anchors ,4 ,num_channels // 4]，并在通道维度上应用 softmax
            c1 = pred_dist.view(b, a, 4, c // 4)  # (4,8400,64) to (4,8400,4,16)  该步骤解算64→16
            c2 = c1.softmax(3)  # (4,8400,4,16)  将数值转化为当前位置的概率分布，仅涉及数学运算，没有需要学习的参数，数据维度信息也没有变化
            c3 = self.proj.type(pred_dist.dtype)
            pred_dist = c2.matmul(c3)  # matmul 一种矩阵乘法 ???  4,8400,4,16 * 16（一维） ==  4，8400,4  矩阵点积，不足维度自动广播
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl 初始化损失张量全是0
        feats = preds[1] if isinstance(preds, tuple) else preds

        # 解析预测结果
        c1 = [xi.view(feats[0].shape[0], self.no, -1) for xi in feats]
        c2 = torch.cat(c1, 2)
        c3 = c2.split((self.reg_max * 4, self.nc), 1)

        pred_distri, pred_scores = c3
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # contiguous 判断张量是否连续    网络预测类别 6
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # 网络预测框，64reg

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)  # 检测头也调用过

        # Targets - 构建包含难度信息的目标张量
        if "difficulty" in batch:
            # 如果有难度信息，构建7列目标：batch_idx, cls, bboxes, difficulty
            targets = torch.cat((batch["batch_idx"].view(-1, 1),
                                 batch["cls"].view(-1, 1),
                                 batch["bboxes"],
                                 batch["difficulty"].to(batch['cls'].device).view(-1, 1)), 1)
        else:
            # 如果没有难度信息，构建6列目标：batch_idx, cls, bboxes
            targets = torch.cat((batch["batch_idx"].view(-1, 1),
                                 batch["cls"].view(-1, 1),
                                 batch["bboxes"]), 1)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        # 分割目标：cls, xyxy, difficulty（如果有）
        if targets.shape[-1] == 6:  # 包含难度信息
            gt_labels, gt_bboxes, gt_difficulties = targets.split((1, 4, 1), 2)
        else:  # 不包含难度信息
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            gt_difficulties = None

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points,
                                       pred_distri)  # xyxy, (batch, h*w合并, 4：xywh) 输入（8400，2）和（4,8400,64），输出（4,8400,4）

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(  # 分配what
            pred_scores.detach().sigmoid(),  # ①pred_scores 每个锚点位置的类别得分（4,8400，cls）
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            # ②pred_bboxes 解码后的边界框坐标 （4,8400,4）.detach 从计算图中分离出预测的边界框，
            # 避免反向传播更新他们    stride_tensor 乘以步幅张量回复到原图尺寸   .type(gt_bboxes.dtype)将预测张量写为与gt相同格式
            anchor_points * stride_tensor,  # ③anchor_points,锚点位置（8400,2）与步幅相乘得原图尺寸（8400， 1）【8， 16， 32】
            gt_labels,  # ④真实类别的目标标签   （4,1,1）
            gt_bboxes,  # ⑤真实类别的边界框坐标  （4,1,4） 格式 xyxy
            mask_gt,  # ⑥掩码张量，识别哪些目标是有效的  （4,1,1）
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 获取难度权重（与YOLOv26保持一致）
        difficulty_weight = None
        if fg_mask.sum() > 0 and gt_difficulties is not None:
            try:
                # 从gt_difficulties中获取每个正样本的难度
                batch_size = fg_mask.shape[0]
                all_selected_difficulties = []

                for i in range(batch_size):
                    mask_i = fg_mask[i]
                    if mask_i.any():
                        # 获取该batch中的目标难度
                        difficulties_i = gt_difficulties[i].squeeze(-1)  # 形状: [max_objects]
                        # 使用target_gt_idx获取对应的难度
                        selected = difficulties_i[target_gt_idx[i][mask_i]]  # 形状: [num_fg_in_i]
                        all_selected_difficulties.append(selected)

                if all_selected_difficulties:
                    # 合并所有batch的正样本难度
                    all_selected_difficulties = torch.cat(all_selected_difficulties, dim=0)
                    # 将难度索引转换为权重 (0->0.8, 1->1.0, 2->1.2)
                    difficulty_mask = all_selected_difficulties.long().squeeze(-1)
                    difficulty_weight = self.difficulty_weights[difficulty_mask]  # 形状: [num_fg]
            except Exception as e:
                print(f"Warning: Failed to get difficulty weights: {e}")

        # Cls loss with difficulty weighting（与YOLOv26保持一致）
        if fg_mask.sum() > 0 and difficulty_weight is not None:
            # 正样本分类损失
            pred_scores_fg = pred_scores[fg_mask]  # 形状: [num_fg, num_classes]
            target_scores_fg = target_scores[fg_mask].to(dtype)  # 形状: [num_fg, num_classes]

            # 计算每个正样本的分类损失（对类别维度求和）
            cls_loss_fg = self.bce(pred_scores_fg, target_scores_fg).sum(dim=1)  # 形状: [num_fg]

            # 应用难度加权
            cls_loss_fg_weighted = cls_loss_fg * difficulty_weight

            # 负样本分类损失（不加权）
            bg_mask = ~fg_mask
            if bg_mask.any():
                pred_scores_bg = pred_scores[bg_mask]
                target_scores_bg = target_scores[bg_mask].to(dtype)
                # 负样本损失按元素计算，然后求和
                cls_loss_bg = self.bce(pred_scores_bg, target_scores_bg).sum()

                # 总分类损失 = 加权正样本损失 + 负样本损失
                loss[1] = (cls_loss_fg_weighted.sum() + cls_loss_bg) / target_scores_sum
            else:
                loss[1] = cls_loss_fg_weighted.sum() / target_scores_sum
        else:
            # 原始计算方法（没有难度加权）
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss with difficulty weighting
        if fg_mask.sum():  # 对fg_mask进行.sum()求和上面讲了这个fg_mask里面是布尔值,但是也是可以求和的True为1, False为0, 如果有一个是正样本是 1 则会进行边界框损失计算, 这是python基础问题怕大家有疑问解释一下！
            target_bboxes /= stride_tensor
            # 将目标边界框的坐标除以步幅，恢复到特征图尺度（这里可以看到stride这个参数其实很重要的需要辅助我们真实图和特征图之间相互转化）
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                difficulty_weight
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            # 构建包含难度信息的目标张量
            if "difficulty" in batch:
                targets = torch.cat(
                    (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"], batch["difficulty"].view(-1, 1)), 1)
            else:
                targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)

            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

            # 分割目标：cls, xyxy, difficulty（如果有）
            if targets.shape[-1] == 6:  # 包含难度信息
                gt_labels, gt_bboxes, gt_difficulties = targets.split((1, 4, 1), 2)
            else:  # 不包含难度信息
                gt_labels, gt_bboxes = targets.split((1, 4), 2)
                gt_difficulties = None

            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 获取难度权重
        difficulty_weight = None
        if fg_mask.sum() > 0 and gt_difficulties is not None:
            try:
                batch_size = fg_mask.shape[0]
                all_selected_difficulties = []

                for i in range(batch_size):
                    mask_i = fg_mask[i]
                    if mask_i.any():
                        difficulties_i = gt_difficulties[i].squeeze(-1)
                        selected = difficulties_i[target_gt_idx[i][mask_i]]
                        all_selected_difficulties.append(selected)

                if all_selected_difficulties:
                    all_selected_difficulties = torch.cat(all_selected_difficulties, dim=0)
                    difficulty_mask = all_selected_difficulties.long().squeeze(-1)
                    difficulty_weight = self.difficulty_weights[difficulty_mask]
            except Exception as e:
                print(f"Warning: Failed to get difficulty weights: {e}")

        # Cls loss with difficulty weighting
        if fg_mask.sum() > 0 and difficulty_weight is not None:
            # 正样本分类损失
            pred_scores_fg = pred_scores[fg_mask]
            target_scores_fg = target_scores[fg_mask].to(dtype)

            # 计算每个正样本的分类损失（对类别维度求和）
            cls_loss_fg = self.bce(pred_scores_fg, target_scores_fg).sum(dim=1)

            # 应用难度加权
            cls_loss_fg_weighted = cls_loss_fg * difficulty_weight

            # 负样本分类损失（不加权）
            bg_mask = ~fg_mask
            if bg_mask.any():
                pred_scores_bg = pred_scores[bg_mask]
                target_scores_bg = target_scores[bg_mask].to(dtype)
                cls_loss_bg = self.bce(pred_scores_bg, target_scores_bg).sum()
                loss[2] = (cls_loss_fg_weighted.sum() + cls_loss_bg) / target_scores_sum
            else:
                loss[2] = cls_loss_fg_weighted.sum() / target_scores_sum
        else:
            # 原始计算方法（没有难度加权）
            loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # Bbox loss with difficulty weighting
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                difficulty_weight,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
            gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
            self,
            fg_mask: torch.Tensor,
            masks: torch.Tensor,
            target_gt_idx: torch.Tensor,
            target_bboxes: torch.Tensor,
            batch_idx: torch.Tensor,
            proto: torch.Tensor,
            pred_masks: torch.Tensor,
            imgsz: torch.Tensor,
            overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)

        # 构建包含难度信息的目标张量
        if "difficulty" in batch:
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"], batch["difficulty"].view(-1, 1)),
                                1)
        else:
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)

        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        # 分割目标：cls, xyxy, difficulty（如果有）
        if targets.shape[-1] == 6:  # 包含难度信息
            gt_labels, gt_bboxes, gt_difficulties = targets.split((1, 4, 1), 2)
        else:  # 不包含难度信息
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            gt_difficulties = None

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 获取难度权重
        difficulty_weight = None
        if fg_mask.sum() > 0 and gt_difficulties is not None:
            try:
                batch_size = fg_mask.shape[0]
                all_selected_difficulties = []

                for i in range(batch_size):
                    mask_i = fg_mask[i]
                    if mask_i.any():
                        difficulties_i = gt_difficulties[i].squeeze(-1)
                        selected = difficulties_i[target_gt_idx[i][mask_i]]
                        all_selected_difficulties.append(selected)

                if all_selected_difficulties:
                    all_selected_difficulties = torch.cat(all_selected_difficulties, dim=0)
                    difficulty_mask = all_selected_difficulties.long().squeeze(-1)
                    difficulty_weight = self.difficulty_weights[difficulty_mask]
            except Exception as e:
                print(f"Warning: Failed to get difficulty weights: {e}")

        # Cls loss with difficulty weighting
        if fg_mask.sum() > 0 and difficulty_weight is not None:
            # 正样本分类损失
            pred_scores_fg = pred_scores[fg_mask]
            target_scores_fg = target_scores[fg_mask].to(dtype)

            # 计算每个正样本的分类损失（对类别维度求和）
            cls_loss_fg = self.bce(pred_scores_fg, target_scores_fg).sum(dim=1)

            # 应用难度加权
            cls_loss_fg_weighted = cls_loss_fg * difficulty_weight

            # 负样本分类损失（不加权）
            bg_mask = ~fg_mask
            if bg_mask.any():
                pred_scores_bg = pred_scores[bg_mask]
                target_scores_bg = target_scores[bg_mask].to(dtype)
                cls_loss_bg = self.bce(pred_scores_bg, target_scores_bg).sum()
                loss[3] = (cls_loss_fg_weighted.sum() + cls_loss_bg) / target_scores_sum
            else:
                loss[3] = cls_loss_fg_weighted.sum() / target_scores_sum
        else:
            # 原始计算方法（没有难度加权）
            loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                difficulty_weight
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
            self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            # 根据是否有难度信息返回不同维度的张量
            if targets.shape[1] == 7:  # batch_idx + cls + bbox(4) + angle(1) + difficulty(1)
                out = torch.zeros(batch_size, 0, 7, device=self.device)  # cls(1) + bbox(4) + angle(1) + difficulty(1)
            else:  # batch_idx + cls + bbox(4) + angle(1)
                out = torch.zeros(batch_size, 0, 6, device=self.device)  # cls(1) + bbox(4) + angle(1)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)

            # 根据输入维度确定输出维度
            if targets.shape[1] == 7:  # 包含难度信息
                out = torch.zeros(batch_size, counts.max(), 7, device=self.device)
            else:  # 不包含难度信息
                out = torch.zeros(batch_size, counts.max(), 6, device=self.device)

            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            # 构建包含难度信息的目标张量
            if "difficulty" in batch:
                targets = torch.cat(
                    (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5), batch["difficulty"].view(-1, 1)),
                    1)
            else:
                targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)

            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

            # 分割目标：cls, xywhr, difficulty（如果有）
            if targets.shape[-1] == 7:  # 包含难度信息
                gt_labels, gt_bboxes, gt_difficulties = targets.split((1, 5, 1), 2)
            else:  # 不包含难度信息
                gt_labels, gt_bboxes = targets.split((1, 5), 2)
                gt_difficulties = None

            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # 获取难度权重
        difficulty_weight = None
        if fg_mask.sum() > 0 and gt_difficulties is not None:
            try:
                batch_size = fg_mask.shape[0]
                all_selected_difficulties = []

                for i in range(batch_size):
                    mask_i = fg_mask[i]
                    if mask_i.any():
                        difficulties_i = gt_difficulties[i].squeeze(-1)
                        selected = difficulties_i[target_gt_idx[i][mask_i]]
                        all_selected_difficulties.append(selected)

                if all_selected_difficulties:
                    all_selected_difficulties = torch.cat(all_selected_difficulties, dim=0)
                    difficulty_mask = all_selected_difficulties.long().squeeze(-1)
                    difficulty_weight = self.difficulty_weights[difficulty_mask]
            except Exception as e:
                print(f"Warning: Failed to get difficulty weights: {e}")

        # Cls loss with difficulty weighting
        if fg_mask.sum() > 0 and difficulty_weight is not None:
            # 正样本分类损失
            pred_scores_fg = pred_scores[fg_mask]
            target_scores_fg = target_scores[fg_mask].to(dtype)

            # 计算每个正样本的分类损失（对类别维度求和）
            cls_loss_fg = self.bce(pred_scores_fg, target_scores_fg).sum(dim=1)

            # 应用难度加权
            cls_loss_fg_weighted = cls_loss_fg * difficulty_weight

            # 负样本分类损失（不加权）
            bg_mask = ~fg_mask
            if bg_mask.any():
                pred_scores_bg = pred_scores[bg_mask]
                target_scores_bg = target_scores[bg_mask].to(dtype)
                cls_loss_bg = self.bce(pred_scores_bg, target_scores_bg).sum()
                loss[1] = (cls_loss_fg_weighted.sum() + cls_loss_bg) / target_scores_sum
            else:
                loss[1] = cls_loss_fg_weighted.sum() / target_scores_sum
        else:
            # 原始计算方法（没有难度加权）
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Bbox loss with difficulty weighting
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                difficulty_weight
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]