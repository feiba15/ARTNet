from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import math

import util.dist as dist
from util import box_ops
from util.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x


class TubeDETR(nn.Module):
    """This is the TubeDETR module that performs spatio-temporal video grounding"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        video_max_len=200,
        stride=5,
        guided_attn=False,
        fast=False,
        fast_mode="",
        sted=True,
        attribute=True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        #用于动作分类
        # self.query_embed = nn.Embedding(num_queries+1, hidden_dim)
        self.action_query_embed = nn.Embedding(2,hidden_dim)
        self.action_query_mask = torch.zeros(1, 1).bool()

        self.noun_query_embed = nn.Embedding(1,hidden_dim)
        self.future_query_embed = nn.Embedding(1,hidden_dim)

        noun_max_number = 300
        self.noun_prototype_embed = nn.Embedding(hidden_dim, noun_max_number)
        self.video_noun_list = ['None']


        action_max_number = 300
        self.action_prototype_embed = nn.Embedding(hidden_dim, action_max_number)
        self.video_action_list = ['None']
        self.middle_action_map = {}

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.video_max_len = video_max_len
        self.stride = stride
        self.guided_attn = guided_attn
        self.fast = fast
        self.fast_mode = fast_mode
        self.sted = sted
        self.attribute = attribute

        if sted:
            self.sted_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.5)

        # input_dim, hidden_dim, output_dim, num_layers
        if attribute:
            self.attribute_embed = MLP(hidden_dim, hidden_dim, 89, 2, dropout=0.5)
            self.each_attribute_number = [2, 4, 3, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            self.sentiment_embed = MLP(hidden_dim, hidden_dim, 7, 2, dropout=0.5)
            self.scene_embed = MLP(hidden_dim, hidden_dim, 21, 2, dropout=0.5)
    def forward(
        self,
        samples: NestedTensor,
        durations,
        captions,
        encode_and_save=True,
        memory_cache=None,
        samples_fast=None,
    ):        
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            b = len(durations)
            t = max(durations)
            features, pos = self.backbone(
                samples
            )  # each frame from each video is forwarded through the backbone
            src, mask = features[
                -1
            ].decompose()  # src (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32)); mask (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))
            if self.fast:
                with torch.no_grad():  # fast branch does not backpropagate to the visual backbone
                    features_fast, pos_fast = self.backbone(samples_fast)
                src_fast, mask_fast = features_fast[-1].decompose()
                src_fast = self.input_proj(src_fast)

            # temporal padding pre-encoder
            src = self.input_proj(src)
            _, f, h, w = src.shape
            f2 = pos[-1].size(1)
            device = src.device
            tpad_mask_t = None
            fast_src = None
            if not self.stride:
                tpad_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask = torch.ones(b, t, h, w).bool().to(device)
                pos_embed = torch.zeros(b, t, f2, h, w).to(device)
                cur_dur = 0
                for i_dur, dur in enumerate(durations):
                    tpad_src[i_dur, :dur] = src[cur_dur : cur_dur + dur]
                    tpad_mask[i_dur, :dur] = mask[cur_dur : cur_dur + dur]
                    pos_embed[i_dur, :dur] = pos[-1][cur_dur : cur_dur + dur]
                    cur_dur += dur
                tpad_src = tpad_src.view(b * t, f, h, w)
                tpad_mask = tpad_mask.view(b * t, h, w)
                tpad_mask[:, 0, 0] = False  # avoid empty masks
                pos_embed = pos_embed.view(b * t, f2, h, w)
            else:  # temporal sampling
                n_clips = math.ceil(t / self.stride)
                tpad_src = src
                tpad_mask = mask
                pos_embed = pos[-1]
                if self.fast:
                    fast_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask_t = (
                    torch.ones(b, t, h, w).bool().to(device)
                )  # temporally padded mask for all frames, will be used for the decoding
                cum_dur = 0  # updated for every video
                cur_dur = 0
                cur_clip = 0
                for i_dur, dur in enumerate(durations):
                    if self.fast:
                        fast_src[i_dur, :dur] = src_fast[cum_dur : cum_dur + dur]
                        tpad_mask_t[i_dur, :dur] = mask_fast[cum_dur : cum_dur + dur]
                    else:
                        for i_clip in range(math.ceil(dur / self.stride)):
                            clip_dur = min(self.stride, dur - i_clip * self.stride)
                            tpad_mask_t[
                                i_dur, cur_dur - cum_dur : cur_dur - cum_dur + clip_dur
                            ] = mask[cur_clip : cur_clip + 1].repeat(clip_dur, 1, 1)
                            cur_dur += clip_dur
                            cur_clip += 1
                    cum_dur += dur
                tpad_src = tpad_src.view(b * n_clips, f, h, w)
                tpad_mask = tpad_mask.view(b * n_clips, h, w)
                pos_embed = pos_embed.view(b * n_clips, f, h, w)
                tpad_mask_t = tpad_mask_t.view(b * t, h, w)
                if self.fast:
                    fast_src = fast_src.view(b * t, f, h, w)
                tpad_mask[:, 0, 0] = False  # avoid empty masks
                tpad_mask_t[:, 0, 0] = False  # avoid empty masks
            query_embed = self.query_embed.weight

            # video-text encoder
            memory_cache = self.transformer(
                tpad_src,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                tpad_mask,  # (n_clips)x(math.ceil(H/32))x(math.ceil(W/32))
                query_embed,  # num_queriesxF
                pos_embed,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                captions,  # list of length batch_size
                encode_and_save=True,
                durations=durations,  # list of length batch_size
                tpad_mask_t=tpad_mask_t,  # (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))
                fast_src=fast_src,  # (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32))
            )

            return memory_cache

        else:
            assert memory_cache is not None
            hs = self.transformer(
                img_memory=memory_cache[
                    "img_memory"
                ],  # (math.ceil(H/32)*math.ceil(W/32) + n_tokens)x(BT)xF
                mask=memory_cache[
                    "mask"
                ],  # (BT)x(math.ceil(H/32)*math.ceil(W/32) + n_tokens)
                pos_embed=memory_cache["pos_embed"],  # n_tokensx(BT)xF
                query_embed=memory_cache["query_embed"],  # (num_queries)x(BT)xF 128*1*256
                query_mask=memory_cache["query_mask"],  # Bx(Txnum_queries) 1*128
                encode_and_save=False,
                text_memory=memory_cache["text_memory"],
                text_mask=memory_cache["text_attention_mask"],
                action_query=self.action_query_embed.weight.unsqueeze(1),
                noun_query=self.noun_query_embed.weight.unsqueeze(1),
                future_query=self.future_query_embed.weight.unsqueeze(1),
            )

            action_personAttr_hs = hs[-1].squeeze(1)
            action_hs = action_personAttr_hs[:1]
            personAttr_hs = action_personAttr_hs[1:]
            

            noun_hs = hs[-3].squeeze(1)
            future_hs = hs[-2].squeeze(1)

            hs = (hs[0], hs[1], hs[2])
            action_weight = torch.mm(action_hs,
                self.action_prototype_embed.weight[:, :len(self.video_action_list)])
            # action_prediction = F.softmax(action_weight, dim=-1)
            action_prediction = action_weight


            
            noun_prediction = torch.mm(noun_hs,
                self.noun_prototype_embed.weight[:, :len(self.video_noun_list)])

            # self.sentiment_embed = MLP(hidden_dim, hidden_dim, 7, 2, dropout=0.5)
            # self.scene_embed = MLP(hidden_dim, hidden_dim, 21, 2, dropout=0.5)
            personAttr_perdiction_list = []
            if self.attribute:
                personAttr_perdiction = self.attribute_embed(personAttr_hs)

                start_index = 0
                for personAttr_item in self.each_attribute_number:
                    end_index = start_index+personAttr_item
                    # prediction_item = F.softmax(personAttr_perdiction
                    #                 [start_index:end_index], dim=-1)
                    prediction_item = personAttr_perdiction[:, start_index:end_index]
                    personAttr_perdiction_list.append(prediction_item)
                    start_index = end_index
            if self.guided_attn:
                hs, weights, cross_weights = hs
            out = {}

            sentiment_prediction = None
            scene_prediction = None
            if self.attribute:
                sentiment_prediction = self.sentiment_embed(hs[-1])
                # sentiment_prediction = F.softmax(sentiment_prediction, dim=-1)
                scene_prediction = self.scene_embed(hs[-1])

            if self.sted:
                outputs_sted = self.sted_embed(hs)

            hs = hs.flatten(1, 2)  # n_layersxbxtxf -> n_layersx(b*t)xf

            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update({"pred_boxes": outputs_coord[-1]})
            if self.sted:
                out.update({"pred_sted": outputs_sted[-1]})
            if self.guided_attn:
                out["weights"] = weights[-1]
                out["ca_weights"] = cross_weights[-1]
            if self.attribute:
                # False
                out['pred_personAttr'] = personAttr_perdiction_list
                out['pred_sentiment'] = sentiment_prediction
                out['pred_scene'] = scene_prediction
            out["pred_action"] = action_prediction
            out["pred_noun"] = noun_prediction
            out["future_hs"] = future_hs

            # auxiliary outputs
            if self.aux_loss:
                out["aux_outputs"] = [
                    {
                        "pred_boxes": b,
                    }
                    for b in outputs_coord[:-1]
                ]
                for i_aux in range(len(out["aux_outputs"])):
                    if self.sted:
                        out["aux_outputs"][i_aux]["pred_sted"] = outputs_sted[i_aux]
                    if self.guided_attn:
                        out["aux_outputs"][i_aux]["weights"] = weights[i_aux]
                        out["aux_outputs"][i_aux]["ca_weights"] = cross_weights[i_aux]

            return out
class SetCriterion(nn.Module):
    """This class computes the loss for TubeDETR."""

    def __init__(self, losses, sigma=1):
        """Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            sigma: standard deviation for the Gaussian targets in the start and end Kullback Leibler divergence loss
        """
        super().__init__()
        self.losses = losses
        self.sigma = sigma

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """

        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / max(num_boxes, 1)

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / max(num_boxes, 1)
        return losses
    def loss_sted(self, outputs, num_boxes, inter_idx, positive_map, time_mask=None):
        """Compute the losses related to the start & end prediction, a KL divergence loss
        targets dicts must contain the key "pred_sted" containing a tensor of logits of dim [T, 2]
        """
        assert "pred_sted" in outputs
        sted = outputs["pred_sted"]
        losses = {}

        # print("000000000000000000")
        # print(sted.shape)
        # print(inter_idx)
        # exit(0)

        target_start = torch.tensor([x[0] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        target_end = torch.tensor([x[1] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        sted = sted.masked_fill(
            ~time_mask[:, :, None], -1e32
        )  # put very low probability on the padded positions before softmax
        eps = 1e-6  # avoid log(0) and division by 0

        sigma = self.sigma
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target

        start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
        pred_start_prob = (sted[:, :, 0]).softmax(1)
        loss_start = (
            pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
        )  # KL div loss
        loss_start = loss_start * time_mask  # not count padded values in the loss
        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
        pred_end_prob = (sted[:, :, 1]).softmax(1)
        loss_end = (
            pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
        )  # KL div loss
        loss_end = loss_end * time_mask  # do not count padded values in the loss

        loss_sted = loss_start + loss_end
        losses["loss_sted"] = loss_sted.mean()

        return losses

    def loss_guided_attn(
        self, outputs, num_boxes, inter_idx, positive_map, time_mask=None
    ):
        """Compute guided attention loss
        targets dicts must contain the key "weights" containing a tensor of attention matrices of dim [B, T, T]
        """
        weights = outputs["weights"]  # BxTxT

        positive_map = positive_map + (
            ~time_mask
        )  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch

        losses = {"loss_guided_attn": loss}
        return losses
    def loss_actions(self, outputs, action_target):
        loss = F.cross_entropy(outputs['pred_action'], action_target.long())
        losses = {"action":loss}

        # print(action_target)
        # print(loss)
        # exit(0)
        return losses

    def loss_nouns(self, outputs, noun_target):
        loss = F.cross_entropy(outputs['pred_noun'], noun_target.long())
        losses = {"noun":loss}
        return losses

    def loss_future(self, output_future, future_target):
        # batch=1,所以unsqueeze
        loss = F.cross_entropy(output_future.unsqueeze(0), future_target.long())
        losses = {"future":loss}
        return losses

    def loss_attribute(self, outputs, attribute_target):
        personAttr_perdiction_list = outputs['pred_personAttr']
        personAttr_target_list = attribute_target['personAttr']

        # print("11111111111111111111111111111111111111")
        # print(len(personAttr_perdiction_list))
        # print(personAttr_perdiction_list[0].size())

        loss_personAttr = 0
        for index in range(len(personAttr_perdiction_list)):
            personAttr_perdiction = personAttr_perdiction_list[index]
            personAttr_target = [personAttr_target_list[0][index]]
            personAttr_target_tensor = torch.tensor(personAttr_target).to(personAttr_perdiction.device)
            loss_personAttr+=F.cross_entropy(personAttr_perdiction, personAttr_target_tensor.long())

        sentiment_prediction_list = outputs['pred_sentiment']
        sentiment_target_list = attribute_target['sentiment']
        loss_sentiment = 0
        for index in range(len(sentiment_target_list[0])):
            sentiment_prediction = sentiment_prediction_list[:, index, :]
            sentiment_target = [sentiment_target_list[0][index]]
            sentiment_target_tensor = torch.tensor(sentiment_target).to(sentiment_prediction.device)
            loss_sentiment+=F.cross_entropy(sentiment_prediction, sentiment_target_tensor.long())
        loss_sentiment = loss_sentiment/len(sentiment_target_list[0])

        scene_prediction_list = outputs['pred_scene']
        scene_target_list = attribute_target['scene']
        loss_scene = 0
        for index in range(len(scene_target_list[0])):
            scene_prediction = scene_prediction_list[:, index, :]
            scene_target = [scene_target_list[0][index]]
            scene_target_tensor = torch.tensor(scene_target).to(scene_prediction.device)
            loss_scene += F.cross_entropy(scene_prediction, scene_target_tensor.long())
        loss_scene = loss_scene/len(scene_target_list[0])

        losses = {"attribute": loss_personAttr+loss_sentiment+loss_scene}
        return losses

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        num_boxes,
        inter_idx,
        positive_map,
        time_mask,
        action_target,
        action_flag,
        attribute_target,
        noun_target,
        future_output,
        future_target,
        attribute=True,
        **kwargs,
    ):
        loss_map = {
            "boxes": self.loss_boxes,
            "sted": self.loss_sted,
            "guided_attn": self.loss_guided_attn,
            "action": self.loss_actions,
            "attribute": self.loss_attribute,
            "noun": self.loss_nouns,
            "future": self.loss_future,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ["sted", "guided_attn"]:
            return loss_map[loss](
                outputs, num_boxes, inter_idx, positive_map, time_mask, **kwargs
            )
        if loss  == "action" and action_flag:
            return loss_map[loss](
                outputs, action_target, **kwargs
            )
        if loss == 'attribute':
            return loss_map[loss](
                outputs, attribute_target, **kwargs
            )
        if loss == 'noun':
            return loss_map[loss](
                outputs, noun_target, **kwargs
            )
        if loss == 'future':
            return loss_map[loss](
                future_output, future_target, **kwargs
            )

        return loss_map[loss](outputs, targets, num_boxes, **kwargs)
    def forward(self, outputs, targets, inter_idx=None, time_mask=None,
                action_target=None, attribute_target=None, noun_target=None, future_output=None, future_target=None, attribute=True):

        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        if inter_idx is not None and time_mask is not None:
            # construct a map such that positive_map[k, i] = True iff num_frame i lies inside the annotated moment k
            positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
            for k, idx in enumerate(inter_idx):
                if idx[0] < 0:  # empty intersection
                    continue
                positive_map[k][idx[0] : idx[1] + 1].fill_(True)

            positive_map = positive_map.to(time_mask.device)
        elif time_mask is None:
            positive_map = None
        if not attribute:
            loss_list = self.losses[:-1]
        else:
            loss_list = self.losses
        losses = {}
        for loss in loss_list:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    num_boxes,
                    inter_idx,
                    positive_map,
                    time_mask,
                    action_target,
                    True,
                    attribute_target,
                    noun_target,
                    future_output,
                    future_target,
                    attribute=attribute,
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses[:-4]:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        num_boxes,
                        inter_idx,
                        positive_map,
                        time_mask,
                        None,
                        False,
                        None,
                        None,
                        None,
                        None,
                        attribute=False,
                        **kwargs,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = TubeDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        video_max_len=args.video_max_len_train,
        stride=args.stride,
        guided_attn=args.guided_attn,
        fast=args.fast,
        fast_mode=args.fast_mode,
        sted=args.sted,
        attribute=args.attribute,
    )
    weight_dict = {
        "loss_bbox": 0, # args.bbox_loss_coef,
        "loss_giou": 0, # args.giou_loss_coef,
        "loss_sted": 0, # args.sted_loss_coef,
        "action": 0.1,   #0.01,#0.05, #0.03, #0.04
        "attribute": 0,   #1.0 
        "noun": 0.1,
        "future": 1.0
    }
    if args.guided_attn:
        weight_dict["loss_guided_attn"] = 0 # args.guided_attn_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["boxes", "sted"] if args.sted else ["boxes"]
    if args.guided_attn:
        losses += ["guided_attn"]

    losses+=['action']
    losses+=['noun']
    losses+=['future']
    losses+=['attribute']

    criterion = SetCriterion(
        losses=losses,
        sigma=args.sigma,
    )
    criterion.to(device)

    return model, criterion, weight_dict