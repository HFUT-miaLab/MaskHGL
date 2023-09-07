import random

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_scatter import scatter_mean, scatter_max
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, dropout=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            residual=True,
            dropout=dropout
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            tmp, A = self.attn(self.norm(x), return_attn=True)
            x = x + tmp
            return x, A
        else:
            x = x + self.attn(self.norm(x), return_attn=False)
            return x


# 训练中随机掩码
class HyperGTv5_2wMHRCv3(nn.Module):
    def __init__(self, feat_dim, trans_dim, n_class, mask_ratio=0.5, mask_p=0.75, dropout=0.2):
        super(HyperGTv5_2wMHRCv3, self).__init__()
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.mask_p = mask_p
        self.feat_dim = feat_dim
        self.trans_dim = trans_dim
        self.hgc1 = HypergraphConv(self.feat_dim, self.feat_dim, use_attention=True, dropout=dropout)
        self.hgc2 = HypergraphConv(self.feat_dim, self.feat_dim, use_attention=True, dropout=dropout)

        self.fc1 = nn.Sequential(nn.Linear(self.feat_dim, self.trans_dim), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(self.feat_dim, self.trans_dim), nn.ReLU())

        self.cls_token = nn.Parameter(torch.randn(1, self.trans_dim))
        self.transformer_encoder = TransLayer(dim=self.trans_dim, dropout=dropout)
        self.norm = nn.LayerNorm(self.trans_dim)

        self.bag_classifier = nn.Sequential(nn.Linear(self.trans_dim, self.trans_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.trans_dim, n_class))
        self.instance_classifier = nn.Sequential(nn.Linear(self.trans_dim, self.trans_dim),
                                                 nn.ReLU(),
                                                 nn.Linear(self.trans_dim, n_class))

        self.mask_token = nn.Parameter(torch.zeros(1, feat_dim))
        self.hgc_mhrc = HypergraphConv(feat_dim, feat_dim)
        self.fc_mhrc = nn.Sequential(nn.Linear(self.feat_dim, feat_dim))

    def forward(self, x, hyperedge_index, hyperedge_attr, train=False):
        if train:  # train
            isMask = (random.uniform(0, 1) < self.mask_p)  # whether mask

            if isMask:
                x_masked, mask, ids_restore = self.random_masking(x, mask_ratio=self.mask_ratio)
            else:
                x_masked = x

            # input: x: N * D
            x_masked = F.relu(self.hgc1(x_masked, hyperedge_index, hyperedge_attr=hyperedge_attr))  # N * 384
            hyperedge_attr = self.hyperedge_attr_update(x_masked, hyperedge_index)
            out = self.fc1(x_masked)
            x_masked = F.relu(self.hgc2(x_masked, hyperedge_index, hyperedge_attr=hyperedge_attr))  # N * 384
            hyperedge_attr = self.hyperedge_attr_update(x_masked, hyperedge_index)
            out += self.fc2(x_masked)

            # MHRC
            if isMask:
                x_masked = self.hgc_mhrc(x_masked, hyperedge_index, hyperedge_attr=hyperedge_attr)
                pred = self.fc_mhrc(x_masked)
                mhrc_loss = self.mhrc_loss(pred=pred, target=x, mask=mask)

            # cls token
            out = torch.unsqueeze(torch.cat((self.cls_token, out), dim=0), dim=0)
            # TransLayer and get cls token for classifier
            out = torch.squeeze(self.norm(self.transformer_encoder(out)))  # B(1) * (N + 1) * n_hid
            bag_feats = out[0, :]
            instance_feats = out

            # WSI-level classification
            bag_logits = self.bag_classifier(bag_feats)
            bag_logits = torch.unsqueeze(bag_logits, dim=0)

            # node classification
            instance_logits = self.instance_classifier(instance_feats)
            if isMask:
                return bag_logits, instance_logits, mhrc_loss
            else:
                return bag_logits, instance_logits, 0
        else:  # inference
            x = F.relu(self.hgc1(x, hyperedge_index, hyperedge_attr=hyperedge_attr))  # N * 384
            hyperedge_attr = self.hyperedge_attr_update(x, hyperedge_index)
            out = self.fc1(x)
            x = F.relu(self.hgc2(x, hyperedge_index, hyperedge_attr=hyperedge_attr))  # N * 384
            out += self.fc2(x)

            # cls token
            out = torch.unsqueeze(torch.cat((self.cls_token, out), dim=0), dim=0)
            # TransLayer and get cls token for classifier
            out, A = self.transformer_encoder(out, return_attn=True)
            A = torch.squeeze(A)[:, -out.shape[1]:, -out.shape[1]:]
            A = torch.mean(A[:, 0, 1:], dim=0)
            out = torch.squeeze(self.norm(out))  # B(1) * (N + 1) * n_hid
            bag_feats = out[0, :]
            instance_feats = out[1:, :]

            # WSI-level classification
            bag_logits = self.bag_classifier(bag_feats)
            bag_logits = torch.unsqueeze(bag_logits, dim=0)

            # node classification
            instance_logits = self.instance_classifier(instance_feats)
            return bag_logits, instance_logits, A

    def hyperedge_attr_update(self, x, hyperedge_index, num_feat_hyperedge=150):
        # print(x.shape)
        feat_hyperedge_index = hyperedge_index[:, :int(hyperedge_index.shape[1] / 2)]
        spatial_hyperedge_index = hyperedge_index[:, int(hyperedge_index.shape[1] / 2):]

        feature_dim = x.shape[1]
        num_hyperedge = torch.max(hyperedge_index[1]).item() + 1

        hyperedge_attr = torch.zeros((num_hyperedge, feature_dim))
        feat_hyperedge_attr = scatter_mean(src=x, index=feat_hyperedge_index[1], dim=0)
        hyperedge_attr[:feat_hyperedge_attr.shape[0], :] = feat_hyperedge_attr

        spatial_hyperedge_attr = scatter_mean(src=x, index=spatial_hyperedge_index[1], dim=0)[num_feat_hyperedge:, :]
        hyperedge_attr[num_feat_hyperedge:, :] = spatial_hyperedge_attr

        return hyperedge_attr.cuda()

    def random_masking(self, x, mask_ratio):
        L, D = x.shape  # batch, length, dim
        len_keep = int(L * mask_ratio)

        noise = torch.rand(L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        # print(ids_keep)
        # x[ids_keep.unsqueeze(-1).repeat(1, D)] = self.mask_token
        x[ids_keep, :] = self.mask_token
        x_masked = x
        assert x_masked.shape == (L, D)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones(L, device=x.device)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=0, index=ids_restore)

        return x_masked, mask, ids_restore

    def mhrc_loss(self, pred, target, mask):
        """
        x: [L, D]
        pred: [L, D]
        mask: [L], 0 is keep, 1 is remove,
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()

        return loss
