#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from network.layers import build_mlp
from network.pointnet2_msg import pointnet2
from graph.Space_edge import Space_Category
import torch.nn.functional as F

"""
PyTorch modules for dealing with graphs.
"""


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """

    def __init__(self, input_dim, attributes_dim=0, output_dim=None, hidden_dim=512,
                 pooling='avg', mlp_normalization='none'):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim + 2 * attributes_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
        - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
        # p vecs have shape (T, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (T, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        return new_obj_vecs, new_p_vecs


class Linear4(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=128):
        super(Linear4, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(in_size, hidden_size),
                        # nn.BatchNorm1d(hidden_size),
                        nn.LeakyReLU(),
                        # nn.Dropout(0.5),
                        nn.Linear(hidden_size, hidden_size),
                        # nn.BatchNorm1d(hidden_size),
                        nn.LeakyReLU(),
                        # nn.Dropout(0.5),
                        nn.Linear(hidden_size, out_size),
                     )

    def forward(self, x):
        return self.model(x)


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                 mlp_normalization='none'):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'mlp_normalization': mlp_normalization,
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs

class GraphNet(nn.Module):
    def __init__(self, embedding_dim=128, attribute_dim=15, gconv_dim=128, gconv_hidden_dim=512, gconv_num_layers=5, mlp_normalization='none', pred_dim=16):
        super(GraphNet, self).__init__()

        self.pointnet2 = pointnet2(embedding_dim)

        self.state_embeddings = nn.Embedding(Space_Category.st_num, int(embedding_dim / 2))
        self.drt_embeddings = nn.Embedding(Space_Category.dr_num, int(embedding_dim / 2))

        self.gconv = GraphTripleConv(
            embedding_dim,
            attributes_dim=attribute_dim, 
            output_dim=gconv_dim,
            hidden_dim=gconv_hidden_dim,
            mlp_normalization=mlp_normalization
        )
        self.gconv_net = GraphTripleConvNet(
            gconv_dim,
            num_layers=gconv_num_layers - 1,
            mlp_normalization=mlp_normalization
        )

        self.linear_edge = Linear4(128, pred_dim)

    def forward(self, objs, triples, pcs, nodes_to_graph, edge_to_graph):
        O, T = objs.size(0), triples.size(0)
        start, state, direct, v_direct, end = triples.chunk(5, dim=1)
        start, state, direct, v_direct, end = [x.squeeze(1) for x in [start, state, direct, v_direct, end]]
        edges = torch.stack([start, end], dim=1)

        '''pointnet++'''
        pcs = pcs.transpose(2, 1)
        # print(pcs.shape)
        pc_vecs = self.pointnet2(pcs)

        '''attribute'''
        obj_vecs = torch.cat([pc_vecs, objs], dim=1)

        '''edge embedding'''
        state_vecs = self.state_embeddings(state)
        drt_vecs = self.drt_embeddings(direct)
        edge_vecs = torch.cat([state_vecs, drt_vecs], dim=1)

        '''graph net'''
        obj_vecs, edge_vecs = self.gconv(obj_vecs, edge_vecs, edges)
        obj_vecs, edge_vecs = self.gconv_net(obj_vecs, edge_vecs, edges)  # (O, 128)

        '''predict'''
        pred = self.linear_edge(edge_vecs)
        return pred


class get_classify_loss(torch.nn.Module):
    def __init__(self):
        super(get_classify_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.cross_entropy(pred, target)
        # total_loss = loss
        return loss

class get_pos_loss(torch.nn.Module):
    def __init__(self):
        super(get_pos_loss, self).__init__()

    def forward(self, pred, gt_pos, gt_drt):
        if(len(pred) > 0):
            loss = torch.linalg.norm(torch.cross(pred - gt_pos, gt_drt) / torch.linalg.norm(gt_drt, dim=1).view(-1, 1), dim=1)
            loss = torch.sum(loss) / len(pred)
        else:
            loss = 0
        # total_loss = loss
        return loss

class get_drt_loss(torch.nn.Module):
    def __init__(self):
        super(get_drt_loss, self).__init__()

    def forward(self, pred, target):
        if(len(pred) > 0):
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_drt = 1 - cos(pred, target)
            loss = F.mse_loss(cos_drt, torch.zeros_like(cos_drt).cuda())
        else:
            loss = 0
        # total_loss = loss
        return loss
