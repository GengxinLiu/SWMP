import os
import sys
from tkinter.messagebox import NO
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_scatter
from dataset_loader.structure_graph import Tree

from network.layers import build_mlp
from graph.Space_edge import Space_Category
from network.pointnet2_msg import pointnet2, pointnet2_reduce
from structure_net.DGCNN import DGCNN_cls
from graph.graph_extraction import motion

'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-----------------------------------------------------------Graph Net-----------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''


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
    def __init__(self, embedding_dim=256, attribute_dim=0, gconv_dim=128, gconv_hidden_dim=512, gconv_num_layers=5,
                 mlp_normalization='none', pred_dim=16):
        super(GraphNet, self).__init__()
        self.semantic_embeddings = nn.Embedding(Tree.num_sem, attribute_dim)

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

    def forward(self, objs, triples, semantics=None):
        # O, T = objs.size(0), triples.size(0)
        start, state, direct, v_direct, end = triples.chunk(5, dim=1)
        start, state, direct, v_direct, end = [x.squeeze(1) for x in [start, state, direct, v_direct, end]]
        edges = torch.stack([start, end], dim=1)

        '''attribute'''
        # print(objs)
        obj_vecs = objs
        if semantics is not None:
            # one hot
            # obj_vecs = torch.cat([obj_vecs, semantics], dim=1)
            # embedding
            semantic_vecs = self.semantic_embeddings(semantics)  # [n_nodes, 128]
            obj_vecs = torch.cat([obj_vecs, semantic_vecs], dim=1)  # [n_nodes, 384]
        # print(obj_vecs.shape)

        '''edge embedding'''
        state_vecs = self.state_embeddings(state)
        drt_vecs = self.drt_embeddings(direct)
        edge_vecs = torch.cat([state_vecs, drt_vecs], dim=1)

        '''graph net'''
        obj_vecs, edge_vecs = self.gconv(obj_vecs, edge_vecs, edges)
        obj_vecs, edge_vecs = self.gconv_net(obj_vecs, edge_vecs, edges)  # (O, 128)

        '''predict'''
        pred = self.linear_edge(edge_vecs)
        pred_m_type = pred[:, :10]
        # pred_pos = torch.sigmoid(pred[:, 10:13])
        pred_pos = pred[:, 10:13]
        pred_drt = torch.tanh(pred[:, 13:16])
        return pred_m_type, pred_pos, pred_drt


'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------Structure Net-----------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''


class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu


class PartEncoder(nn.Module):

    def __init__(self, feat_len):
        super(PartEncoder, self).__init__()

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_len, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(feat_len)

        self.sampler = PartFeatSampler(feature_size=feat_len, probabilistic=False)

    def forward(self, pc):
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net = self.sampler(net)

        return net


class NodeEncoder(nn.Module):

    def __init__(self, geo_feat_len, node_feat_len):
        super(NodeEncoder, self).__init__()

        self.part_encoder = DGCNN_cls(geo_feat_len)

        # box feature
        self.mlp1 = nn.Linear(15, geo_feat_len)
        self.mlp2 = nn.Linear(2 * geo_feat_len, node_feat_len)

    def forward(self, geo, box):
        # print(geo.shape)
        # print(box.shape)
        geo = geo.transpose(2, 1)
        geo_feat = self.part_encoder(geo)
        box_feat = torch.relu(self.mlp1(box))
        # print(geo_feat.shape)
        # print(box_feat.shape)

        all_feat = torch.cat([box_feat, geo_feat], dim=1)
        all_feat = torch.relu(self.mlp2(all_feat))

        return all_feat, geo_feat


class GNNChildEncoder(nn.Module):

    def __init__(self, node_feat_size, attribute_size, hidden_size, node_symmetric_type, \
                 edge_symmetric_type, num_iterations, edge_type_num):
        super(GNNChildEncoder, self).__init__()

        self.node_symmetric_type = node_symmetric_type
        self.edge_symmetric_type = edge_symmetric_type
        self.num_iterations = num_iterations
        self.edge_type_num = edge_type_num

        self.child_op = nn.Linear(node_feat_size + attribute_size, hidden_size)
        self.node_edge_op = torch.nn.ModuleList()
        for i in range(self.num_iterations):
            self.node_edge_op.append(nn.Linear(hidden_size * 2 + edge_type_num, hidden_size))

        self.parent_op = nn.Linear(hidden_size * (self.num_iterations + 1), node_feat_size)

    """
        Input Arguments:
            child feats: b x max_childs x feat_dim
            child exists: b x max_childs x 1
            edge_type_onehot: b x num_edges x edge_type_num
            edge_indices: b x num_edges x 2
    """

    def forward(self, child_feats, child_exists, edge_type_onehot, edge_indices):
        batch_size = child_feats.shape[0]
        max_childs = child_feats.shape[1]
        num_edges = edge_indices.shape[1]

        if batch_size != 1:
            raise ValueError('Currently only a single batch is supported.')

        # perform MLP for child features
        child_feats = torch.relu(self.child_op(child_feats))
        hidden_size = child_feats.size(-1)

        # zero out non-existent children
        child_feats = child_feats * child_exists
        child_feats = child_feats.view(1, max_childs, -1)

        # combine node features before and after message-passing into one parent feature
        iter_parent_feats = []
        if self.node_symmetric_type == 'max':
            iter_parent_feats.append(child_feats.max(dim=1)[0])
        elif self.node_symmetric_type == 'sum':
            iter_parent_feats.append(child_feats.sum(dim=1))
        elif self.node_symmetric_type == 'avg':
            iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
        else:
            raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        if self.num_iterations > 0 and num_edges > 0:
            edge_feats = edge_type_onehot

        edge_indices_from = edge_indices[:, :, 0].view(-1, 1).expand(-1, hidden_size)

        # perform Graph Neural Network for message-passing among sibling nodes
        for i in range(self.num_iterations):
            if num_edges > 0:
                # MLP for edge features concatenated with adjacent node features
                node_edge_feats = torch.cat([
                    child_feats[0:1, edge_indices[0, :, 0], :],  # start node features
                    child_feats[0:1, edge_indices[0, :, 1], :],  # end node features
                    edge_feats], dim=2)  # edge features

                node_edge_feats = node_edge_feats.view(num_edges, -1)
                node_edge_feats = torch.relu(self.node_edge_op[i](node_edge_feats))
                node_edge_feats = node_edge_feats.view(num_edges, -1)

                # aggregate information from neighboring nodes
                new_child_feats = child_feats.new_zeros(max_childs, hidden_size)
                if self.edge_symmetric_type == 'max':
                    new_child_feats, _ = torch_scatter.scatter_max(node_edge_feats, edge_indices_from, dim=0,
                                                                   out=new_child_feats)
                elif self.edge_symmetric_type == 'sum':
                    new_child_feats = torch_scatter.scatter_add(node_edge_feats, edge_indices_from, dim=0,
                                                                out=new_child_feats)
                elif self.edge_symmetric_type == 'avg':
                    new_child_feats = torch_scatter.scatter_mean(node_edge_feats, edge_indices_from, dim=0,
                                                                 out=new_child_feats)
                else:
                    raise ValueError(f'Unknown edge symmetric type: {self.edge_symmetric_type}')

                child_feats = new_child_feats.view(1, max_childs, hidden_size)

            # combine node features before and after message-passing into one parent feature
            if self.node_symmetric_type == 'max':
                iter_parent_feats.append(child_feats.max(dim=1)[0])
            elif self.node_symmetric_type == 'sum':
                iter_parent_feats.append(child_feats.sum(dim=1))
            elif self.node_symmetric_type == 'avg':
                iter_parent_feats.append(child_feats.sum(dim=1) / child_exists.sum(dim=1))
            else:
                raise ValueError(f'Unknown node symmetric type: {self.node_symmetric_type}')

        # concatenation of the parent features from all iterations (as in GIN, like skip connections)
        parent_feat = torch.cat(iter_parent_feats, dim=1)

        # back to standard feature space size
        parent_feat = torch.relu(self.parent_op(parent_feat))

        return parent_feat


class RecursiveEncoder(nn.Module):

    def __init__(self, geo_feat_size,
                 node_feature_size,
                 attribute_size,
                 hidden_size,
                 node_symmetric_type,
                 edge_symmetric_type,
                 num_gnn_iterations,
                 edge_types_size,
                 max_child_num):
        super(RecursiveEncoder, self).__init__()

        self.node_encoder = NodeEncoder(geo_feat_len=geo_feat_size, node_feat_len=node_feature_size)

        # semantic embedding
        self.semantic_embeddings = nn.Embedding(Tree.num_sem, attribute_size)
        self.state_embeddings = nn.Embedding(Space_Category.st_num, int(edge_types_size / 2))
        self.drt_embeddings = nn.Embedding(Space_Category.dr_num, int(edge_types_size / 2))

        self.child_encoder = GNNChildEncoder(
            node_feat_size=node_feature_size,
            attribute_size=attribute_size,
            hidden_size=hidden_size,
            node_symmetric_type=node_symmetric_type,
            edge_symmetric_type=edge_symmetric_type,
            num_iterations=num_gnn_iterations,
            edge_type_num=edge_types_size)

        self.max_child_num = max_child_num

        # one hot
        # self.graph_net = GraphNet(embedding_dim=node_feature_size, attribute_dim=Tree.num_sem, pred_dim=motion().len - 1 + 6)
        # embedding
        # self.semantic_embeddings =  nn.Embedding(Tree.num_sem, geo_feat_size)
        self.graph_net = GraphNet(embedding_dim=node_feature_size, attribute_dim=128, pred_dim=motion().len - 1 + 6)

    def encode_node(self, node):
        # all_feat, geo_feat = self.node_encoder(node.geo, node.box)
        # node.geo_feat = geo_feat

        if node.is_leaf:
            all_feat, geo_feat = self.node_encoder(node.geo, node.box)
            node.node_feat = all_feat
            return all_feat
        else:
            # get features of all children
            child_feats = []
            for child in node.children:
                cur_child_feat = torch.cat(
                    [self.encode_node(child), self.semantic_embeddings(child.get_semantic_class())], dim=1)
                child_feats.append(cur_child_feat.unsqueeze(dim=1))
            child_feats = torch.cat(child_feats, dim=1)

            if child_feats.shape[1] > self.max_child_num:
                raise ValueError('Node has too many children.')

            # pad with zeros
            if child_feats.shape[1] < self.max_child_num:
                padding = child_feats.new_zeros(child_feats.shape[0], \
                                                self.max_child_num - child_feats.shape[1], child_feats.shape[2])
                child_feats = torch.cat([child_feats, padding], dim=1)

            # 1 if the child exists, 0 if it is padded
            child_exists = child_feats.new_zeros(child_feats.shape[0], self.max_child_num, 1)
            child_exists[:, :len(node.children), :] = 1

            edge_type_class, edge_indices = node.edge_tensors(type_onehot=False)
            state_vecs = self.state_embeddings(edge_type_class[:, :, 0])
            drt_vecs = self.drt_embeddings(edge_type_class[:, :, 1])
            edge_vecs = torch.cat([state_vecs, drt_vecs], dim=2)
            node_feat = self.child_encoder(child_feats, child_exists, edge_vecs, edge_indices)

            node.node_feat = node_feat

            return node_feat

    def forward(self, obj):
        # encoder node feat

        _ = self.encode_node(obj.root)  # [1, 256]

        nodes_feat, edges, target, e_mask, edge_forward, edge_backward, semantics = obj.root.get_graph()

        pred_m_type, pred_pos, pred_drt = self.graph_net(nodes_feat, edges, semantics)

        return pred_m_type, pred_pos, pred_drt, target, e_mask, edge_forward, edge_backward


'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------     loss    -----------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------------------------------------------------------------------------------------------------------'''


class get_classify_loss(torch.nn.Module):
    def __init__(self, weight=None):
        # weight: Tensor [C]，weight for each class
        super(get_classify_loss, self).__init__()
        self.weight = weight

    def forward(self, pred, target):
        loss = F.cross_entropy(pred, target, weight=self.weight)
        # total_loss = loss
        return loss


class get_pos_loss(torch.nn.Module):
    def __init__(self):
        super(get_pos_loss, self).__init__()

    def forward(self, pred, gt_pos, gt_drt):
        if (len(pred) > 0):
            loss = torch.linalg.norm(torch.cross(pred - gt_pos, gt_drt) / torch.linalg.norm(gt_drt, dim=1).view(-1, 1),
                                     dim=1)
            loss = torch.sum(loss) / len(pred)
        else:
            loss = torch.tensor(0.0)
            if pred.is_cuda:
                loss = loss.cuda()
        # total_loss = loss
        return loss


class get_drt_loss(torch.nn.Module):
    def __init__(self):
        super(get_drt_loss, self).__init__()

    def forward(self, pred, target):
        loss = torch.norm(pred - target, p=2)
        return loss


# class get_consistency_loss(torch.nn.Module):
#     def __init__(self):
#         super(get_consistency_loss, self).__init__()

#     def forward(self, edge_forward, edge_backward):
#         if len(edge_forward) > 0:
#             edge_forward = F.softmax(edge_forward, dim=1)
#             edge_backward = F.softmax(edge_backward, dim=1)
#             loss = torch.linalg.norm(edge_forward - edge_backward, dim=1)
#             loss = torch.sum(loss) / len(edge_forward)
#         else:
#             loss = torch.tensor(0.0)
#             if edge_forward.is_cuda:
#                 loss = loss.cuda()

#         return loss
