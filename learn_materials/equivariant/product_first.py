import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch_scatter import scatter_mean, scatter_add
import learn_materials.utils as utils
from learn_materials.equivariant.lattice import build_generators
from learn_materials.equivariant.groups import *
from learn_materials.equivariant.autoequiv.core import (
    LinearEquiv,
)
import learn_materials.training as trn
from learn_materials.models.embeddings import swish
import learn_materials.models.embeddings as embed
from learn_materials.models.hyperparams import create_full_hyperparams


LAYER_GENERATORS = utils.load_json(utils.SRC_PATH / "equivariant" / "layer_generators_prism.json")

default_parameters = {
    "model": "ProductFirst",
    "cell_dim": 100,
    "supergraphs": True,
    "graphs": True,
    "hierarchical": False,
    "break_symmetry": True,
    "final_embedding": 100,
    "lr": 1e-3,
    "bond_props": ["distance"],
    "bond_expansion": 20,
    "bond_width_factor": 2.0,
    "site_embedding": 100,
    "bond_embedding": 20,
    "num_conv_layers": 6,
    "num_fc_layers": 1,
    "batch_norm": False,
    "dropout": 0.0,
    "nonlinearity": "swish",
    "normalization": "none",
    "group": "set",
}

"""
Modifications with respect to the original architecture:
*
"""

NON_LINEARITIES = {"relu": F.relu, "swish": swish}
SITE_LEN = {"atomic_number": 103, "hubbard": 104, "period": 39}


def get_generators(lattice, position="intermediate"):
    if position == "first":
        in_generators = LAYER_GENERATORS[lattice]["restricted"]
        out_generators = LAYER_GENERATORS[lattice]["full"]
    elif position == "intermediate":
        in_generators = LAYER_GENERATORS[lattice]["full"]
        out_generators = LAYER_GENERATORS[lattice]["full"]
    elif position == "last":
        in_generators = LAYER_GENERATORS[lattice]["full"]
        out_generators = LAYER_GENERATORS[lattice]["restricted"]
    elif position == "restricted":
        in_generators = LAYER_GENERATORS[lattice]["restricted"]
        out_generators = LAYER_GENERATORS[lattice]["restricted"]
    else:
        raise ValueError("Type must be one of 'first', 'intermediate' or 'last'.")

    return in_generators, out_generators


class NormLinear(nn.Module):
    def __init__(self, normalization, input_channel, output_channel):
        super(NormLinear, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.linear_layer = nn.Linear(input_channel, output_channel)
        self.normalization = normalization
        if self.normalization == "layer":
            self.norm_layer = nn.LayerNorm(output_channel)
        elif self.normalization == "instance":
            self.norm_layer = nn.InstanceNorm1d(output_channel, affine=True)
        elif self.normalization == "batch":
            self.norm_layer = nn.BatchNorm1d(output_channel, affine=True)

    def forward(self, x):
        x = self.linear_layer(x)
        if self.normalization in ["layer", "batch"]:
            x = self.norm_layer(x)

        return x


class NodeFunction(nn.Module):
    def __init__(self, site_len, message_len, h1, nonlinearity, normalization):
        super(NodeFunction, self).__init__()
        self.site_len = site_len
        self.message_len = message_len
        self.nonlinearity = NON_LINEARITIES[nonlinearity]
        self.normalization = normalization

        self.fc1 = NormLinear(self.normalization, self.site_len + self.message_len, h1)
        self.fc2 = NormLinear(self.normalization, h1, self.site_len)

    def forward(self, sites, sites_update):
        vectors = torch.cat((sites_update, sites), 1)

        vectors = self.nonlinearity(self.fc1(vectors))
        vectors = self.fc2(vectors)
        sites = sites + vectors

        return sites


class GroupLayer(nn.Module):
    def __init__(self, in_cell_dim, out_cell_dim, position, lattice, normalization):
        super(GroupLayer, self).__init__()
        self.in_cell_dim = in_cell_dim
        self.out_cell_dim = out_cell_dim
        self.lattice = lattice
        self.normalization = normalization
        self.fan = {
            "first": "channels",
            "intermediate": "features",
            "last": "default",
            "restricted": "channels",
        }
        if self.normalization == "layer":
            self.norm_layer = nn.LayerNorm([out_cell_dim, len(get_generators(lattice, position)[1][0])])
        elif self.normalization == "instance":
            self.norm_layer = nn.InstanceNorm1d(out_cell_dim, affine=True)
        elif self.normalization == "batch":
            self.norm_layer = nn.BatchNorm1d(out_cell_dim, affine=True)

        self.group_layer = LinearEquiv(
            *get_generators(lattice, position),
            self.in_cell_dim,
            self.out_cell_dim,
            fan=self.fan[position],
        )

    def forward(self, cells):
        cells = self.group_layer(cells)
        if self.normalization in ["layer", "batch", "instance"]:
            cells = self.norm_layer(cells)

        return cells


class CellLayer(nn.Module):
    def __init__(
        self,
        site_len,
        bond_len,
        cell_dim,
        lattice,
        model_type,
        num_fc_layers,
        nonlinearity,
        normalization,
    ):
        super(CellLayer, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.cell_dim = cell_dim
        self.lattice = lattice
        self.model_type = model_type
        self.num_fc_layers = num_fc_layers
        self.nonlinearity = NON_LINEARITIES[nonlinearity]
        self.normalization = normalization

        if self.model_type == "regular_layer":
            self.group_layer_1 = GroupLayer(
                2 * self.site_len + self.bond_len, self.cell_dim, "first", lattice, self.normalization
            )
            self.group_layer_2 = GroupLayer(self.cell_dim, self.cell_dim, "intermediate", lattice, self.normalization)
            self.group_layer_3 = GroupLayer(self.cell_dim, self.site_len, "last", lattice, self.normalization)
        elif self.model_type == "simple":
            self.group_layer_1 = GroupLayer(2 * self.site_len + self.bond_len, self.site_len, "restricted", lattice, self.normalization)
            self.group_layer_2 = GroupLayer(self.site_len, self.site_len, "restricted", lattice, self.normalization)

    def forward(self, cells):
        if self.model_type == "regular_layer":
            cells = self.nonlinearity(self.group_layer_1(cells))
            cells = self.nonlinearity(self.group_layer_2(cells))
            cells = self.nonlinearity(self.group_layer_3(cells))
        elif self.model_type == "simple":
            cells = self.nonlinearity(self.group_layer_1(cells))
            cells = self.nonlinearity(self.group_layer_2(cells))

        return cells


class ProductLayer(nn.Module):
    def __init__(
        self,
        batch_size,
        site_len,
        bond_len,
        cell_dim,
        lattice,
        batch_norm,
        model_type,
        num_fc_layers,
        hierarchical,
        dropout,
        nonlinearity,
        normalization,
    ):
        super(ProductLayer, self).__init__()
        self.batch_size = batch_size
        self.site_len = site_len
        self.bond_len = bond_len
        self.cell_dim = cell_dim
        self.lattice = lattice
        self.batch_norm = batch_norm
        self.model_type = model_type
        self.num_fc_layers = num_fc_layers
        self.hierarchical = hierarchical
        self.dropout = dropout
        self.nonlinearity = nonlinearity
        self.normalization = normalization

        self.cell_layer = CellLayer(
            self.site_len,
            self.bond_len,
            self.cell_dim,
            lattice,
            self.model_type,
            self.num_fc_layers,
            self.nonlinearity,
            self.normalization,
        )
        self.node_function = NodeFunction(
            self.site_len, self.site_len, self.site_len, self.nonlinearity, self.normalization
        )
        self.lattice_attention = NormLinear(self.normalization, self.site_len, 1)

        self.cell_layer_identity = CellLayer(
            self.site_len,
            self.bond_len,
            self.cell_dim,
            lattice,
            self.model_type,
            self.num_fc_layers,
            self.nonlinearity,
            self.normalization,
        )
        self.lattice_attention_identity = NormLinear(self.normalization, self.site_len, 1)

        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        sites,
        bonds,
        bonds_identity,
        indices1,
        indices2,
        indices1_cells,
        indices2_cells_oh,
        indices1_idenity,
        indices2_idenity,
        indices1_idenity_cells,
        indices2_idenity_cells_oh,
    ):
        sites = self.dropout(sites)

        lattice_sites = self.compute_cells(
            sites,
            bonds,
            indices1,
            indices2,
            indices1_cells,
            indices2_cells_oh,
            self.cell_layer,
        )

        lattice_sites_identity = self.compute_cells(
            sites,
            bonds_identity,
            indices1_idenity,
            indices2_idenity,
            indices1_idenity_cells,
            indices2_idenity_cells_oh,
            self.cell_layer_identity,
        )
        if self.hierarchical:
            sites = self.node_function(sites, (lattice_sites))
        else:
            sites = self.node_function(sites, (lattice_sites + lattice_sites_identity))

        return sites

    def compute_cells(
        self,
        sites,
        bonds,
        indices1,
        indices2,
        indices1_cells,
        indices2_cells_oh,
        layer,
    ):
        sites1 = torch.index_select(sites, 0, indices1)
        sites2 = torch.index_select(sites, 0, indices2)
        vectors = torch.cat((sites1, sites2, bonds), 1)
        vectors_cells = torch.einsum("ij,ik->ijk", vectors, indices2_cells_oh)
        vectors_cells = layer(vectors_cells)
        indices1_cells = indices1_cells[:, None, None].repeat(1, self.site_len, 1)
        lattice_sites = torch.gather(vectors_cells, 2, indices1_cells, sparse_grad=False).squeeze()
        lattice_sites = torch.sigmoid(self.lattice_attention(lattice_sites)) * lattice_sites
        lattice_sites = scatter_add(lattice_sites, indices1, 0)

        return lattice_sites


class FinalLayer(nn.Module):
    def __init__(
        self, site_prediction, site_len, final_size, output_len, dropout, nonlinearity, normalization
    ):
        super(FinalLayer, self).__init__()
        self.site_len = site_len
        self.site_prediction = site_prediction

        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout)] * 4)
        self.normalization = normalization
        self.fc1 = NormLinear(self.normalization, self.site_len, final_size)
        self.fc2 = NormLinear(self.normalization, final_size, final_size)
        self.fc3 = NormLinear(self.normalization, final_size, final_size)
        self.fc4 = NormLinear("none", final_size, output_len)
        self.nonlinearity = NON_LINEARITIES[nonlinearity]

    def forward(self, sites, graph_to_sites):
        sites = self.dropout_layers[0](sites)
        sites = self.nonlinearity(self.fc1(sites))
        sites = self.dropout_layers[1](sites)
        sites = self.fc2(sites)

        if not self.site_prediction:
            vectors = scatter_mean(sites, graph_to_sites, 0)
        else:
            vectors = sites
        vectors = self.dropout_layers[2](vectors)
        vectors = self.nonlinearity(self.fc3(vectors))
        vectors = self.dropout_layers[3](vectors)
        properties = self.fc4(vectors)

        return properties


class ProductFirst(nn.Module):
    def __init__(self, hyperparams):
        super(ProductFirst, self).__init__()
        self.supercell_size = hyperparams.supercell_size
        self.site_props = SITE_LEN[hyperparams.site_props]
        self.num_elements = self.site_props
        self.in_site_len = self.site_props + self.supercell_size
        self.in_bond_len = len(hyperparams.bond_props)
        self.bond_expansion = hyperparams.bond_expansion
        self.bond_width_factor = hyperparams.bond_width_factor
        self.site_embedding_len = hyperparams.site_embedding
        self.bond_embedding_len = hyperparams.bond_embedding
        self.cell_dim = hyperparams.cell_dim
        self.final_embedding = hyperparams.final_embedding
        self.output_len = len(hyperparams.target_props) * hyperparams.loss_output_len
        self.group = hyperparams.group
        self.batch_size = hyperparams.batch_size
        self.batch_norm = hyperparams.batch_norm
        self.model_type = hyperparams.model_type
        self.num_fc_layers = hyperparams.num_fc_layers
        self.dropout = hyperparams.dropout
        self.nonlinearity = hyperparams.nonlinearity
        self.max_distance = hyperparams.max_distance
        self.hierarchical = hyperparams.hierarchical
        self.break_symmetry = hyperparams.break_symmetry
        self.normalization = hyperparams.normalization
        self.site_prediction = hyperparams.target_props == ["Magmoms"]

        if self.break_symmetry:
            self.site_embedding_layer = embed.EmbeddingLayer(self.in_site_len, self.site_embedding_len)
        else:
            self.site_embedding_layer = embed.EmbeddingLayer(self.num_elements, self.site_embedding_len)
        self.bond_embedding_layer = embed.EmbeddingLayer(self.bond_expansion, self.bond_embedding_len)
        self.layer_1 = ProductLayer(
            self.batch_size,
            self.site_embedding_len,
            self.bond_embedding_len,
            self.cell_dim,
            self.group,
            self.batch_norm,
            self.model_type,
            self.num_fc_layers,
            self.hierarchical,
            self.dropout,
            self.nonlinearity,
            self.normalization,
        )
        self.layers = nn.ModuleList(
            [
                ProductLayer(
                    self.batch_size,
                    self.site_embedding_len,
                    self.bond_embedding_len,
                    self.cell_dim,
                    self.group,
                    self.batch_norm,
                    self.model_type,
                    self.num_fc_layers,
                    self.hierarchical,
                    self.dropout,
                    self.nonlinearity,
                    self.normalization,
                )
                for i in range(hyperparams.num_conv_layers - 1)
            ]
        )

        self.final_layer = FinalLayer(
            self.site_prediction,
            self.site_embedding_len,
            self.final_embedding,
            self.output_len,
            self.dropout,
            self.nonlinearity,
            self.normalization,
        )

    def forward(self, x):
        (
            sites,
            bonds,
            _,
            indices1,
            indices2,
            indices_cells,
            indices_identity,
            graph_to_sites,
            _,
        ) = x

        bonds = embed.gaussian_basis(
            bonds,
            self.max_distance,
            self.bond_expansion,
            self.bond_width_factor * self.max_distance / self.bond_expansion,
        )
        if not self.break_symmetry:
            sites = sites[:, : self.num_elements]
        sites = self.site_embedding_layer(sites)
        bonds = self.bond_embedding_layer(bonds)

        (
            indices1_identity,
            indices2_identity,
            bonds_identity,
        ) = self.compute_indices_identity(indices1, indices2, bonds, indices_identity)
        (
            indices1_identity_cells,
            indices2_identity_cells_oh,
        ) = embed.compute_cells_sparse(indices1_identity, indices2_identity, indices_cells)
        indices1_cells, indices2_cells_oh = embed.compute_cells_sparse(indices1, indices2, indices_cells)

        sites = self.layer_1(
            sites,
            bonds,
            bonds_identity,
            indices1,
            indices2,
            indices1_cells,
            indices2_cells_oh,
            indices1_identity,
            indices2_identity,
            indices1_identity_cells,
            indices2_identity_cells_oh,
        )

        for layer in self.layers:
            sites = layer(
                sites,
                bonds,
                bonds_identity,
                indices1,
                indices2,
                indices1_cells,
                indices2_cells_oh,
                indices1_identity,
                indices2_identity,
                indices1_identity_cells,
                indices2_identity_cells_oh,
            )

        properties = self.final_layer(sites, graph_to_sites)

        return properties

    def compute_indices_identity(self, indices1, indices2, bonds, indices_identity):
        identity1 = torch.index_select(indices_identity, 0, indices1)
        identity2 = torch.index_select(indices_identity, 0, indices2)

        edges = identity1 == identity2

        indices1_identity = indices1[edges]
        indices2_identity = indices2[edges]
        bonds_identity = bonds[edges]

        return indices1_identity, indices2_identity, bonds_identity


def main(argv):
    # FIXME: creating the hyperparams object outside this file made the tests
    # fail because the arg_parser seemed to be shared.
    hyperparams = create_full_hyperparams(default_parameters, {}, argv)

    torch.manual_seed(hyperparams.seed)
    torch.cuda.manual_seed(hyperparams.seed)
    np.random.seed(hyperparams.seed)
    random.seed(hyperparams.seed)
    model = ProductFirst(hyperparams)
    # print(list(model.named_parameters()))
    if hyperparams.mode == "train":
        trn.train_and_monitor(hyperparams, model)
    if hyperparams.mode == "evaluate":
        print(f"Running model HierarchicalFirst on dataset {hyperparams.run_dataset}")
        trn.evaluate(model, hyperparams)
    elif hyperparams.mode == "inference":
        print(f"Running model HierarchicalFirst on dataset {hyperparams.run_dataset}")
        trn.inference(model, hyperparams)


if __name__ == "__main__":
    main(sys.argv[1:])
