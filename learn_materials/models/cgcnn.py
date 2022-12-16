import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

import learn_materials.training as trn
import learn_materials.models.embeddings as embed
from learn_materials.models.hyperparams import create_full_hyperparams


default_parameters = {
    "model": "CGCNN",
    "graphs": True,
    "h1": 100,
    "h2": 100,
    "lr": 1e-3,
    "site_props": list(range(103)),
    "bond_props": ["distance"],
    # "target_props": ["Formation energy per atom"],
    "bond_expansion": 10,
    "site_embedding": 100,
    "bond_embedding": 20,
    "batch_norm": False,
    "use_supergraphs": False,
    "wandb_project": "equivariant_crystals_cgcnn",
}

"""
Modifications with respect to the original architecture:
*
"""

# FIXME: Vérifier si la fonction d'activation est vraiment softmax
class ConvLayer(nn.Module):
    def __init__(self, residual=True, **kwargs):
        super(ConvLayer, self).__init__()
        self.site_len = kwargs["site_len"]
        self.bond_len = kwargs["bond_len"]
        self.batch_norm = kwargs["batch_norm"]
        self.residual = residual

        self.sigmoid_layer = nn.Linear(2 * self.site_len + self.bond_len, self.site_len)
        self.softmax_layer = nn.Linear(2 * self.site_len + self.bond_len, self.site_len)
        if self.batch_norm:
            self.sigmoid_batch_norm = nn.BatchNorm1d(self.site_len)
            self.softmax_batch_norm = nn.BatchNorm1d(self.site_len)

    def forward(self, sites, bonds, indices1, indices2):
        sites1 = torch.index_select(sites, 0, indices1)
        sites2 = torch.index_select(sites, 0, indices2)

        vectors = torch.cat((sites1, sites2, bonds), 1)
        if self.batch_norm:
            vectors = torch.sigmoid(self.sigmoid_batch_norm(self.sigmoid_layer(vectors))) * F.relu(
                self.softmax_batch_norm(self.softmax_layer(vectors))
            )
        else:
            vectors = torch.sigmoid(self.sigmoid_layer(vectors)) * F.relu(self.softmax_layer(vectors))
        sites = sites + scatter_add(vectors, indices1, 0) if self.residual else scatter_add(vectors, indices1, 0)

        return sites


class CGCNN(nn.Module):
    def __init__(self, hyperparams):
        super(CGCNN, self).__init__()
        self.supergraphs = hyperparams.use_supergraphs
        self.supercell_size = hyperparams.supercell_size
        self.in_site_len = (
            len(hyperparams.site_props) + self.supercell_size if self.supergraphs else len(hyperparams.site_props)
        )
        self.in_bond_len = len(hyperparams.bond_props)
        self.max_distance = hyperparams.max_distance
        self.bond_expansion = hyperparams.bond_expansion
        self.site_embedding_len = hyperparams.site_embedding
        self.bond_embedding_len = hyperparams.bond_embedding
        self.batch_norm = hyperparams.batch_norm
        self.h1 = hyperparams.h1
        self.h2 = hyperparams.h2
        self.output_len = len(hyperparams.target_props) * hyperparams.loss_output_len
        self.input_size = self.in_site_len + self.supercell_size

        self.site_embedding_layer = embed.EmbeddingLayer(self.in_site_len, self.site_embedding_len, self.batch_norm)
        self.bond_embedding_layer = embed.EmbeddingLayer(self.bond_expansion, self.bond_embedding_len, self.batch_norm)
        self.convlayer1 = ConvLayer(
            site_len=self.site_embedding_len,
            bond_len=self.bond_embedding_len,
            batch_norm=self.batch_norm,
        )
        self.convlayer2 = ConvLayer(
            site_len=self.site_embedding_len,
            bond_len=self.bond_embedding_len,
            batch_norm=self.batch_norm,
        )

        self.fc1 = nn.Linear(self.site_embedding_len, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.output_len)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.h1)
            self.batch_norm2 = nn.BatchNorm1d(self.h2)

    def forward(self, x, group=None):
        if self.supergraphs:
            (
                sites,
                bonds,
                _,
                indices1,
                indices2,
                _,
                _,
                graph_to_sites,
                _,
            ) = x
        else:
            sites, bonds, _, indices1, indices2, graph_to_sites, _ = x

        bonds = embed.gaussian_basis(
            bonds, self.max_distance, self.bond_expansion, self.max_distance / self.bond_expansion
        )

        sites = self.site_embedding_layer(sites)
        bonds = self.bond_embedding_layer(bonds)

        sites = self.convlayer1(sites, bonds, indices1, indices2)
        sites = self.convlayer2(sites, bonds, indices1, indices2)

        vector = scatter_mean(sites, graph_to_sites, 0)
        if self.batch_norm:
            vector = F.relu(self.batch_norm1(self.fc1(vector)))
            vector = F.relu(self.batch_norm2(self.fc2(vector)))
        else:
            vector = F.relu(self.fc1(vector))
            vector = F.relu(self.fc2(vector))

        out = self.fc3(vector)
        return out


def main(argv):
    # FIXME: creating the hyperparams object outside this file made the tests
    # fail because the arg_parser seemed to be shared.
    hyperparams = create_full_hyperparams(default_parameters, {}, argv)

    torch.manual_seed(hyperparams.seed)
    torch.cuda.manual_seed(hyperparams.seed)
    model = CGCNN(hyperparams)
    if hyperparams.mode == "train":
        trn.train_and_monitor(hyperparams, model)
    if hyperparams.mode == "evaluate":
        print(f"Running model CGNN on dataset {hyperparams.run_dataset}")
        trn.evaluate(model, hyperparams)
    elif hyperparams.mode == "inference":
        print(f"Running model CGCNN on dataset {hyperparams.run_dataset}")
        trn.inference(model, hyperparams)


if __name__ == "__main__":
    main(sys.argv[1:])
