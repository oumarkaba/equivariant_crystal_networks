import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.glob import Set2Set

import learn_materials.models.embeddings as embed
import learn_materials.training as trn
from learn_materials.models.hyperparams import create_full_hyperparams


default_parameters = {
    "model": "MEGNet",
    "graphs": True,
    "megnet_h1": 64,
    "megnet_h2": 64,
    "premegnet_h1": 64,
    "premegnet_h2": 32,
    "postmegnet_h1": 32,
    "postmegnet_h2": 16,
    "lr": 1e-3,
    "site_props": list(range(103)),
    "bond_props": ["distance"],
    "state_len": 2,
    # "target_props": ["Formation energy per atom"],
    "bond_expansion": 100,
    "state_embedding": 36,
    "bond_embedding": 36,
    "site_embedding": 36,
    "run_model": True,
    "use_supergraphs": False,
    "wandb_project": "equivariant_crystals_megnet",
}

"""
Modifications with respect to the original architecture:
*
"""


class BondUpdate(nn.Module):
    def __init__(self, site_len, bond_len, state_len, h1, h2):
        super(BondUpdate, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len

        self.fc1 = nn.Linear(2 * self.site_len + self.bond_len + self.state_len, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, self.bond_len)

    def forward(self, sites, bonds, states, indices1, indices2, graph_to_bonds):
        sites1 = torch.index_select(sites, 0, indices1)
        sites2 = torch.index_select(sites, 0, indices2)
        states = torch.index_select(states, 0, graph_to_bonds)

        vectors = torch.cat((sites1, sites2, bonds, states), 1)

        vectors = F.relu(self.fc1(vectors))
        vectors = F.relu(self.fc2(vectors))
        bonds = F.relu(self.fc3(vectors))

        return bonds


class SiteUpdate(nn.Module):
    def __init__(self, site_len, bond_len, state_len, h1, h2):
        super(SiteUpdate, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len

        self.fc1 = nn.Linear(self.site_len + self.bond_len + self.state_len, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, self.site_len)

    def forward(self, sites, bonds, states, indices1, graph_to_sites):
        bonds_pool = self.bonds_to_site(bonds, indices1)
        states = torch.index_select(states, 0, graph_to_sites)

        vectors = torch.cat((bonds_pool, sites, states), 1)

        vectors = F.relu(self.fc1(vectors))
        vectors = F.relu(self.fc2(vectors))
        sites = F.relu(self.fc3(vectors))

        return sites

    def bonds_to_site(self, bonds, indices1):
        return scatter_mean(bonds, indices1, 0)


class StateUpdate(nn.Module):
    def __init__(self, site_len, bond_len, state_len, h1, h2):
        super(StateUpdate, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len

        self.fc1 = nn.Linear(self.site_len + self.bond_len + self.state_len, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, self.state_len)

    def forward(self, sites, bonds, states, graph_to_sites, graph_to_bonds):
        bonds_pool = self.bonds_to_state(bonds, graph_to_bonds)
        sites_pool = self.sites_to_state(sites, graph_to_sites)

        vectors = torch.cat((bonds_pool, sites_pool, states), 1)

        vectors = F.relu(self.fc1(vectors))
        vectors = F.relu(self.fc2(vectors))
        states = F.relu(self.fc3(vectors))

        return states

    def bonds_to_state(self, bonds, graph_to_bonds):
        return scatter_mean(bonds, graph_to_bonds, 0)

    def sites_to_state(self, sites, graph_to_sites):
        return scatter_mean(sites, graph_to_sites, 0)


class MEGNetBlock(nn.Module):
    def __init__(
        self,
        site_len,
        bond_len,
        state_len,
        megnet_h1,
        megnet_h2,
        premegnet_h1,
        premegnet_h2,
        first_block,
    ):
        super(MEGNetBlock, self).__init__()
        self.site_len = site_len
        self.bond_len = bond_len
        self.state_len = state_len
        self.megnet_h1 = megnet_h1
        self.megnet_h2 = megnet_h2
        self.premegnet_h1 = premegnet_h1
        self.premegnet_h2 = premegnet_h2
        self.first_block = first_block

        self.bonds_fc1 = nn.Linear(self.bond_len, self.premegnet_h1)
        self.bonds_fc2 = nn.Linear(self.premegnet_h1, self.premegnet_h2)
        self.sites_fc1 = nn.Linear(self.site_len, self.premegnet_h1)
        self.sites_fc2 = nn.Linear(self.premegnet_h1, self.premegnet_h2)
        self.states_fc1 = nn.Linear(self.state_len, self.premegnet_h1)
        self.states_fc2 = nn.Linear(self.premegnet_h1, self.premegnet_h2)

        self.bondupdate = BondUpdate(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
        )
        self.siteupdate = SiteUpdate(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
        )
        self.stateupdate = StateUpdate(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
        )

    def forward(self, sites, bonds, states, indices1, indices2, graph_to_sites, graph_to_bonds):
        initial_sites, initial_bonds, initial_states = sites, bonds, states
        sites, bonds, states = self.fc_layers(sites, bonds, states)
        if self.first_block:
            initial_sites, initial_bonds, initial_states = sites, bonds, states

        bonds = self.bondupdate(sites, bonds, states, indices1, indices2, graph_to_bonds)
        sites = self.siteupdate(sites, bonds, states, indices1, graph_to_sites)
        states = self.stateupdate(sites, bonds, states, graph_to_sites, graph_to_bonds)

        sites += initial_sites
        bonds += initial_bonds
        states += initial_states

        return sites, bonds, states

    def fc_layers(self, sites, bonds, states):
        bonds = F.relu(self.bonds_fc1(bonds))
        bonds = F.relu(self.bonds_fc2(bonds))
        sites = F.relu(self.sites_fc1(sites))
        sites = F.relu(self.sites_fc2(sites))
        states = F.relu(self.states_fc1(states))
        states = F.relu(self.states_fc2(states))

        return sites, bonds, states


class MEGNet(nn.Module):
    def __init__(self, hyperparams):
        super(MEGNet, self).__init__()
        self.supergraphs = hyperparams.use_supergraphs
        self.supercell_size = hyperparams.supercell_size
        self.in_site_len = (
            len(hyperparams.site_props) + self.supercell_size if self.supergraphs else len(hyperparams.site_props)
        )
        self.in_bond_len = len(hyperparams.bond_props)
        self.max_distance = hyperparams.max_distance
        self.in_state_len = hyperparams.state_len
        self.bond_expansion = hyperparams.bond_expansion
        self.site_embedding_len = hyperparams.site_embedding
        self.bond_embedding_len = hyperparams.bond_embedding
        self.state_embedding_len = hyperparams.state_embedding
        self.megnet_h1 = hyperparams.megnet_h1
        self.megnet_h2 = hyperparams.megnet_h2
        self.premegnet_h1 = hyperparams.premegnet_h1
        self.premegnet_h2 = hyperparams.premegnet_h2
        self.postmegnet_h1 = hyperparams.postmegnet_h1
        self.postmegnet_h2 = hyperparams.postmegnet_h2
        self.output_len = len(hyperparams.target_props) * hyperparams.loss_output_len

        self.site_embedding_layer = embed.EmbeddingLayer(self.in_site_len, self.site_embedding_len)
        self.bond_embedding_layer = embed.EmbeddingLayer(self.bond_expansion, self.bond_embedding_len)
        self.state_embedding_layer = embed.EmbeddingLayer(self.in_state_len, self.state_embedding_len)

        self.megnetblock1 = MEGNetBlock(
            self.site_embedding_len,
            self.bond_embedding_len,
            self.state_embedding_len,
            self.megnet_h1,
            self.megnet_h2,
            self.premegnet_h1,
            self.premegnet_h2,
            True,
        )
        self.megnetblock2 = MEGNetBlock(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
            self.premegnet_h1,
            self.premegnet_h2,
            False,
        )
        self.megnetblock3 = MEGNetBlock(
            self.premegnet_h2,
            self.premegnet_h2,
            self.premegnet_h2,
            self.megnet_h1,
            self.megnet_h2,
            self.premegnet_h1,
            self.premegnet_h2,
            False,
        )
        # FIXME: Implementation du Set2set a vérifier
        self.sites_set2set = Set2Set(self.premegnet_h2, 3)
        self.bonds_set2set = Set2Set(self.premegnet_h2, 3)
        self.fc1 = nn.Linear(self.premegnet_h2 * 5, self.postmegnet_h1)
        self.fc2 = nn.Linear(self.postmegnet_h1, self.postmegnet_h2)
        self.fc3 = nn.Linear(self.postmegnet_h2, self.output_len)

    def forward(self, x):
        if self.supergraphs:
            (
                sites,
                bonds,
                states,
                indices1,
                indices2,
                _,
                _,
                graph_to_sites,
                graph_to_bonds,
            ) = x
        else:
            sites, bonds, states, indices1, indices2, graph_to_sites, graph_to_bonds = x

        bonds = embed.gaussian_basis(bonds, self.max_distance, self.bond_expansion, 0.5)

        sites = self.site_embedding_layer(sites)
        bonds = self.bond_embedding_layer(bonds)
        states = self.state_embedding_layer(states)

        sites, bonds, states = self.megnetblock1(
            sites, bonds, states, indices1, indices2, graph_to_sites, graph_to_bonds
        )
        sites, bonds, states = self.megnetblock2(
            sites, bonds, states, indices1, indices2, graph_to_sites, graph_to_bonds
        )
        sites, bonds, states = self.megnetblock2(
            sites, bonds, states, indices1, indices2, graph_to_sites, graph_to_bonds
        )

        sites = self.sites_set2set(sites, graph_to_sites)
        bonds = self.bonds_set2set(bonds, graph_to_bonds)
        vector = torch.cat((sites, bonds, states), 1)

        vector = F.relu(self.fc1(vector))
        vector = F.relu(self.fc2(vector))
        out = self.fc3(vector)
        return out


def main(argv):
    hyperparams = create_full_hyperparams(default_parameters, {}, argv)

    torch.manual_seed(hyperparams.seed)
    torch.cuda.manual_seed(hyperparams.seed)
    model = MEGNet(hyperparams)
    if hyperparams.mode == "train":
        trn.train_and_monitor(hyperparams, model)
    if hyperparams.mode == "evaluate":
        print(f"Running model MEGNET on dataset {hyperparams.run_dataset}")
        trn.evaluate(model, hyperparams)
    elif hyperparams.mode == "inference":
        print(f"Running model MEGNET on dataset {hyperparams.run_dataset}")
        trn.inference(model, hyperparams)


if __name__ == "__main__":
    main(sys.argv[1:])
