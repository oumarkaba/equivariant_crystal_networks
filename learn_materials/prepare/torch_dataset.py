import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import random_split
import random
import wandb

from learn_materials import utils
from learn_materials.prepare.datamodel import load_material_file
from learn_materials.prepare.graphs import EuclideanGraph, SuperGraph
from learn_materials.prepare.chemistry import PERIODICTABLE, get_atomic_onehot
from learn_materials.models.embeddings import gaussian_basis
from learn_materials.equivariant.lattice import LATTICE_NAMES


def create_dataset(
    materials_data,
    descriptor_names,
    target_names,
    missing_entries=None,
    uncert=None,
    missing_uncert=None,
    save_file=None,
    save=True,
):
    print("reading descriptors")
    descriptors = []
    targets = []
    missed = 0
    total = len(materials_data)
    for i, material in enumerate(materials_data):
        descriptor = material.get_properties_values(
            descriptor_names,
            nan=missing_entries,
            uncert_nan=missing_uncert,
        )

        target = material.get_properties_values(
            target_names,
            uncert=False,
            nan=0,
            uncert_nan=missing_uncert,
        )

        if (not None in descriptor) and (not None in target):
            descriptors.append(descriptor)
            targets.append(target)
        else:
            missed += 1

    print(f"{missed}/{total} ignored because of missing properties")

    tensor_descriptors = torch.FloatTensor(descriptors)
    tensor_targets = torch.FloatTensor(targets)

    dataset = utils.DescriptorDataset(tensor_descriptors, tensor_targets, target_names)

    if save:
        torch.save(dataset, save_file)

    return dataset


def create_graph_dataset(
    material_file,
    graphs_file,
    site_descriptor_names,
    target_names,
    bond_angles,
    state_len=None,
    missing_entries=None,
    uncert=False,
    missing_uncert=None,
    save_file=None,
    save=True,
):
    materials = load_material_file(material_file)
    print("Materials file loaded")
    graph_data = utils.load_json(graphs_file)
    print("Graph file loaded")
    invalid_graphs = 0
    missing_targets = 0

    graphs_descriptors = {name: [] for name, _ in LATTICE_NAMES.items()}
    targets = {name: [] for name, _ in LATTICE_NAMES.items()}
    for i, material in enumerate(materials):
        if i % 100 == 0:
            print(f"Adding material {i} out of {len(materials)}")

        mp_id = material.get_mp_id()
        graph = graph_data.get(mp_id, None)
        if not graph:
            print(
                f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because graph was not found"
            )
            missing_targets += 1
            continue
        else:
            graph = EuclideanGraph(None, None, None, None, attributes_dict=graph)
        graph_descriptors, lattice_string = graph.get_graph_descriptors(site_descriptor_names, state_len, True)
        # material.add_magnetic_moment()
        target = material.get_properties_values(
            target_names,
            uncert=uncert,
            nan=missing_entries,
            uncert_nan=missing_uncert,
        )
        if not graph_descriptors:
            print(
                f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because graph is invalid or is disconnected"
            )
            invalid_graphs += 1
            continue
        if None in target:
            print(
                f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because some targets are not available"
            )
            continue
        graphs_descriptors[lattice_string].append(graph_descriptors)
        targets[lattice_string].append(target)

    tensor_targets = {name: torch.FloatTensor(target) for name, target in targets.items()}

    datasets = {
        name: utils.DescriptorDataset(graphs_descriptors[name], tensor_targets[name], target_names)
        for name in LATTICE_NAMES.keys()
    }
    print(f"Entries with invalid graphs : {invalid_graphs}")
    print(f"Entries with missing graphs : {missing_targets}")
    if save:
        torch.save(datasets, save_file)

    return datasets


def create_supergraph_dataset(
    material_file,
    graphs_data,
    site_descriptor_names,
    target_names,
    state_len=None,
    missing_entries=None,
    uncert=False,
    missing_uncert=None,
    save_file=None,
    save=True,
):
    materials = load_material_file(material_file)
    print("Loading graphs")
    invalid_graphs = 0
    missing_targets = 0

    graphs_descriptors = {name: [] for name, _ in LATTICE_NAMES.items()}
    targets = {name: [] for name, _ in LATTICE_NAMES.items()}
    for i, material in enumerate(materials):
        if i % 100 == 0:
            print(f"Adding material {i} out of {len(materials)}")

        is_hubbbard = "GGA+U" in material.tags
        mp_id = material.get_mp_id(source="perov5")
        # mp_id_int = int(mp_id.split("-")[1])
        mp_id_int = int(mp_id)
        # graph = graphs_data.get(mp_id, None)
        graph = graphs_data.get(str(mp_id), None)
        if not graph:
            print(
                f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because graph was not found"
            )
            continue
        graph = SuperGraph(None, None, None, None, attributes_dict=graph)
        if not graph.sites1:
            print(
                f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because graph is invalid or is disconnected"
            )
            invalid_graphs += 1
            continue
        graph_descriptors, lattice_string = graph.get_graph_descriptors(
            site_descriptor_names, state_len, True, hubbard=is_hubbbard
        )
        # material.add_magnetic_moment()
        if not graph_descriptors:
            print(
                f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because graph is invalid or is disconnected"
            )
            invalid_graphs += 1
            continue
        if target_names == ["Magmoms"]:
            if not graph.magmoms or None in graph.magmoms:
                print(
                    f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because some targets are not available"
                )
                missing_targets += 1
                continue
            target = torch.FloatTensor(graph.magmoms)
        else:
            target = material.get_properties_values(
                target_names,
                uncert=uncert,
                nan=missing_entries,
                uncert_nan=missing_uncert,
            )
            if None in target:
                print(
                    f"Could not include material with id {material.uid} ({material.get_mp_id()}) in dataset because some targets are not available"
                )
                missing_targets += 1
                continue
        graph_descriptors = tuple(list(graph_descriptors) + [mp_id_int])
        graphs_descriptors[lattice_string].append(graph_descriptors)
        targets[lattice_string].append(target)

    if not target_names == ["Magmoms"]:
        tensor_targets = {name: torch.FloatTensor(target) for name, target in targets.items()}
    else:
        tensor_targets = targets

    datasets = {
        name: utils.DescriptorDataset(graphs_descriptors[name], tensor_targets[name], target_names)
        for name in LATTICE_NAMES.keys()
    }
    print(f"Entries with invalid graphs : {invalid_graphs}")
    print(f"Entries with missing graphs : {missing_targets}")
    if save:
        torch.save(datasets, save_file)

    return datasets


def create_icsd_dataset(
    graphs_file,
    site_descriptor_names,
    bond_angles,
    state_len=None,
    save_file=None,
    save=True,
):
    print("Loading graphs...")
    graph_data = utils.load_json(graphs_file)
    print("Graph file loaded")

    graphs_descriptors = []
    icsd_list_location = []
    icsd_ids = {}
    invalid_ids = []
    for i, data in enumerate(graph_data.items()):
        if i % 100 == 0:
            print(f"Adding material {i} out of {len(graph_data)}")

        icsd_id, graph = data
        graph = EuclideanGraph(None, None, None, None, attributes_dict=graph)
        graph_descriptors = graph.get_graph_descriptors(site_descriptor_names, state_len, True)
        if not graph_descriptors:
            print(
                f"Could not include material with id {icsd_id} in dataset because graph is invalid or is disconnected"
            )
            invalid_ids.append(icsd_id)
            continue
        graphs_descriptors.append(graph_descriptors)
        icsd_list_location.append(i)
        icsd_ids[i] = icsd_id

    tensor_icsd_list_location = torch.LongTensor(icsd_list_location)

    dataset = utils.IcsdDataset(graphs_descriptors, tensor_icsd_list_location, icsd_ids)
    utils.save_json(invalid_ids, utils.DATA_PATH / "icsd" / "invalid_graph_ids.json")
    utils.save_json(icsd_ids, utils.DATA_PATH / "icsd" / "icsd_ids_dataset_map.json")

    if save:
        torch.save(dataset, save_file)

    return dataset


def merge_datasets(dataset_dict, keys, tensor_target):
    datasets = [dataset for key, dataset in dataset_dict.items() if key in keys]
    descriptors = [dataset.descriptors for dataset in datasets]
    targets = [dataset.targets for dataset in datasets]
    target_names = datasets[0].target_names

    merged_descriptors = [item for sublist in descriptors for item in sublist]
    if tensor_target:
        merged_targets = torch.cat(tuple(targets))
    else:
        merged_targets = [item for sublist in targets for item in sublist]
    dataset = utils.DescriptorDataset(merged_descriptors, merged_targets, target_names)

    return dataset


def combine_graph_data(graphs_data, no_targets=False):
    # FIXME: temporary fix before cleaning the data
    graphs_data = list(filter(lambda entry: entry is not None, graphs_data))
    if not list(graphs_data):
        return None, None
    graphs_descriptors, targets = zip(*graphs_data)
    (
        sites_descriptors,
        bonds_descriptors,
        state_descriptors,
        indices1,
        indices2,
    ) = zip(*graphs_descriptors)

    graph_to_sites = []
    for i, sites in enumerate(sites_descriptors):
        graph_to_sites += [i] * len(sites)

    graph_to_bonds = []
    for i, bonds in enumerate(bonds_descriptors):
        graph_to_bonds += [i] * len(bonds)

    batch_targets = torch.Tensor(targets) if no_targets else torch.cat(targets, 0)
    batch_sites = torch.cat(sites_descriptors, 0)
    batch_bonds = torch.cat(bonds_descriptors, 0)
    batch_states = torch.cat(state_descriptors)

    batch_indices1 = []
    batch_indices2 = []
    offset = 0

    for index1, index2 in zip(indices1, indices2):
        batch_indices1 += [i + offset for i in index1]
        batch_indices2 += [i + offset for i in index2]
        offset += max(index1) + 1

    batch_indices1 = torch.LongTensor(batch_indices1)
    batch_indices2 = torch.LongTensor(batch_indices2)
    graph_to_sites = torch.LongTensor(graph_to_sites)
    graph_to_bonds = torch.LongTensor(graph_to_bonds)

    return (
        (
            batch_sites,
            batch_bonds,
            batch_states,
            batch_indices1,
            batch_indices2,
            graph_to_sites,
            graph_to_bonds,
        ),
        batch_targets,
    )


def combine_supergraph_data(graphs_data, no_targets=False):
    # FIXME: temporary fix before cleaning the data
    graphs_data = list(filter(lambda entry: entry is not None, graphs_data))
    if not list(graphs_data):
        return None, None
    graphs_descriptors, targets = zip(*graphs_data)
    (
        sites_descriptors,
        bonds_descriptors,
        state_descriptors,
        indices1,
        indices2,
        indices_cells,
        indices_identity,
        mp_id,
    ) = zip(*graphs_descriptors)

    graph_to_sites = []
    for i, sites in enumerate(sites_descriptors):
        graph_to_sites += [i] * len(sites)

    graph_to_bonds = []
    for i, bonds in enumerate(bonds_descriptors):
        graph_to_bonds += [i] * len(bonds)

    batch_targets = torch.Tensor(targets) if no_targets else torch.cat(targets, 0)
    batch_sites = torch.cat(sites_descriptors, 0)
    batch_bonds = torch.cat(bonds_descriptors, 0)
    batch_states = torch.cat(state_descriptors)

    batch_indices1 = []
    batch_indices2 = []
    batch_indices_cells = []
    batch_indices_identity = []
    offset_nodes = 0
    offset_identity = 0
    offset_cells = 0

    for i, (index1, index2) in enumerate(zip(indices1, indices2)):
        batch_indices1 += [i + offset_nodes for i in index1]
        batch_indices2 += [i + offset_nodes for i in index2]
        offset_nodes += max(index1) + 1
        assert (max(index1) + 1) == sites_descriptors[i].shape[0]

    batch_indices1 += list(range(max(batch_indices1) + 1))
    batch_indices2 += list(range(max(batch_indices2) + 1))
    batch_bonds = torch.cat((batch_bonds, torch.zeros(max(batch_indices1) + 1, 1)))

    for index_cells in indices_cells:
        batch_indices_cells += index_cells

    for index_identity in indices_identity:
        batch_indices_identity += [i + offset_identity for i in index_identity]
        offset_identity += max(index_identity) + 1

    batch_indices1 = torch.LongTensor(batch_indices1)
    batch_indices2 = torch.LongTensor(batch_indices2)
    batch_indices_cells = torch.LongTensor(batch_indices_cells)
    batch_indices_identity = torch.LongTensor(batch_indices_identity)
    graph_to_sites = torch.LongTensor(graph_to_sites)
    graph_to_bonds = torch.LongTensor(graph_to_bonds)

    added_to_bonds = torch.index_select(graph_to_sites, 0, torch.arange(max(batch_indices1) + 1))
    graph_to_bonds = torch.cat((graph_to_bonds, added_to_bonds), 0)

    return (
        (
            batch_sites,
            batch_bonds,
            batch_states,
            batch_indices1,
            batch_indices2,
            batch_indices_cells,
            batch_indices_identity,
            graph_to_sites,
            graph_to_bonds,
        ),
        batch_targets,
    )


def split_dataset(dataset, train_proportion, validation_proportion=None, test_proportion=None):
    train_size = int(len(dataset) * train_proportion)
    validation_size = len(dataset) - train_size
    if validation_proportion:
        validation_size = int(len(dataset) * validation_proportion)
        test_size = len(dataset) - train_size - validation_size
        if test_proportion:
            test_size = int(len(dataset) * test_proportion)
        return random_split(dataset, (train_size, validation_size, test_size))
    return random_split(dataset, (train_size, validation_size))


def create_sample_dataset(datasets, size, target_names):
    sample_datasets = {}
    for structure, dataset in datasets.items():
        print(f"Creating sample dataset for {structure}")
        samples = list(zip(*random.choices(dataset, k=size)))
        sample_datasets[structure] = utils.DescriptorDataset(samples[0], torch.stack(samples[1]), target_names)

    return sample_datasets


def save_artifact(dataset_path):
    wandb.log_artifact(
        str(dataset_path),
        name=dataset_path.stem,
        type="dataset",
    )


def add_descriptors(dataset_path, descriptors, feature_conversion, save_path):
    new_graphs_descriptors = []
    dataset = torch.load(dataset_path)
    graphs_descriptors = dataset.descriptors
    # new_descriptors_values = []
    for i, graph_descriptors in enumerate(graphs_descriptors):
        if i % 100 == 0:
            print(f"Entry {i} out of {len(graphs_descriptors)}")
        (
            sites_descriptors,
            bonds_descriptors,
            state_descriptors,
            indices1,
            indices2,
        ) = graph_descriptors

        new_site_descriptors = torch.zeros(len(sites_descriptors), 70)
        for i, site in enumerate(sites_descriptors):
            atomic_number = (torch.nonzero(site, as_tuple=True)[0] + 1).item()
            new_descriptors = [
                PERIODICTABLE[descriptor_name][atomic_number]
                for i, descriptor_name in enumerate(descriptors)
                if descriptor_name != "atomic number onehot"
            ]
            new_descriptors = [
                new_descriptor if not np.isnan(new_descriptor) else -1.0 for new_descriptor in new_descriptors
            ]
            new_descriptors = [
                feature_conversion[i](new_descriptor) for i, new_descriptor in enumerate(new_descriptors)
            ]
            new_descriptors = [item for sublist in new_descriptors for item in sublist]
            # new_descriptors_values.append(new_descriptors)
            new_site_descriptors[i] = torch.tensor(new_descriptors)
        new_site_descriptors = torch.cat([sites_descriptors, new_site_descriptors], dim=1)

        graph_descriptors = (
            new_site_descriptors,
            bonds_descriptors,
            state_descriptors,
            indices1,
            indices2,
        )
        new_graphs_descriptors.append(graph_descriptors)

    # new_descriptors_values = np.array(new_descriptors_values)
    # df_values = pd.DataFrame(new_descriptors_values, columns=descriptors)
    # df_values.to_pickle(save_path)
    dataset = utils.DescriptorDataset(new_graphs_descriptors, dataset.targets, dataset.target_names)

    if save_path:
        torch.save(dataset, save_path)
    return dataset


def main(argv):
    description = """
    This script is used to create torch datasets from materials files for use in deep
    learning models. There are two modes:
    graph to create GraphDatasets.
    descriptors to create standard torch Datasets.
    """
    datapath = utils.DATA_PATH / "materials_project"
    default_args = {
        "materials_file": [
            str(
                datapath / "parsed" / "subsample_tr" / "test_materials.json"
            ),
            # str(
            #     datapath / "parsed" / "parsed_matproj_v2021_05_nonduplicate_small_3d_valid_graphs_valid.json"
            # ),
            # str(
            #     datapath / "parsed" / "parsed_matproj_v2021_05_nonduplicate_small_3d_valid_graphs_test.json"
            # ),
            # str(datapath / "parsed" / "parsed_matproj_v2021_05_nonduplicate_small_3d_valid_graphs_sample.json"),
        ],
        "graphs_file": str(
            datapath / "graphs" / "test_graphs_all_cut=5.0_tol=0.5.json"
        ),
        "save": True,
        "save_file": [
            datapath
            / "torch_datasets"
            / "test_graphs_all_cut=5.0_tol=0.5.pt",
            # datapath
            # / "torch_datasets"
            # / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_valid.pt",
            # datapath
            # / "torch_datasets"
            # / "matproj_supergraphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_hubbard_test.pt",
            # datapath
            # / "torch_datasets"
            # / "matproj_graphs_v2021_05_2x2x2_isayev_cut=6_tol=05_v2021_05_nonduplicate_small_3d_valid_graphs_sample.pt",
        ],
        "angles": False,
        "atomic_descriptors": ["atomic number onehot"],
        "formula_descriptors": ["Average atomic number"],
        "targets": [
            "heat_all",
            "heat_ref",
            "dir_gap",
            "ind_gap",
        ],
        # "targets": [
        #     "Magmoms",
        # ],
        "state_len": 2,
        "mode": "graph",
    }
    help_dict = {
        "materials_file": "input materials files for dataset",
        "graphs_file": "graph objects file if mode is graph",
        "save": "whether the resulting dataset should be saved",
        "save_file": "output file for the torch dataset",
        "angles": "compute graph with angles",
        "state_len": "lenght of the state attribute for graphs",
        "atomic_descriptors": "list of descriptors to use for atomic sites in graphs",
        "formula_descriptors": "list of formula descriptors to use in descriptors mode",
        "targets": "list of targets",
        "mode": "type of mode",
    }
    args = utils.read_args(
        default_args=default_args,
        help_dict=help_dict,
        description=description,
        argv=argv,
    )
    out_list = []
    for i in range(len(args.materials_file)):
        if args.mode in ["graph", "graphs"]:
            out = create_graph_dataset(
                args.materials_file[i],
                args.graphs_file,
                args.atomic_descriptors,
                args.targets,
                args.angles,
                state_len=args.state_len,
                save_file=args.save_file[i],
                save=args.save,
            )
            if args.save:
                continue
        elif args.mode in ["supergraph", "supergraphs"]:
            graph_data = utils.load_json(args.graphs_file)
            print("Graph file loaded")
            out = create_supergraph_dataset(
                args.materials_file[i],
                graph_data,
                args.atomic_descriptors,
                args.targets,
                state_len=args.state_len,
                save_file=args.save_file[i],
                save=args.save,
            )
            if args.save:
                continue
        else:
            materials_data = load_material_file(args.materials_file[i])
            out = create_dataset(
                materials_data,
                args.formula_descriptors,
                args.targets,
                save_file=args.save_file[i],
                save=args.save,
            )
            if args.save:
                continue
        out_list.append(out)
    return out_list


if __name__ == "__main__":
    # ICSD_PATH = utils.DATA_PATH / "icsd" / "icsd_graphs.json"
    # SAVE_FILE = utils.DATA_PATH / "icsd" / "icsd_torchdataset.pt"
    # create_icsd_dataset(
    #     ICSD_PATH, ["atomic number onehot"], False, 2, save_file=SAVE_FILE
    # )

    # original_dataset_path = (
    #     utils.DATA_PATH / "materials_project" / "benjamin_torch_datasets"
    # )
    # new_dataset_path = utils.DATA_PATH / "materials_project" / "benjamin_torch_datasets"

    # datasets = [
    #     "matproj_e_hull>01_v2020_06_voronoi_test.pt",
    #     "matproj_e_hull>005_v2020_06_voronoi_test.pt",
    #     "matproj_e_hull>005_v2020_06_voronoi_valid.pt",
    #     "matproj_e_hull>005_v2020_06_voronoi_train.pt",
    # ]
    # new_datasets = [
    #     "matproj_e_hull>01_v2020_06_voronoi_descriptors_test.pt",
    #     "matproj_e_hull>005_v2020_06_voronoi_descriptors_test.pt",
    #     "matproj_e_hull>005_v2020_06_voronoi_descriptors_valid.pt",
    #     "matproj_e_hull>005_v2020_06_voronoi_descriptors_train.pt",
    # ]

    # feature_conversion = [
    #     lambda valence: get_atomic_onehot(int(valence) + 1, 15),
    #     lambda valence: get_atomic_onehot(int(valence) + 1, 11),
    #     lambda valence: get_atomic_onehot(int(valence), 16),
    #     lambda magnetic_moment: get_atomic_onehot(int(magnetic_moment), 8),
    #     lambda electron_affinity: gaussian_basis(
    #         electron_affinity, 350, 10, 25.0, 0, False
    #     ).tolist(),
    #     lambda magp_atomic_volume: gaussian_basis(
    #         magp_atomic_volume ** (0.33), 32, 10, 4.5, 1.7, False
    #     ).tolist(),
    # ]

    # for dataset, new_dataset in zip(datasets, new_datasets):
    #     add_descriptors(
    #         original_dataset_path / dataset,
    #         [
    #             "valence_f",
    #             "valence_d",
    #             "imat_valence",
    #             "magnetic_moment",
    #             "electron_affinity",
    #             "magp_atomic_volume",
    #         ],
    #         feature_conversion,
    #         new_dataset_path / new_dataset,
    #     )
    # targets = [
    #     "Formation energy per atom",
    #     "Final energy per atom",
    #     "Band gap",
    #     "Magnetic moment per unit cell",
    #     "Magnetic moment per unit cell per atom",
    #     "Energy above hull",
    # ]
    # train_dataset_path = (
    #     utils.DATA_PATH
    #     / "materials_project"
    #     / "torch_datasets"
    #     / "clean_graphs"
    #     / "matproj_graphs_clean_v2020_06_voronoi_train_nonoriginal.pt"
    # )
    # sample_dataset_path = (
    #     utils.DATA_PATH
    #     / "materials_project"
    #     / "torch_datasets"
    #     / "clean_graphs"
    #     / "matproj_graphs_clean_v2020_06_voronoi_sample_nonoriginal.pt"
    # )
    # train_dataset = torch.load(train_dataset_path)
    # sample_dataset = create_sample_dataset(train_dataset, 128, targets)
    # torch.save(sample_dataset, sample_dataset_path)

    main(sys.argv[1:])
