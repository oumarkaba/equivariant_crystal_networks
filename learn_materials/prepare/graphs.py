import random

random.seed(0)

import numpy as np
from pymatgen import Structure, PeriodicSite
import pymatgen.analysis.local_env as env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.transformations.site_transformations import TranslateSitesTransformation
from joblib import Parallel, delayed
import sys
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# import pygraphviz as pgv
import networkx as nx
from itertools import islice
import pymatgen.analysis.dimensionality as dim

from learn_materials import utils
from learn_materials.prepare.chemistry import PERIODICTABLE, get_atomic_onehot, get_group_period
from learn_materials.equivariant.lattice import get_bravais_lattice
from learn_materials.prepare.pymatgen_classes import CustomStructure, CustomVoronoiNN, CustomIsayevNN, CustomCrystallNN
from learn_materials.prepare.plot import annotate3D


class EuclideanGraph:
    def __init__(
        self,
        structure_type,
        structure_input,
        mp_id,
        strategy,
        cutoff=None,
        bond_tol=0.25,
        attributes_dict=None,
        status="OK",
    ):
        if attributes_dict:
            self.__dict__.update(attributes_dict)
        else:
            self.structure_type = structure_type
            self.structure_input = structure_input
            self.mp_id = mp_id if mp_id else str(self.cif_path).split("/")[-1].split(".")[0]
            self.strategy = strategy
            self.cutoff = cutoff
            self.bond_tol = bond_tol
            self.status = status
            (
                self.site_atomic_numbers,
                self.site_positions,
                self.bond_distances,
                self.sites1,
                self.sites2,
                self.magmoms,
                self.lattice_string,
                self.status,
            ) = graph_from_cif(
                self.structure_type,
                self.structure_input,
                self.strategy,
                self.cutoff,
                self.bond_tol,
            )

    def compute_adjacency_matrix(self):
        adjacency_matrix = [
            [0 for i in range(len(self.site_atomic_numbers))] for j in range(len(self.site_atomic_numbers))
        ]

        for site1, site2 in zip(self.sites1, self.sites2):
            adjacency_matrix[site1][site2] += 1

        return np.array(adjacency_matrix)

    def compute_site_descriptors(self, site_descriptor_names):
        if not self.site_atomic_numbers:
            return (None, None)
        site_descriptors = []
        for i, site in enumerate(self.site_atomic_numbers):
            site_descriptors.append(
                [
                    PERIODICTABLE[descriptor_name][site]
                    for descriptor_name in site_descriptor_names
                    if descriptor_name != "atomic number onehot"
                ]
            )
            if "atomic number onehot" in site_descriptor_names:
                site_descriptors[i].extend(get_atomic_onehot(site, PERIODICTABLE.max_Z))
        return site_descriptors

    def is_connected(self):
        indices_sites = np.arange(len(self.site_atomic_numbers))
        sites_intersection = np.intersect1d(indices_sites, np.array(self.sites1))
        all_sites_connected = sites_intersection.shape[0] == len(self.site_atomic_numbers)
        G = nx.DiGraph(list(zip(self.sites1, self.sites2)))
        return all_sites_connected and nx.is_strongly_connected(G)

    def is_undirected(self):
        adjacency_matrix = self.compute_adjacency_matrix()
        return (adjacency_matrix == adjacency_matrix.T).all()

    def get_graph_descriptors(
        self,
        site_descriptors_names=None,
        state_len=0,
        none_if_disconnected=True,
    ):
        if none_if_disconnected and not self.is_connected():
            return (None, None)
        site_descriptors = self.compute_site_descriptors(site_descriptors_names)
        state_tensor = torch.zeros(1, state_len)
        bond_descriptors = [[distance] for distance in self.bond_distances]
        return (
            (
                (
                    torch.FloatTensor(site_descriptors),
                    torch.FloatTensor(bond_descriptors),
                    state_tensor,
                    self.sites1,
                    self.sites2,
                ),
                self.lattice_string,
            )
            if site_descriptors
            else (None, None)
        )

    def print_to_R(self):
        print(f"sites1 : {self.sites1}\n sites2 : {self.sites2}")

    def to_dict(self):
        return vars(self)


class SuperGraph(EuclideanGraph):
    def __init__(
        self,
        structure_type,
        structure_input,
        mp_id,
        strategy,
        cutoff=None,
        to_primitive=False,
        bond_tol=0.25,
        attributes_dict=None,
        cell_mapping="index",
        status="OK",
    ):
        if attributes_dict:
            self.__dict__.update(attributes_dict)
        else:
            self.structure_type = structure_type
            self.structure_input = structure_input
            self.mp_id = mp_id if mp_id else str(self.cif_path).split("/")[-1].split(".")[0]
            self.cell_mapping = cell_mapping
            self.strategy = strategy
            self.cutoff = cutoff
            self.bond_tol = bond_tol
            self.status = status
            self.to_primitive = to_primitive
            (
                self.site_atomic_numbers,
                self.site_positions,
                self.bond_distances,
                self.sites1,
                self.sites2,
                self.cell_indices,
                self.cell_positions,
                self.identity,
                self.magmoms,
                self.lattice_string,
                self.status,
            ) = supergraph_from_cif(
                self.structure_type,
                self.structure_input,
                self.strategy,
                self.cutoff,
                self.bond_tol,
                self.to_primitive,
            )

    def compute_site_descriptors(self, site_descriptor_names, hubbard):
        if not self.site_atomic_numbers:
            return None
        site_descriptors = []
        for i, atomic_number in enumerate(self.site_atomic_numbers):
            if self.cell_mapping == "index":
                cell = get_atomic_onehot(self.cell_indices[i], max(self.cell_indices) + 1)
            else:
                cell = self.cell_positions[i]
            site_descriptors.append(
                [
                    PERIODICTABLE[descriptor_name][atomic_number]
                    for descriptor_name in site_descriptor_names
                    if (descriptor_name != "atomic number onehot")
                    and (descriptor_name != "group period")
                    and (descriptor_name != "hubbard")
                ]
            )
            if "atomic number onehot" in site_descriptor_names:
                site_descriptors[i].extend(get_atomic_onehot(atomic_number, PERIODICTABLE.max_Z))
            elif "group period" in site_descriptor_names:
                site_descriptors[i].extend(get_group_period(atomic_number))
            if "hubbard" in site_descriptor_names:
                site_descriptors[i].append(1 if hubbard else 0)
            site_descriptors[i].extend(cell)
        return site_descriptors

    def get_graph_descriptors(
        self,
        site_descriptors_names=None,
        state_len=0,
        none_if_disconnected=True,
        hubbard=False,
    ):
        if none_if_disconnected and not self.is_connected():
            return (None, None)
        site_descriptors = self.compute_site_descriptors(site_descriptors_names, hubbard)
        state_tensor = torch.zeros(1, state_len)
        bond_descriptors = [[distance] for distance in self.bond_distances]
        return (
            (
                [
                    torch.FloatTensor(site_descriptors),
                    torch.FloatTensor(bond_descriptors),
                    state_tensor,
                    self.sites1,
                    self.sites2,
                    self.cell_indices,
                    self.identity,
                ],
                self.lattice_string,
            )
            if site_descriptors
            else (None, None)
        )

    def plot(self):
        x_pos, y_pos, z_pos = list(zip(*self.site_positions))
        edges = list(zip(self.sites1, self.sites2))
        segments = [(self.site_positions[s], self.site_positions[t]) for s, t in edges]

        # create figure
        fig = plt.figure(dpi=60)
        ax = fig.gca(projection="3d")
        ax.set_axis_off()

        # plot vertices
        ax.scatter(x_pos, y_pos, z_pos, marker="o", c=self.cell_indices, s=500, cmap="Set1")
        # plot edges
        edge_col = Line3DCollection(segments, lw=0.2)
        ax.add_collection3d(edge_col)
        # add vertices annotation.
        for j, xyz_ in enumerate(self.site_positions):
            annotate3D(
                ax, s=str(j), xyz=xyz_, fontsize=10, xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom"
            )
        plt.show()


def get_dimensionality(strategy, structure_input, structure_type, cutoff, bond_tol):
    try:
        material_structure = Structure.from_str(structure_input, structure_type)
        print("Parsed CIF")
    except ValueError:
        print("CIF could not be parsed")

    if strategy == "isayev":
        neighbor_strategy = env.IsayevNN(tol=bond_tol, cutoff=cutoff)
    if strategy == "voronoi":
        neighbor_strategy = env.VoronoiNN(cutoff=cutoff)
    if strategy == "crystal":
        neighbor_strategy = env.CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
    elif strategy == "all":
        neighbor_strategy = env.MinimumDistanceNN(cutoff=cutoff, get_all_sites=True)

    bonded_structure = neighbor_strategy.get_bonded_structure(material_structure)
    larsen_dim = dim.get_dimensionality_larsen(bonded_structure)
    gorai_dim = dim.get_dimensionality_gorai(material_structure)

    return larsen_dim, gorai_dim


def graph_from_cif(structure_type, structure_input, strategy, cutoff, bond_tol, to_primitive=False):
    if structure_type == "dir":
        material_structure = Structure.from_file(structure_input)
    else:
        try:
            material_structure = Structure.from_str(structure_input, structure_type)
            print("Parsed CIF")
        except ValueError:
            print("CIF could not be parsed")
            status = "Invalid input"
            return (None,) * 7 + (status,)

    material_structure.__class__ = CustomStructure
    _, lattice_string = get_bravais_lattice(material_structure)

    sites_atomic_numbers = []
    sites_positions = []
    magmoms = [] if material_structure.site_properties else None
    try:
        for site in material_structure:
            # for numerical stability of voronoi algorithm
            site.frac_coords = np.round(site.frac_coords, 5)

            site_element = PERIODICTABLE.get_attribute(
                "atomic_number", site.species.elements[0].as_dict()["element"], "symbol"
            )
            assert len(site.species.elements) == 1
            sites_atomic_numbers.append(site_element)
            sites_positions.append(site.coords.tolist())
            if magmoms is not None:
                magmoms.append(site.properties["magmom"])
        if strategy == "isayev":
            neighbor_strategy = CustomIsayevNN(tol=bond_tol, cutoff=cutoff)
        if strategy == "voronoi":
            neighbor_strategy = CustomVoronoiNN(cutoff=cutoff)
        # if strategy == "crystal":
        #     neighbor_strategy = env.CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        elif strategy == "all":
            neighbor_strategy = env.MinimumDistanceNN(cutoff=cutoff, get_all_sites=True)
        neighbor_strategy.to_primitive = to_primitive
    except IndexError:
        print("Contains elements that do not exist")
        status = "Invalid elements"
        return (None,) * 7 + (status,)
    except AssertionError:
        print("More than one element associated with one site")
        status = "More than one element per site"
        return (None,) * 7 + (status,)
    try:
        sites_neighbors = neighbor_strategy.get_all_nn_info(material_structure)
    except ValueError:
        print("No Voronoi neighbours found for some sites")
        status = "No Voronoi neighbours"
        return (None,) * 7 + (status,)
    except RuntimeError:
        print("Pathological structure, graph for this material could not be computed")
        status = "Pathological structure"
        return (None,) * 7 + (status,)
    neighbors_coords = []
    bonds = []
    sites1 = []
    sites2 = []
    for i, site in enumerate(sites_neighbors):
        neighbors_coords.append([])
        for neighbor in site:
            neighbors_coords[i].append(neighbor["site"].coords)
            bonds.append(
                super(PeriodicSite, material_structure[i]).distance(neighbor["site"]),
            )
            sites1.append(i)
            sites2.append(neighbor["site_index"])

    if not material_structure.site_properties:
        status = "No magmoms"
    else:
        status = "OK"
    return sites_atomic_numbers, sites_positions, bonds, sites1, sites2, magmoms, lattice_string, status


def supergraph_from_cif(structure_type, structure_input, strategy, cutoff, bond_tol, to_primitive):
    if structure_type == "dir":
        material_structure = Structure.from_file(structure_input)
    else:
        try:
            material_structure = Structure.from_str(structure_input, structure_type)
            print("Parsed input")
        except ValueError:
            print("Input could not be parsed")
            status = "Invalid input"
            return (None,) * 10 + (status,)

    material_structure.__class__ = CustomStructure
    bravais_lattice, lattice_string = get_bravais_lattice(material_structure)

    analyzer = SpacegroupAnalyzer(material_structure)
    primitive_structure = analyzer.find_primitive()
    primitive_structure.__class__ = CustomStructure
    identity = sum(
        [[i] * bravais_lattice.cluster.size for i in range(primitive_structure.num_sites)],
        [],
    )

    structure_matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)
    supercell, translation, mapping = structure_matcher.get_transformation(material_structure, primitive_structure)

    is_primitive = len(material_structure) == len(primitive_structure)
    if is_primitive:
        material_structure.make_supercell(bravais_lattice.cluster.supercell, to_unit_cell=False)
    else:
        print(f"CELL IS NOT PRIMITIVE")
        try:
            supercell_inv = bravais_lattice.cluster.supercell @ np.linalg.inv(supercell)
            material_structure.make_supercell(supercell_inv, False)
            primitive_structure.make_supercell(bravais_lattice.cluster.supercell, False)
            _, _, mapping = structure_matcher.get_transformation(material_structure, primitive_structure)
            if material_structure.site_properties:
                for i, index in enumerate(mapping):
                    primitive_structure[index].properties = material_structure[i].properties
            material_structure = CustomStructure.from_sites(primitive_structure.sites)
        except (np.linalg.LinAlgError, AssertionError) as e:
            print("Singular supercell")
            status = "Singular supercell"
            return (None,) * 10 + (status,)

    sites_atomic_numbers = []
    cell_positions = []
    site_positions = []
    magmoms = [] if material_structure.site_properties else None
    mapping = bravais_lattice.cluster.map_cluster()
    cell_indices = []
    try:
        for i, site in enumerate(material_structure):
            # for numerical stability of voronoi algorithm
            site.frac_coords = np.round(site.frac_coords, 5)

            site_element = PERIODICTABLE.get_attribute(
                "atomic_number", site.species.elements[0].as_dict()["element"], "symbol"
            )
            assert len(site.species.elements) == 1
            sites_atomic_numbers.append(site_element)
            site_positions.append(site.coords.tolist())

            cell = site.cell @ bravais_lattice.cluster.supercell @ bravais_lattice.cluster.translation_generators
            cell_positions.append(cell.tolist())
            cell_index = mapping(cell)
            cell_indices.append(cell_index)

            if magmoms is not None:
                magmoms.append(site.properties["magmom"])

        if strategy == "isayev":
            neighbor_strategy = CustomIsayevNN(tol=bond_tol, cutoff=cutoff)
        if strategy == "voronoi":
            neighbor_strategy = CustomVoronoiNN(cutoff=cutoff)
        if strategy == "crystal":
            neighbor_strategy = CustomCrystallNN()
        elif strategy == "all":
            neighbor_strategy = env.MinimumDistanceNN(cutoff=cutoff, get_all_sites=True)
        neighbor_strategy.to_primitive = to_primitive
    except IndexError:
        print("Contains elements that do not exist")
        status = "Invalid elements"
        return (None,) * 10 + (status,)
    except AssertionError:
        print("More than one element associated with one site")
        status = "More than one element per site"
        return (None,) * 10 + (status,)
    try:
        sites_neighbors = neighbor_strategy.get_all_nn_info(material_structure)
    except ValueError:
        print("No Voronoi neighbours found for some sites")
        status = "No Voronoi neighbours"
        return (None,) * 10 + (status,)
    except RuntimeError:
        print("Pathological structure, graph for this material could not be computed")
        status = "Pathological structure"
        return (None,) * 10 + (status,)
    neighbors_coords = []
    bonds = []
    sites1 = []
    sites2 = []
    for i, site in enumerate(sites_neighbors):
        neighbors_coords.append([])
        for neighbor in site:
            neighbors_coords[i].append(neighbor["site"].coords)
            bonds.append(
                super(PeriodicSite, material_structure[i]).distance(neighbor["site"]),
            )
            sites1.append(i)
            sites2.append(neighbor["site_index"])

    status = ""
    if None in cell_indices:
        cell_indices = None
        status += "Could not find cells"
    if not is_primitive:
        status += "Original cell not primitive"
    if not material_structure.site_properties:
        status += "No magmoms"
    if not (None in cell_indices) and is_primitive and material_structure.site_properties:
        status = "OK"
    return (
        sites_atomic_numbers,
        site_positions,
        bonds,
        sites1,
        sites2,
        cell_indices,
        cell_positions,
        identity,
        magmoms,
        lattice_string,
        status,
    )


def create_graphs(structures, structure_type, save_path, strategy, cutoff, bond_tol, parallel, save):
    print((save_path))
    compute_graph = lambda structure_type, structure_input, strategy, mp_id=None: EuclideanGraph(
        structure_type, structure_input, mp_id, strategy, cutoff, bond_tol
    ).to_dict()
    if structure_type == "dir":
        graphs = (
            Parallel(n_jobs=parallel, prefer="processes", verbose=10)(
                delayed(compute_graph)(structure_type, structure_input, strategy) for structure_input in structures
            )
            if parallel > 1
            else [compute_graph(structure_type, structure_input, strategy) for structure_input in structures]
        )
    else:
        graphs = (
            Parallel(n_jobs=parallel, prefer="processes", verbose=10)(
                delayed(compute_graph)(structure_type, structure_input, strategy, mp_id)
                for mp_id, structure_input in structures.items()
            )
            if parallel > 1
            else [
                compute_graph(structure_type, structure_input, strategy, mp_id)
                for mp_id, structure_input in structures.items()
            ]
        )
    graphs_dict = {graph.pop("mp_id"): graph for graph in graphs}

    if "_train.json" in save_path:
        prefix = str(save_path).replace("_train.json", "")
        suffix = "_train.json"
    elif "_valid.json" in save_path:
        prefix = str(save_path).replace("_valid.json", "")
        suffix = "_valid.json"
    elif "_test.json" in save_path:
        prefix = str(save_path).replace("_test.json", "")
        suffix = "_test.json"
    else:
        prefix = str(save_path).replace(".json", "")
        suffix = ".json"

    if save:
        save_path = f"{prefix}_{strategy}_cut={cutoff}_tol={bond_tol}{suffix}"
        utils.save_json(graphs_dict, save_path)
    return graphs_dict


def create_supergraphs(
    structures,
    structure_type,
    save_path,
    strategy,
    cutoff,
    bond_tol,
    parallel,
    save,
    cell_mapping="index",
):
    compute_graph = lambda structure_type, structure_input, strategy, mp_id=None: SuperGraph(
        structure_type,
        structure_input,
        mp_id,
        strategy=strategy,
        cutoff=cutoff,
        to_primitive=False,
        bond_tol=bond_tol,
        cell_mapping=cell_mapping,
    ).to_dict()
    if structure_type == "dir":
        graphs = (
            Parallel(n_jobs=parallel, prefer="processes", verbose=10)(
                delayed(compute_graph)(structure_type, structure_input, strategy) for structure_input in structures
            )
            if parallel > 1
            else [compute_graph(structure_type, structure_input, strategy) for structure_input in structures]
        )
    else:
        graphs = (
            Parallel(n_jobs=parallel, prefer="processes", verbose=10)(
                delayed(compute_graph)(structure_type, structure_input, strategy, mp_id)
                for mp_id, structure_input in structures.items()
            )
            if parallel > 1
            else [
                compute_graph(structure_type, structure_input, strategy, mp_id)
                for mp_id, structure_input in structures.items()
            ]
        )
    graphs_dict = {graph.pop("mp_id"): graph for graph in graphs}

    if "_train.json" in save_path:
        prefix = str(save_path).replace("_train.json", "")
        suffix = "_train.json"
    elif "_valid.json" in save_path:
        prefix = str(save_path).replace("_valid.json", "")
        suffix = "_valid.json"
    elif "_test.json" in save_path:
        prefix = str(save_path).replace("_test.json", "")
        suffix = "_test.json"
    else:
        prefix = str(save_path).replace(".json", "")
        suffix = ".json"

    if save:
        save_path = f"{prefix}_{strategy}_cut={cutoff}_tol={bond_tol}{suffix}"
        utils.save_json(graphs_dict, save_path)
    return graphs_dict


def separate_chunks(function, sequence, input_type, chunk_size, chunk, *args, **kwargs):
    """Used to split the graph creation task in smaller chunks. For use on compute clusters"""
    print(args)
    if input_type == "dir":
        list_chunks = [sequence[x : x + chunk_size] for x in range(0, len(sequence), chunk_size)]
        list_chunk = list_chunks[chunk]
    else:
        list_arg = list(sequence.items())
        list_chunks = [list_arg[x : x + chunk_size] for x in range(0, len(list_arg), chunk_size)]
        list_chunk = list_chunks[chunk]
        list_chunk = dict(list_chunk)
    kwargs["save_path"] = (
        str(kwargs.get("save_path")).split(".")[0]
        + f"_chunk={str(chunk)}."
        + str(kwargs.get("save_path")).split(".")[1]
    )

    return function(list_chunk, input_type, *args, **kwargs)


def main(argv):
    datapath = utils.DATA_PATH / "perov5"
    description = """
    This script creates graphs from .cif structure files.
    """
    default_args = {
        "structures_file": [
            (datapath / "structures" / "test_structures.json"),
        ],
        "structure_type": "json",
        "save_path": [
            str(datapath / "graphs" / "test_graphs.json"),
        ],
        "supergraphs": False,
        "strategy": "all",
        "cutoff": 5.0,
        "bond_tol": 0.5,
        "parallel": 8,
        "split_chunks": False,
        "chunk": 68,
        "chunk_size": 1000,
        "save": True,
    }
    help_dict = {
        "structures_dir": "path to structures directory (containing .json)",
        "structures_file": "path to json structures file (containing dict of cif strings)",
        "input_type": "type of input (directory or file)",
        "save_path": "path to save files",
        "strategy": "neighbor strategy for graph calculation, can be either 'voronoi' or 'all'",
        "cutoff": "cutoff distance for neighbor calculation, (recommended value with 'voronoi' is 13.0, with 'all' is 5.0)",
        "parallel": "number of workers for parrallel calculation",
        "split_chunks": "option to split the calculation in multiple chunks for compute clusters",
        "chunk": "if split_chunk, identifies the portion of the .cif upon which the calculation is done",
        "chunk_size": "number of .cif files per chunks",
    }
    args = utils.read_args(
        default_args=default_args,
        help_dict=help_dict,
        description=description,
        argv=argv,
    )

    out_list = []
    for i in range(len(args.structures_file)):
        structures = utils.load_json(args.structures_file[i])

        builder_function = create_supergraphs if args.supergraphs else create_graphs
        if args.split_chunks:
            out = separate_chunks(
                builder_function,
                structures,
                args.structure_type,
                args.chunk_size,
                args.chunk,
                save_path=args.save_path[i],
                strategy=args.strategy,
                cutoff=args.cutoff,
                bond_tol=args.bond_tol,
                parallel=args.parallel,
                save=args.save,
            )
        else:
            out = builder_function(
                structures,
                args.structure_type,
                save_path=args.save_path[i],
                strategy=args.strategy,
                cutoff=args.cutoff,
                bond_tol=args.bond_tol,
                parallel=args.parallel,
                save=args.save,
            )
        out_list.append(out)
    return out_list


def test():
    structures_path = utils.DATA_PATH / "materials_project" / "structures" / "structures_matproj_v2021_05_filtered.json"
    structures_cifs = utils.load_json(structures_path)
    # print(len(structures_cifs))
    l = list(structures_cifs.items())
    random.shuffle(l)
    structures_cifs = dict(l)
    # sample_structures_cif = list(structures_cifs.values())[710:800]
    # sample_ids = list(structures_cifs.keys())[710:800]
    # sample_structures_cif = [structures_cifs["mp-1214505"]]
    # sample_ids = ["mp-1214505"]
    sample_structures_cif = [structures_cifs["mp-22862"]]
    sample_ids = ["mp-22862"]
    # sample_structures_cif = [structures_cifs["mp-1183345"]]
    # sample_ids = ["mp-1183345"]
    num_invalid = 0
    num_disconnnected = 0
    num_directed = 0
    num_bonds = 0
    num_sites = 0
    bonds = []
    distances = []
    laren_dims = []
    gorai_dims = []
    num_3d = 0
    num_good = 0
    num_bad = 0

    for i, structure_cif in enumerate(sample_structures_cif):
        strategy = "isayev"
        cutoff = 6
        bond_tol = 0.25

        print(f"Structure {i} with ID {sample_ids[i]} PRIMITIVE")
        graph = SuperGraph(
            "json", structure_cif, sample_ids[i], strategy, cutoff=cutoff, bond_tol=bond_tol, to_primitive=False
        )
        if not graph.sites1:
            print(graph.status)
            num_invalid += 1
            continue
        connected = graph.is_connected()
        undirected = graph.is_undirected()
        adj = graph.compute_adjacency_matrix()
        print(f"Connected : {connected}, undirected : {undirected}")
        # print(f"Adjacency : {adj}")

        laren_dim, gorai_dim = get_dimensionality(strategy, structure_cif, "json", cutoff, bond_tol)
        laren_dims.append(laren_dim)
        gorai_dims.append(gorai_dim)
        # if laren_dim != 3:
        #     print("Structure is not 3d")
        #     continue
        # else:
        #     num_3d += 1
        # if any(np.sum(adj, 1) == 1):
        #     print("Site with one bond")
        num_bonds += len(graph.sites1)
        num_sites += len(graph.site_atomic_numbers)
        bonds.extend(np.sum(adj, 1))
        distances.extend(graph.bond_distances)
        if not undirected:
            num_directed += 1
        if not connected:
            num_disconnnected += 1
        graph.plot()
        print(f"Laren dim {laren_dim} , Gorai dim {gorai_dim}")
        print(f"Number sites: {num_sites}")
        print(f"Number bonds: {num_bonds}")
        print(f"Number disconnected : {num_disconnnected}")
        print(f"Number directed : {num_directed}")
    print(f"Average number of bonds : {np.average(bonds)} std : {np.std(bonds)}")
    print(f"3D structures : {num_3d}")
    print(f"Invalid structures : {num_invalid}")
    print(f"Num good {num_good}")
    print(f"Num bad {num_bad}")

    # laren_dims = np.array(laren_dims)
    # plt.hist(laren_dims, bins=5)

    # gorai_dims = np.array(gorai_dims)
    # plt.hist(gorai_dims, bins=5)

    # plt.show()

    # distances = np.array(distances)
    # q25, q75 = np.percentile(distances, [25, 75])
    # bin_width = 2 * (q75 - q25) * len(distances) ** (-1 / 3)
    # bins = round((distances.max() - distances.min()) / bin_width)
    # # print("Freedman–Diaconis number of bins:", bins)
    # plt.hist(distances, bins=bins)

    # plt.show()

    # bonds = np.array(bonds)
    # q25, q75 = np.percentile(bonds, [25, 75])
    # bin_width = 2 * (q75 - q25) * len(bonds) ** (-1 / 3)
    # bins = round((bonds.max() - bonds.min()) / bin_width)
    # # print("Freedman–Diaconis number of bins:", bins)
    # plt.hist(bonds, bins=bins)

    # plt.show()


def draw_graph():
    structure_dict = utils.load_json(utils.DATA_PATH / "mp-1186159_graph.json")
    structure_graph = StructureGraph.from_dict(structure_dict)
    print(structure_graph)

    # graph.draw_graph_to_file(
    #     utils.DATA_PATH / "graph-1186159.pdf",
    #     hide_image_edges=False,
    #     keep_dot=True,
    #     algo="neato",
    # )
    # G = pgv.AGraph(utils.DATA_PATH / "graph-10054.dot")
    # G.draw(utils.DATA_PATH / "graph-10054.pdf", format="pdf", prog="dot")


def test_supercell():
    structures_path = utils.DATA_PATH / "materials_project" / "structures" / "structures_matproj_v2021_05.json"
    structures_cifs = utils.load_json(structures_path)
    # sample_structures_cif = list(structures_cifs.values())[:1000]
    # sample_ids = list(structures_cifs.keys())[:1000]
    structure_input = structures_cifs["mp-19006"]
    sample_ids = "mp-19006"
    try:
        material_structure = Structure.from_str(structure_input, "json")
        print("Parsed CIF")
    except ValueError:
        print("CIF could not be parsed")

    material_structure.__class__ = CustomStructure
    _, lattice_string = get_bravais_lattice(material_structure)
    if not material_structure.site_properties:
        print(" NO MAGMONS")

    analyzer = SpacegroupAnalyzer(material_structure)
    primitive_structure = analyzer.find_primitive()
    # primitive_structure.translate_sites(list(range(len(primitive_structure))), primitive_structure[0].frac_coords)
    symmetry_ops = analyzer.get_symmetry_operations()
    structure_matcher = StructureMatcher(primitive_cell=False, attempt_supercell=True)
    supercell, translation, indices = structure_matcher.get_transformation(material_structure, primitive_structure)
    supercell_inv = 2 * np.linalg.inv(supercell)
    material_structure.make_supercell(supercell_inv, False)
    equality = material_structure == primitive_structure
    id_supercell = (supercell == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all()
    u = 2


if __name__ == "__main__":
    main(sys.argv[1:])
    # test()
    # test_supercell()
