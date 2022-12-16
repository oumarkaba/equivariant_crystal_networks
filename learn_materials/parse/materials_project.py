import sys
import json
import pymatgen as mg
from pymatgen.ext.matproj import MPRester
from copy import deepcopy

from learn_materials import utils
from learn_materials.prepare.datamodel import (
    save_material_file,
    load_material_dicts,
    load_material_file,
)

# FIXME: this should be removed from github, but needs a mention in the README
MPRESTER = MPRester("41K885Dm2G3OqCcg0G")

ALL_KNOWN_PROPERTIES = [
    "material_id",  # str
    "anonymous_formula",  # dict
    "band_gap",  # float  # The calculated band gap
    "band_structure",  # The calculated "line mode" band structure (along selected symmetry lines -- aka "branches", e.g. \Gamma to Z -- in the Brillouin zone) in the pymatgen json representation
    "bandstructure_uniform",  # The calculated uniform band structure in the pymatgen json representation
    "blessed_tasks",  # dict
    "bond_valence",
    "chemsys",  # str
    "cif",  # str  # A string containing the structure in the CIF format.
    # "cifs",  # dict
    "created_at",  # str
    "delta_volume",
    "density",  # float  # Final relaxed density of the material
    "diel",  # Dielectric properties. Contains a tensor (one each for total and electronic contribution) and derived properties, e.g. dielectric constant, refractive index, and recognized potential for ferroelectricity.
    "doi",
    "doi_bibtex",
    "dos",  # The calculated density of states in the pymatgen json representation
    "e_above_hull",  # int  # Calculated energy above convex hull for structure. Please see Phase Diagram Manual for the interpretation of this quantity.
    "efermi",  # float
    "elasticity",  # Mechanical properties in the elastic limit. Contains the full elastic tensor as well as derived properties, e.g. Poisson ratio and bulk (K) and shear (G) moduli. Consult our hierarchical documentation for the particular names of sub-keys.
    # "elasticity_third_order",
    "elements",  # list  # A array of the elements in the material
    "encut",  # int
    "energy",  # float  # Calculated vasp energy for structure
    "energy_per_atom",  # float  # Calculated vasp energy normalized to per atom in the unit cell
    "entry",  # This is a special property that returns a pymatgen ComputedEntry in the json representation. ComputedEntries are the basic unit for many structural and thermodynamic analyses in the pymatgen code base.
    "exp",  # dict
    "final_energy",  # float
    "final_energy_per_atom",  # float
    "formation_energy_per_atom",  # float  # Calculated formation energy from the elements normalized to per atom in the unit cell
    "formula_anonymous",  # str
    "full_formula",  # str
    "has",  # list
    "has_bandstructure",  # bool
    # "hubbards",  # dict  # An array of Hubbard U values, where applicable.
    "icsd_ids",  # list  # List of Inorganic Crystal Structure Database (ICSD) ids for structures that have been deemed to be structurally similar to this material based on pymatgen's StructureMatcher algorithm, if any.
    # "initial_structure",  # pymatgen.core.structure.Structure  # The initial input structure for the calculation in the pymatgen json representation (see later section).
    # "input",  # dict
    "is_compatible",  # bool  # Whether this calculation is considered compatible under the GGA/GGA+U mixing scheme.
    "is_hubbard",  # bool  # A boolean indicating whether the structure was calculated using the Hubbard U extension to DFT
    "is_ordered",  # bool
    "last_updated",  # str
    "magnetic_type",  # str
    "magnetism",  # dict
    "nelements",  # int  # The number of elements in the material
    "nkpts",
    "nsites",  # int  # Number of sites in the unit cell
    "ntask_ids",  # int
    "original_task_id",  # str
    "oxide_type",  # str
    "pf_ids",  # list
    # "piezo",  # Piezoelectric properties. Contains a tensor and derived properties. Again, consult our repository for the names of sub-keys.
    "pretty_formula",  # A nice formula where the element amounts are normalized
    "pseudo_potential",  # dict
    "reduced_cell_formula",  # dict
    "run_type",  # str
    # "snl",  ## causes ModuleNotFoundError: No module named 'pybtex'
    # "snl_final",  ## causes ModuleNotFoundError: No module named 'pybtex'
    "spacegroup",  # dict  # An associative array containing basic space group information.
    "structure",  # pymatgen.core.structure.Structure  # An alias for final_structure. # The final relaxed structure in the pymatgen json representation (see later section).
    "task_id",  # str
    "task_ids",  # list
    # "total_magnetization",  # float  # magnetic moment per formula unit
    "unit_cell_formula",  # dict  # The full explicit formula for the unit cell
    "volume",  # float  # Final relaxed volume of the material
    "warnings",  # list
    # "xrd",  # dict
]


def download_all_materials_project(
    min_nelements=1,
    max_nelements=10,
    max_nsites=500,
    nsites_step=500,
    chunk_size=1000,
    properties=ALL_KNOWN_PROPERTIES,
):
    """Download all data from the materials project and returns a list of
    dictionnary.

    Only 124 331 out of 124 515 materials claimed on the website are obtainable.
    the database is downloaded by a sequence of queries, each one is for a given
    number of elements in the materials and then number of sites. Queries are
    broken into chunks to prevent exceeding the limit size allowed.
    Parameters:
    - max_nelements: the maximum value of nelements used in query loop
    - max_nsites: the maximum value of nsites used for each nvalues
    - nsites_step: number of different value taken by nsites in each queries
    - chunk_size: size to break queries
    - properties: the properties to be downloaded (affect the size of queries)
    some properties are dictionnary themselves and may require to be unrolled
    (magnetism as an example in the source file of this function)
    """
    list_of_dict = []
    print("downloading items (multiple queries required)")
    for nelements in range(min_nelements, max_nelements + 1):
        for nsites in range(1, max_nsites, nsites_step + 1):
            query = MPRESTER.query(
                criteria={
                    "nelements": nelements,
                    "nsites": {"$gt": nsites, "$lt": nsites + nsites_step + 1},
                },
                properties=properties,
                chunk_size=chunk_size,
            )
            list_of_dict.extend(query)
    return list_of_dict


def unfold_magnetism(mp_list):
    mp_list = deepcopy(mp_list)
    new_list = []
    print("unfolding magnetism")
    for mat in mp_list:
        try:
            magnetism = mat.pop("magnetism")
            magnetism["num_magnetic_sites"] = int(magnetism.pop("num_magnetic_sites"))  # str to int
            magnetism["true_total_magnetization"] = magnetism.pop(
                "total_magnetization"
            )  # different from mp's "total magnetisation" which is the total magnetisation per formula
            mat.update(magnetism)
            new_list.append(mat)
        except KeyError:
            pass
    return mp_list


def parse_mp(save_pif=None, save_structures=None):
    mp_data = unfold_magnetism(download_all_materials_project())
    parsed_entries = []

    for i, entry in enumerate(mp_data):
        parsed_entry = parse_entry(entry)
        parsed_entries.append(parsed_entry)
        if i % 10 == 0:
            print(f"Parsed {i} entries out of {len(mp_data)}")

    materials = load_material_dicts(parsed_entries)
    if save_pif:
        save_material_file(materials, save_pif)
        print(f"Saved {len(materials)} parsed entries from Materials Project in {save_pif}")
    if save_structures:
        structures = {entry["material_id"]: entry["structure"].to("json") for entry in mp_data}
        utils.save_json(structures, save_structures)
        print(f"Saved {len(materials)} structures from Materials Project in {save_structures}")

    return materials


def parse_entry(entry):
    parsed_entry = {
        "tags": ["Materials Project", entry["run_type"]],
        "ids": [{"name": "Materials Project", "value": entry["material_id"]}]
        + [{"name": "ICSD", "value": icsd_id} for icsd_id in entry["icsd_ids"]],
        "references": [
            {
                "url": f"https://materialsproject.org/materials/{entry['material_id']}",
                "doi": "http://dx.doi.org/10.1063/1.4812323",
            }
        ],
        "chemicalFormula": mg.Composition(entry["full_formula"]).formula,
        "properties": parse_properties(entry),
        "category": "system.chemical",
    }
    return parsed_entry


def parse_properties(entry):
    selected_properties = {
        "energy": "Final energy",
        "energy_per_atom": "Final energy per atom",
        "efermi": "Fermi energy",
        "volume": "Volume",
        # "volume_per_atom*": "Volume per atom",
        "nsites": "Number of sites",
        "formation_energy_per_atom": "Formation energy per atom",
        "e_above_hull": "Energy above hull",
        "band_gap": "Band gap",
        "true_total_magnetization": "True total magnetization",
        "number": "Spacegroup number",
        "crystal_system": "Crystal system",
        "point_group": "Point group",
        "magnetic_type": "Magnetic type",
    }
    parsed_properties = []
    entry.update(entry["spacegroup"])
    for name, value in entry.items():
        prop = {}
        if name in selected_properties:
            prop["name"] = selected_properties[name]
            prop["scalars"] = (
                [{"value": value, "tags": ["Materials Project"]}] if not isinstance(value, list) else value
            )
            parsed_properties.append(prop)

    volume_per_atom = {
        "name": "Volume per atom",
        "scalars": [{"value": entry["volume"] / entry["nsites"], "tags": ["Materials Project"]}],
    }
    parsed_properties.append(volume_per_atom)
    magnetization_per_atom = {
        "name": "True total magnetization per atom",
        "scalars": [
            {
                "value": entry["true_total_magnetization"] / entry["nsites"],
                "tags": ["Materials Project"],
            }
        ],
    }
    parsed_properties.append(magnetization_per_atom)

    return parsed_properties


def collect_ids(materials_file, output_path):
    print(str(output_path))
    materials = load_material_file(materials_file)
    id_list = [material.get_mp_id() for material in materials]

    utils.save_json(id_list, output_path)


def main(argv):
    datapath = utils.DATA_PATH / "materials_project"
    description = """
        this script parses the rew files of the materials project dataset 
        and saves the result in the `parsed` file in PIF format
    """
    default_args = {
        "parsed": [
            str(datapath / "parsed_matproj_v2021_05.json"),
        ],
        "cif": [str(datapath / "structures_matproj_v2021_05.json")],
    }
    help_dict = {
        "ids": "list of matproj ids to download data from",
        "parsed": "list of paths to save the parsed dataset (in order)",
        "cif": "list of paths to save the cif structures strings",
    }
    args = utils.read_args(
        default_args=default_args,
        help_dict=help_dict,
        description=description,
        argv=argv,
    )

    for i in range(len(args.parsed)):
        parse_mp(args.parsed[i], args.cif[i])


if __name__ == "__main__":
    # materials_file = (
    #     utils.DATA_PATH
    #     / "materials_project"
    #     / "parsed"
    #     / "parsed_matproj_v2021_05.json"
    # )
    # ids_file = utils.DATA_PATH / "materials_project" / "ids_matproj_v2020_06_train.json"

    # collect_ids(materials_file, ids_file)
    main(sys.argv[1:])
