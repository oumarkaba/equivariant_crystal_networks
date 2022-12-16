import sys
from learn_materials import utils
from learn_materials.parse import magnetic_materials as mm
from learn_materials.parse import materials_project as mp
from learn_materials.prepare.datamodel import (
    Material,
    load_material_file,
    save_material_file,
)


def deduplicate(materials, mm_stats=False):
    duplicate_materials = {}
    fused_materials = []
    if mm_stats:
        curie = 0
        neel = 0
        curie_and_neel = 0
    for i, material1 in enumerate(materials):
        print(f"processing material {i} of {len(materials)}")
        if not duplicate_materials.get(material1.uid, False):
            fused_material = material1
            for material2 in materials[i + 1 :]:
                if (
                    not duplicate_materials.get(material2.uid, False)
                    and material1.chemical_formula
                    and material1.chemical_formula == material2.chemical_formula
                ):
                    fused_material = Material.combine(fused_material, material2)
                    duplicate_materials[material2.uid] = True
            if mm_stats:
                curie += (
                    1
                    if fused_material.get_property("Curie temperature")
                    and not fused_material.get_property("Néel temperature")
                    else 0
                )
                neel += (
                    1
                    if fused_material.get_property("Néel temperature")
                    and not fused_material.get_property("Curie temperature")
                    else 0
                )
                curie_and_neel += (
                    1
                    if fused_material.get_property("Curie temperature")
                    and fused_material.get_property("Néel temperature")
                    else 0
                )
            fused_materials.append(fused_material)
    if mm_stats:
        print(
            f"output {len(fused_materials)} materials. curie :{curie}, neel:{neel}, curie and neel:{curie_and_neel}, duplicates:{len(duplicate_materials)} out of {len(materials)} total"
        )
    return fused_materials


def filter_materials_with_prop(
    materials, chemical_formula=False, all_properties=None, one_of_properties=None
):
    initial_number = len(materials)
    if chemical_formula:
        materials = [material for material in materials if material.chemical_formula]
    print(f"output {len(materials)} out of {initial_number} materials")
    if all_properties:
        for prop in all_properties:
            materials = [
                material for material in materials if material.get_property(prop)
            ]
    if one_of_properties:
        materials = [
            material
            for material in materials
            if material.property_names()
            and not set(one_of_properties).isdisjoint(material.property_names())
        ]
    print(f"output {len(materials)} out of {initial_number} materials")
    return materials


def filter_materials_prop(materials, prop, condition, value):
    initial_number = len(materials)

    conditions = {
        "lower": lambda prop_value, threshold: prop_value is not None
        and prop_value < threshold,
        "greater": lambda prop_value, threshold: prop_value is not None
        and prop_value > threshold,
        "equal": lambda prop_value, threshold: prop_value is not None
        and prop_value == threshold,
    }

    materials = [
        material
        for material in materials
        if conditions[condition](material.get_property_value(prop), value)
    ]
    print(f"output {len(materials)} out of {initial_number} materials")
    return materials


def filter_materials_by_id(materials, id_file):
    id_list = utils.load_json(id_file)
    filtered_materials = [
        material for material in materials if material.get_mp_id() in id_list
    ]
    print(f"{len(filtered_materials)} filtered materials out of {len(materials)}")
    return filtered_materials


def purge(materials):
    deduplicated_materials = deduplicate(materials, True)
    filtered_materials = filter_materials_with_prop(
        deduplicated_materials,
        chemical_formula=True,
        one_of_properties=["Curie temperature", "Néel temperature"],
    )
    return filtered_materials


def update_descriptors(materials, verbose=False):
    materials_with_props = []
    N_material = len(materials)
    for i, material in enumerate(materials):
        if verbose and N_material > 100:
            if i == 0 or ((i + 1) % (N_material // 100) == 0):
                print(f"compound {i+1}/{N_material}")
        elif verbose:
            print(f"compound {i+1}/{N_material}")
        material.update_composition_descriptors()
        materials_with_props.append(material)
    return materials_with_props


def main(argv):
    description = """
        This script applies various filters and updates to the materials dataset
        from `parsed` file and saves the result in the `out` file in PIF format.
        Options enable to also parse the original file beforehand. 
    """
    usage = """
        Use default options to process materials project:
            
            $ python3 cleaner.py
        
        options to process magnetic materials:

            $ cleaner.py 
                --purge 
                --parsed ../datas/magnetic_materials/train_parsed.json
                --out ../data/prepared/train_prepared.json 
    
        options for full treatment of magnetic materials:

            $ cleaner.py 
                --purge 
                --parse_mm 
                --original ../data/magnetic_materials/phase_transitions_train.json
                --parsed ../data/magnetic_materials/train_parsed.json 
                --out ../data/train_prepared.json 
    """
    datapath = utils.DATA_PATH / "materials_project"
    default_args = {
        "original": [
            str(datapath / "MP_ICSD_compounds_PIFs_train.json"),
            str(datapath / "MP_ICSD_compounds_PIFs_valid.json"),
            str(datapath / "MP_ICSD_compounds_PIFs_test.json"),
        ],
        "parsed": [
            str(datapath / "parsed_matproj_v2020_06_train.json"),
            str(datapath / "parsed_matproj_v2020_06_valid.json"),
            str(datapath / "parsed_matproj_v2020_06_test.json"),
        ],
        "out": [
            str(datapath / "parsed_matproj_e_hull>0,05_v2020_06_train.json"),
            str(datapath / "parsed_matproj_e_hull>0,05_v2020_06_valid.json"),
            str(datapath / "parsed_matproj_e_hull>0,05_v2020_06_test.json"),
        ],
        "parse_mm": False,
        "parse_mp": False,
        "purge": False,
        "new_descriptors": False,
        "filter": "lower",
        "filter_prop": "Energy above hull",
        "value": 0.05,
        "filter_ids": None,
        "materials": str(datapath / "parsed_matproj_v2020_06.json"),
        "filter_ids_out": str(datapath / "parsed_matproj_v2020_06_benjamin.json"),
    }
    help_dict = {
        "original": "path to the original raw dataset",
        "parsed": "path to the parsed dataset",
        "out": "path to save the processed dataset",
        "parse_mp": "runs the parsing step for materials project",
        "parse_mm": "runs the parsing step for magnetic materials (disables parse_mp)",
        "purge": "remove bad data and merge duplicates (for magnetic materials)",
        "new_descriptors": "update descriptors",
        "filter": "filters materials based on a prop value, greater, lower or equal",
        "filter_prop": "property on which to filter",
        "value": "filter threshold",
    }
    args = utils.read_args(
        default_args=default_args,
        help_dict=help_dict,
        description=description,
        usage=usage,
        argv=argv,
    )

    is_default_original = args.original == default_args["original"]
    is_default_parsed = args.parsed == default_args["parsed"]
    is_default_out = args.out == default_args["out"]
    is_one_default = is_default_original or is_default_parsed or is_default_out

    if is_one_default and args.parse_mm:
        raise ValueError("you must change paths to parse magnetic materials")

    if args.filter_ids:
        materials_list = load_material_file(args.materials)
        filtered_materials = filter_materials_by_id(materials_list, args.filter_ids)
        save_material_file(filtered_materials, args.filter_ids_out)
        return

    for i, savepath in enumerate(args.parsed):
        if args.parse_mm:
            print("parsing magnetic materials dataset")
            stats_path = savepath.replace(".json", "_stats.json")
            mm.parse_mm(args.original[i], args.parsed[i], stats_path)
        elif args.parse_mp:
            print("parsing materials project dataset")
            mp.parse_mp(args.original[i], args.parsed[i])
        else:
            print(f"loading {args.parsed[i]}")
            materials = load_material_file(args.parsed[i])

        if args.purge:
            print("removing duplicates and filtering")
            materials = purge(materials)

        if args.new_descriptors:
            print("updating descriptors")
            materials = update_descriptors(materials, verbose=True)

        if args.filter:
            print(f"filtering materials based on property {args.filter_prop}")
            materials = filter_materials_prop(
                materials, args.filter_prop, args.filter, args.value
            )

        save_material_file(materials, args.out[i])
        print(f"{len(materials)} materials saved in {args.out[i]}")

    print("\navailable properties:")
    for prop in materials[0].properties:
        print("  ", prop.name)
    print()


if __name__ == "__main__":
    main(sys.argv)
