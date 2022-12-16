import os
import sys
import learn_materials.utils as utils
import json
import random


def truncate_json_list(in_file, stop, outfile=None):
    in_file = str(in_file)
    prefix = in_file.replace(".json", "")

    try:
        in_list = utils.load_json(in_file)
        is_stack = False
    except json.decoder.JSONDecodeError:
        in_list = utils.load_stacked_json(in_file)
        is_stack = True

    if len(in_list) < stop:
        stop = len(in_list)

    out_list = in_list[:stop]
    if outfile == "":
        outfile = f"{prefix}_{stop}.json"

    if is_stack:
        utils.save_stacked_json(out_list, outfile)
    else:
        utils.save_json(out_list, outfile)


def split_train_json(
    file_path,
    train_proportion,
    validation_proportion,
    test_proportion,
    seed=0,
    output_train_path=None,
    output_validation_path=None,
    output_test_path=None,
):
    random.seed(seed)
    try:
        data_list = utils.load_json(file_path)
        is_stack = False
    except json.decoder.JSONDecodeError:
        data_list = utils.load_stacked_json(file_path)
        is_stack = True
    random.shuffle(data_list)

    train_size = int(len(data_list) * train_proportion)
    validation_size = int(len(data_list) * validation_proportion)
    test_size = (
        int(len(data_list) * validation_proportion)
        if test_proportion
        else len(data_list) - train_size - validation_size
    )
    train_list = data_list[:train_size]
    validation_list = data_list[train_size : train_size + validation_size]
    test_list = data_list[train_size + validation_size : train_size + validation_size + test_size]

    file_prefix = str(file_path).replace(".json", "")
    if output_train_path is None:
        output_train_path = f"{file_prefix}_train.json"
    if output_validation_path is None:
        output_validation_path = f"{file_prefix}_valid.json"
    if output_test_path is None:
        output_test_path = f"{file_prefix}_test.json"

    if is_stack:
        utils.save_stacked_json(train_list, output_train_path)
        utils.save_stacked_json(validation_list, output_validation_path)
        utils.save_stacked_json(test_list, output_test_path)
    else:
        utils.save_json(train_list, output_train_path)
        utils.save_json(validation_list, output_validation_path)
        utils.save_json(test_list, output_test_path)


def subsample_train_json(file_path, train_proportion, number_reductions, seed=0, output_path=None):
    random.seed(seed)
    try:
        data_list = utils.load_json(file_path)
        is_stack = False
    except json.decoder.JSONDecodeError:
        data_list = utils.load_stacked_json(file_path)
        is_stack = True
    random.shuffle(data_list)

    train_size = len(data_list)
    train_subsets = []
    for i in range(1, number_reductions):
        train_size = int(train_size * train_proportion)
        data_list = data_list[:train_size]

        output_train_path = output_path / f"train_data_{train_proportion ** i}.json"
        utils.save_json(data_list, output_train_path)


def merge_json_files(input_dir, output_path):
    """Used to merge files together"""
    print("Merging files")
    json_dict = {}
    json_list = []
    for file in input_dir.glob("**/*"):
        print(f"Adding file {file}")
        json_object = utils.load_json(file)
        if isinstance(json_object, dict):
            json_dict.update(json_object)
        elif isinstance(json_object, list):
            json_list.extend(json_object)
    merged_json = json_dict if len(json_list) == 0 else json_list.extend(json_dict.values)
    print(len(merged_json))
    utils.save_json(merged_json, output_path)


def main(argv):
    description = """
    This script has three modes. 
    truncate takes any .json file containing a list of entries and produce a new
    file containing the N first entries, with an indicative suffix to the same
    name. split takes any .json and randomly split it two file according to the
    defined proportion. Saves the files with suffixes 'train' and 'test'.
    merge takes multiple json files containing lists or dictionnaries and merges
    them into a single json file   
    """
    mp_path = utils.DATA_PATH / "materials_project"
    default_args = {
        "path": utils.DATA_PATH / "materials_project" / "matproj_graphs",
        "output": utils.DATA_PATH / "materials_project" / "matproj_graphs_crystalnn.json",
        "stop": 4000,
        "train_proportion": 0.8,
        "validation_proportion": 0.1,
        "test_proportion": 0.1,
        "seed": 72,
        "mode": "subsample",
    }
    help_dict = {
        "path": "path to the file to truncate",
        "stop": "number of entries in the reduced dataset for truncate",
        "train_proportion": "proportion of entries in the training dataset for split",
        "validation_proportion": "proportion of entries in the training dataset for split",
        "test_proportion": "proportion of entries in the test dataset for split",
        "seed": "random seed for split",
        "mode": "either 'truncate' or 'split'",
    }
    args = utils.read_args(
        default_args=default_args,
        help_dict=help_dict,
        description=description,
        argv=argv,
    )
    if args.mode == "truncate":
        truncate_json_list(args.path, args.stop, args.output)
    elif args.mode == "split":
        split_train_json(
            args.path,
            args.train_proportion,
            args.validation_proportion,
            args.test_proportion,
            args.seed,
        )
    elif args.mode == "merge":
        merge_json_files(args.path, args.output)
    elif args.mode == "subsample":
        subsample_train_json(
            mp_path / "parsed" / "parsed_matproj_v2021_05_nonduplicate_small_3d_valid_graphs_train.json",
            0.5,
            4,
            output_path=mp_path / "parsed",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
