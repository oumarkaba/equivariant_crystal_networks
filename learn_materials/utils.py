import os
import sys
import json
import argparse
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

SRC_PATH = pathlib.Path(__file__).parent
DATA_PATH = SRC_PATH / "data"
RESULTS_PATH = SRC_PATH / "results"
if "LM_DATA" in os.environ:
    DATA_PATH = os.environ["LM_DATA"]
if "LM_RESULTS" in os.environ:
    RESULTS_PATH = os.environ["LM_RESULTS"]


def read_args(
    default_args,
    help_dict=None,
    description=None,
    usage=None,
    argv=sys.argv,
    out_dict=False,
):
    parser = argparse.ArgumentParser(description=description, usage=usage)
    for name, default in default_args.items():
        helpstr = ""
        try:
            helpstr = help_dict[name]
        except (KeyError, NameError, TypeError):
            pass

        if type(default) is list:
            parser.add_argument(
                "--" + name,
                nargs="+",
                type=type(default[0]),
                default=default,
                help=helpstr,
            )
        elif type(default) is bool:
            if default:
                parser.add_argument(
                    "--no_" + name,
                    dest=name,
                    action="store_false",
                    help=f"disables {name}",
                )
            else:
                parser.add_argument(
                    "--" + name, action="store_true", default=default, help=helpstr
                )
        else:
            parser.add_argument(
                "--" + name, type=type(default), default=default, help=helpstr
            )
    if out_dict:
        return vars(parser.parse_known_args(argv)[0])
    else:
        return parser.parse_known_args(argv)[0]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class DescriptorDataset(Dataset):
    def __init__(self, descriptors, targets, target_names):
        self.descriptors = descriptors
        self.targets = targets
        self.target_names = target_names

    def select_targets(self, selected_targets):
        target_indices = [
            index
            for index, item in enumerate(self.target_names)
            if item in set(selected_targets)
        ]
        target_indices = torch.LongTensor(target_indices)
        self.targets = torch.index_select(self.targets, 1, target_indices)

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, index):
        return self.descriptors[index], self.targets[index]


class MulticategoryDataset(Dataset):
    def __init__(self, descriptors, targets, target_names):
        self.descriptors = descriptors
        self.targets = targets
        self.target_names = target_names
        self.category = None

    def select_targets(self, selected_targets):
        target_indices = [
            index
            for index, item in enumerate(self.target_names)
            if item in set(selected_targets)
        ]
        target_indices = torch.LongTensor(target_indices)
        self.targets = {
            key: torch.index_select(targets, 1, target_indices)
            if targets.numel() != 0
            else targets
            for key, targets in self.targets.items()
        }

    def set_category(self, category):
        self.category = category
        return self

    def get_categories(self):
        categories = [
            key for key, targets in self.targets.items() if targets.numel() != 0
        ]
        return categories

    def __len__(self):
        return len(self.descriptors[self.category])

    def __getitem__(self, index):
        return (
            self.descriptors[self.category][index],
            self.targets[self.category][index],
        )


class IcsdDataset(Dataset):
    def __init__(self, descriptors, icsd_list_location, icsd_ids):
        self.descriptors = descriptors
        self.icsd_list_location = icsd_list_location
        self.icsd_ids = icsd_ids

    def __len__(self):
        return len(self.descriptors)

    def __getitem__(self, index):
        if self.descriptors[index] is None:
            print(self.descriptors[index])
        else:
            return self.descriptors[index], self.icsd_list_location[index]


def load_stacked_json(path):
    with open(path, "r+", encoding="utf-8") as f:
        entries = []
        for line in f:
            entries.append(json.loads(line))
        return entries


def save_stacked_json(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, cls=NpEncoder))
            f.write("\n")


def load_json(path):
    with open(path, "r+") as f:
        return json.load(f)


def save_json(entries, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, cls=NpEncoder)


def save_model(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    return model


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=device))
    return model


def string_to_float(string):
    new_string = string.replace("−", "-")
    try:
        return float(new_string)
    except ValueError:
        return None
