import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation


from learn_materials import utils
from learn_materials.prepare.datamodel import Material, load_material_file

prepareed_mm_file = (
    utils.SRC_PATH / "datasets" / "prepared" / "prepareed_mm.json"
)

missing_entries = -10


class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def plot_descriptors(path):
    """
    Plots every descriptor for each material. In x the materials, in y the value of the descriptor. Colors are associated with descriptors.
    """
    descriptor_names = Material.all_descriptors()
    descriptor_names.extend(["Curie temperature", "Néel temperature"])
    materials = load_material_file(path)

    mat_descriptors = [
        material.get_properties(descriptor_names, nan=missing_entries)
        for material in materials
    ]

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    colors = cm.rainbow(np.linspace(0, 1, len(mat_descriptors)))
    for material, c in zip(mat_descriptors, colors):
        ax.scatter(range(len(descriptor_names)), material, color=c, s=5)

    # ax.set_yscale("log")
    plt.xticks(
        range(len(descriptor_names)), descriptor_names, rotation="vertical", ha="center"
    )
    plt.tight_layout()
    plt.ylim(-15, 100)
    plt.show()


def plot_materials(path, descript=None):
    """
    Plots every descriptor for each material. In x the descriptors, in y the value of the descriptor.
    """
    descriptor_names = Material.all_descriptors()
    descriptor_names.extend(["Curie temperature", "Néel temperature"])
    if descript:
        descriptor_names = list(set(descriptor_names).intersection(descript))

    materials = load_material_file(path)
    print(len(materials))
    mat_descriptors = [
        material.get_properties_values(descriptor_names, nan=missing_entries)
        for material in materials
    ]

    descriptors = list(zip(*mat_descriptors))

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    colors = cm.rainbow(np.linspace(0, 1, len(descriptors)))
    for material, c in zip(descriptors, colors):
        ax.scatter(range(len(materials)), material, color=c, s=5)

    # ax.set_yscale("log")
    plt.tight_layout()
    # plt.ylim(-15, 100)
    plt.show()


def plot_temperatures(path):
    """
    Plots Curie and Neel for each material. In x the materials, in y the value of the descriptor. Colors are associated with descriptors.
    """
    materials = load_material_file(path)
    print(len(materials))
    data = {"Curie": [], "Néel": []}
    for i, material in enumerate(materials):
        curie_temperatures = (
            material.get_property("Curie temperature").scalars
            if material.get_property("Curie temperature")
            else None
        )
        if curie_temperatures:
            curie_values = [
                (temperature.value, i) for temperature in curie_temperatures
            ]
            data["Curie"].extend(curie_values)
        neel_temperatures = (
            material.get_property("Néel temperature").scalars
            if material.get_property("Néel temperature")
            else None
        )
        if neel_temperatures:
            neel_values = [(temperature.value, i) for temperature in neel_temperatures]
            data["Néel"].extend(neel_values)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    curie_temps, curie_labels = zip(*data["Curie"])
    neel_temps, neel_labels = zip(*data["Néel"])
    ax.scatter(curie_labels, curie_temps, color="red", s=5)
    ax.scatter(neel_labels, neel_temps, color="blue", s=5)
    ax.set_yscale("symlog")
    # ax.set_ylim(0, 10000)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.show()


def main():
    # plot_temperatures(prepareed_mm_file)
    sample_cif = utils.SRC_PATH / "datasets" / "original" / "cif" / "9000046.cif"
    # plot_graph(sample_cif)


if __name__ == "__main__":
    main()
