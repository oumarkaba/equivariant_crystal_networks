import itertools
import numpy as np
import matplotlib.pyplot as plt
from learn_materials.prepare.plot import annotate3D
from scipy.sparse import csr_matrix
import torch

import learn_materials.utils as utils
from learn_materials.equivariant.groups import *
from learn_materials.equivariant.autoequiv.core import (
    create_colored_matrix,
    create_colored_vector,
    dict_to_matrix,
    LinearEquiv,
    LinearEquivDepth,
)
from learn_materials.equivariant.autoequiv.viz import (
    draw_colored_matrix,
    draw_colored_bipartite_graph,
    draw_colored_vector,
)


class Cluster:
    def __init__(self, side):
        self.side = side
        self.dimension = 2
        self.size = self.side ** self.dimension
        self.supercell = self.side * np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )

    def plot_grid(self):
        if self.dimension == 2:
            x_coordinates = self.points.T[0]
            y_coordinates = self.points.T[1]
            labels = [str(i) for i in range(len(self.points))]
            offset = self.unit_lenght * 0.05
            plt.plot(x_coordinates, y_coordinates, marker=".", color="k", linestyle="none")
            for i, label in enumerate(labels):
                plt.text(x_coordinates[i] + offset, y_coordinates[i] + offset, label)
        elif self.dimension == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            x_coordinates = self.points.T[0]
            y_coordinates = self.points.T[1]
            z_coordinates = self.points.T[2]
            ax.scatter(x_coordinates, y_coordinates, z_coordinates)
            for j, xyz_ in enumerate(self.points):
                annotate3D(
                    ax,
                    s=str(j),
                    xyz=xyz_,
                    fontsize=10,
                    xytext=(-3, 3),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                )

        plt.show()

    def build_prism_cluster(self):
        x_coordinates = np.arange(0, self.side, 1)
        y_coordinates = np.arange(0, self.side, 1)
        z_coordinates = np.arange(0, self.side, 1)

        x_values, y_values, z_values = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)
        points = np.vstack((x_values.flatten(), y_values.flatten(), z_values.flatten()))
        points = points.T

        return points @ self.translation_generators

    def generate_neighbors(self):
        x_coordinates = (self.side - 1) * np.linspace(-1, 1, 2 * self.side - 1)
        y_coordinates = (self.side - 1) * np.linspace(-1, 1, 2 * self.side - 1)
        z_coordinates = (self.side - 1) * np.linspace(-1, 1, 2 * self.side - 1)

        x_values, y_values, z_values = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)
        neighbors = np.vstack((x_values.flatten(), y_values.flatten(), z_values.flatten()))
        return neighbors.T

    def apply_translation(self, point, translation):
        new_point = point + translation
        if np.any(np.all(np.isclose(new_point, self.points), 1), 0):
            return new_point
        vectors = self.generate_neighbors()
        supercell_vectors = self.supercell @ self.translation_generators
        vectors = vectors @ supercell_vectors
        neighbors = []
        tentative_new_points = new_point - vectors

        for vector in vectors:
            for point2 in self.points:
                tentative_new_point = new_point - vector
                if np.all(np.isclose(tentative_new_point, point2)):
                    new_point2 = point2
                    neighbors.append(vector)
        if len(neighbors) == 0:
            print(f"point {new_point} not in original cluster")
        if len(neighbors) > 1:
            print(f"point {new_point} exists in multiple copies in original cluster")

        return new_point2

    def map_cluster(self):
        def mapping(point):
            new_point = self.apply_translation(point, np.zeros(3))
            for i, cluster_point in enumerate(self.points):
                if np.all(np.isclose(new_point, cluster_point)):
                    return i

        return mapping


class SquareCluster(Cluster):
    def __init__(self, side):
        super(SquareCluster, self).__init__(side)
        self.points, self.max_coordinate = self.build_cluster()
        self.unit_lenght = 1 if (self.side % 2 == 1) else 2
        self.translation_generators = np.array([[1, 0], [0, 1]])

    def build_cluster(self):
        max_coordinate = int(self.side / 2) if (self.side % 2 == 1) else self.side - 1
        x_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        y_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)

        x_values, y_values = np.meshgrid(x_coordinates, y_coordinates)
        points = np.vstack((x_values.flatten(), y_values.flatten()))
        points = points.T

        return points, max_coordinate


class HexagonalCluster(Cluster):
    def __init__(self, side):
        super(HexagonalCluster, self).__init__(side)
        self.points, self.max_coordinate = self.build_cluster()
        self.unit_lenght = 1
        self.size = self.points.shape[0]
        self.translation_generators = np.array([[1, 0], [0.5, 3 ** 0.5 / 2]])

    def build_cluster(self):
        diameter = self.side * 2 - 1
        max_coordinate = self.side - 1
        coordinates_increment = np.array([0.5, 3 ** 0.5 / 2])
        horizontal_coordinates = np.zeros((diameter, 2))
        horizontal_coordinates[:, 0] = np.linspace(-max_coordinate, max_coordinate, diameter)
        points = np.array(horizontal_coordinates)
        for i in range(max_coordinate):
            step = i + 1
            up_coordinates = horizontal_coordinates + step * coordinates_increment
            up_coordinates = up_coordinates[: diameter - step]
            points = np.vstack((points, up_coordinates))
            down_coordinates = horizontal_coordinates - step * coordinates_increment
            down_coordinates = down_coordinates[step:diameter]
            points = np.vstack((points, down_coordinates))

        return points, max_coordinate

    def apply_translation(self, point, translation):
        new_point = np.array(point)

        new_point += translation
        if not np.any(np.all(np.isclose(new_point, self.points), 1), 0):
            size = self.side - 1
            vectors = [
                self.translation_generators[0] * (2 * size + 1) - self.translation_generators[1] * self.side,
                self.translation_generators[0] * (size + 1) + self.translation_generators[1] * size,
                -self.translation_generators[0] * size + self.translation_generators[1] * (2 * size + 1),
            ]
            in_original = 0
            for vector in vectors:
                for point2 in self.points:
                    tentative_new_point = new_point - vector
                    if np.all(np.isclose(tentative_new_point, point2)):
                        new_point = point2
                        in_original += 1
            if in_original == 0:
                print(f"point {new_point} not in original cluster")
            if in_original > 1:
                print(f"point {new_point} exists in multiple copies in original cluster")

        return new_point


class CubicCluster(Cluster):
    def __init__(self, side):
        super(CubicCluster, self).__init__(side)
        self.unit_lenght = 1 if (self.side % 2 == 1) else 2
        self.dimension = 3
        self.translation_generators = self.unit_lenght * np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        self.conventional_supercell = self.side * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.points = self.build_prism_cluster()
        self.size = self.points.shape[0]


class CCCubicCluster(Cluster):
    def __init__(self, side):
        super(CCCubicCluster, self).__init__(side)
        self.unit_lenght = 1 if (self.side % 2 == 1) else 2
        self.dimension = 3
        self.translation_generators = self.unit_lenght * np.array(
            np.array(
                [
                    [0.5, 0.5, 0],
                    [0.5, -0.5, 0],
                    [0, 0, 1],
                ]
            )
        )
        self.conventional_supercell = np.array(
            [
                [self.side, self.side - 1, 0],
                [-self.side + 1, self.side, 0],
                [0, 0, self.side],
            ]
        )
        self.points = self.build_prism_cluster()
        self.size = self.points.shape[0]

    def build_cluster(self):
        max_coordinate = int(self.side / 2) if (self.side % 2 == 1) else self.side - 1
        x_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        y_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        z_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)

        bases_max_coordinate = max_coordinate - self.unit_lenght / 2
        bases_x_coordinates = np.linspace(-bases_max_coordinate, bases_max_coordinate, self.side - 1)
        bases_y_coordinates = np.linspace(-bases_max_coordinate, bases_max_coordinate, self.side - 1)

        x_values, y_values, z_values = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)
        points = np.vstack((x_values.flatten(), y_values.flatten(), z_values.flatten()))
        points = points.T

        base_x_values, base_y_values, base_z_values = np.meshgrid(
            bases_x_coordinates, bases_y_coordinates, z_coordinates
        )
        base_points = np.vstack((base_x_values.flatten(), base_y_values.flatten(), base_z_values.flatten()))
        base_points = base_points.T

        points = np.concatenate((points, base_points))

        return points, max_coordinate


class BCCubicCluster(Cluster):
    def __init__(self, side):
        super(BCCubicCluster, self).__init__(side)
        self.unit_lenght = 1 if (self.side % 2 == 1) else 2
        self.dimension = 3
        self.translation_generators = self.unit_lenght * np.array(
            np.array(
                [
                    [0.5, 0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ]
            )
        )
        self.conventional_supercell = np.array(
            [
                [self.side, self.side - 1, 0],
                [0, self.side, self.side - 1],
                [self.side - 1, 0, self.side],
            ]
        )
        self.points = self.build_prism_cluster()
        self.size = self.points.shape[0]

    def build_cluster(self):
        max_coordinate = int(self.side / 2) if (self.side % 2 == 1) else self.side - 1
        x_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        y_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        z_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)

        bases_max_coordinate = max_coordinate - self.unit_lenght / 2
        bases_x_coordinates = np.linspace(-bases_max_coordinate, bases_max_coordinate, self.side - 1)
        bases_y_coordinates = np.linspace(-bases_max_coordinate, bases_max_coordinate, self.side - 1)
        bases_z_coordinates = np.linspace(-bases_max_coordinate, bases_max_coordinate, self.side - 1)

        x_values, y_values, z_values = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)
        points = np.vstack((x_values.flatten(), y_values.flatten(), z_values.flatten()))
        points = points.T

        base_x_values, base_y_values, base_z_values = np.meshgrid(
            bases_x_coordinates, bases_y_coordinates, bases_z_coordinates
        )
        base_points = np.vstack((base_x_values.flatten(), base_y_values.flatten(), base_z_values.flatten()))
        base_points = base_points.T

        points = np.concatenate((points, base_points))

        return points, max_coordinate


class FCCubicCluster(Cluster):
    def __init__(self, side):
        super(FCCubicCluster, self).__init__(side)
        self.unit_lenght = 1 if (self.side % 2 == 1) else 2
        self.dimension = 3
        self.translation_generators = self.unit_lenght * np.array(
            np.array(
                [
                    [0.5, 0.5, 0],
                    [0.5, 0, 0.5],
                    [0, 0.5, 0.5],
                ]
            )
        )
        self.conventional_supercell = np.array(
            [
                [self.side, self.side - 1, -self.side + 1],
                [-self.side + 1, self.side, self.side - 1],
                [self.side - 1, -self.side + 1, self.side],
            ]
        )
        self.points = self.build_prism_cluster()
        self.size = self.points.shape[0]

    def build_cluster(self):
        max_coordinate = int(self.side / 2) if (self.side % 2 == 1) else self.side - 1
        x_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        y_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        z_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)

        faces_max_coordinate = max_coordinate - self.unit_lenght / 2

        x_values, y_values, z_values = np.meshgrid(x_coordinates, y_coordinates, z_coordinates)
        points = np.vstack((x_values.flatten(), y_values.flatten(), z_values.flatten()))
        points = points.T

        axis_coordinates = np.linspace(-max_coordinate, max_coordinate, self.side)
        faces_coordinates = np.linspace(-faces_max_coordinate, faces_max_coordinate, self.side - 1)

        face_x_values, face_y_values, face_z_values = np.meshgrid(
            faces_coordinates, faces_coordinates, axis_coordinates
        )
        face_points = np.vstack((face_x_values.flatten(), face_y_values.flatten(), face_z_values.flatten()))
        face_points = face_points.T

        points = np.concatenate((points, face_points))

        face_x_values, face_y_values, face_z_values = np.meshgrid(
            faces_coordinates, axis_coordinates, faces_coordinates
        )
        face_points = np.vstack((face_x_values.flatten(), face_y_values.flatten(), face_z_values.flatten()))
        face_points = face_points.T

        points = np.concatenate((points, face_points))

        face_x_values, face_y_values, face_z_values = np.meshgrid(
            axis_coordinates, faces_coordinates, faces_coordinates
        )
        face_points = np.vstack((face_x_values.flatten(), face_y_values.flatten(), face_z_values.flatten()))
        face_points = face_points.T

        points = np.concatenate((points, face_points))

        return points, max_coordinate


class HexagonalPrismCluster(Cluster):
    def __init__(self, side):
        super(HexagonalPrismCluster, self).__init__(side)
        self.side = side
        self.unit_lenght = 1 if (self.side % 2 == 1) else 2
        self.dimension = 3
        self.translation_generators = np.array(
            [
                [1, 0, 0],
                [0.5, 3 ** 0.5 / 2, 0],
                [0, 0, 1],
            ]
        )
        self.conventional_supercell = np.array(
            [
                [self.side, self.side - 1, 0],
                [-self.side + 1, 2 * (self.side) - 1, 0],
                [0, 0, self.side],
            ]
        )
        self.points = self.build_prism_cluster()
        self.size = self.points.shape[0]

    def build_cluster(self):
        diameter = self.side * 2 - 1
        max_coordinate = self.side - 1
        coordinates_increment = np.array([0.5, 3 ** 0.5 / 2])
        horizontal_coordinates = np.zeros((diameter, 2))
        horizontal_coordinates[:, 0] = np.linspace(-max_coordinate, max_coordinate, diameter)
        plane_points = np.array(horizontal_coordinates)
        for i in range(max_coordinate):
            step = i + 1
            up_coordinates = horizontal_coordinates + step * coordinates_increment
            up_coordinates = up_coordinates[: diameter - step]
            plane_points = np.vstack((plane_points, up_coordinates))
            down_coordinates = horizontal_coordinates - step * coordinates_increment
            down_coordinates = down_coordinates[step:diameter]
            plane_points = np.vstack((plane_points, down_coordinates))

        max_z_coordinate = int(self.side / 2) if (self.side % 2 == 1) else self.side - 1
        z_coordinates = np.linspace(-max_z_coordinate, max_z_coordinate, self.side)
        points = np.zeros((plane_points.shape[0] * self.side, 3))
        for i, z_coordinate in enumerate(z_coordinates):
            for j, point in enumerate(plane_points):
                points[i * plane_points.shape[0] + j, :] = np.concatenate((point, z_coordinate[None]))

        return points, max_z_coordinate


class BravaisLattice:
    def __init__(self, side):
        self.cluster = Cluster(side)
        self.point_generators = []
        self.point_group_elements = []
        self.group_elements = self.generate_group_elements()

    def generate_group_elements(self):
        group_elements = []
        for point_element in self.point_group_elements:
            for point in self.cluster.points:
                group_elements.append((point, point_element))

        return group_elements

    def generate_first_layer_permutations(self):
        permutations = []
        for point_element in self.point_generators:
            transformed_points = [
                self.cluster.apply_translation(point_element @ point, np.zeros(3)) for point in self.cluster.points
            ]
            permutation_indices = self.find_permutation(self.cluster.points, transformed_points)
            permutations.append(permutation_indices)

        for translation in self.cluster.translation_generators:
            transformed_points = [self.cluster.apply_translation(point, translation) for point in self.cluster.points]
            permutation_indices = self.find_permutation(self.cluster.points, transformed_points)
            permutations.append(permutation_indices)

        return permutations

    def generate_permutations(self):
        permutations = []
        for point_element2 in self.point_generators:
            transformed_elements = [
                (
                    self.cluster.apply_translation(point_element2 @ translation, np.zeros(3)),
                    point_element2 @ point_element,
                )
                for translation, point_element in self.group_elements
            ]
            permutation_indices = self.find_elements_permutation(transformed_elements)
            permutations.append(permutation_indices)

        for translation2 in self.cluster.translation_generators:
            transformed_elements = [
                (
                    self.cluster.apply_translation(translation, translation2),
                    point_element,
                )
                for translation, point_element in self.group_elements
            ]
            permutation_indices = self.find_elements_permutation(transformed_elements)
            permutations.append(permutation_indices)

        return permutations

    def find_elements_permutation(self, transformed_elements):
        flattened_group_elements = [self.flatten_group_element(group_element) for group_element in self.group_elements]
        flattened_transformed_elements = [
            self.flatten_group_element(transformed_element) for transformed_element in transformed_elements
        ]

        permutation_indices = self.find_permutation(flattened_group_elements, flattened_transformed_elements)
        return permutation_indices

    @staticmethod
    def flatten_group_element(group_element):
        flattened_element = np.concatenate((group_element[0], group_element[1].flatten()))
        return flattened_element

    @staticmethod
    def find_permutation(original, permuted):
        permutation = []
        for element in permuted:
            in_original = 0
            for i, element2 in enumerate(original):
                if np.all(np.isclose(element, element2)):
                    in_original += 1
                    permutation.append(i)
            if in_original == 0:
                print(f"element {element} not in original array")
            if in_original > 1:
                print(f"element {element} exists in multiple copies in original array")

        return permutation


class RectangularLattice(BravaisLattice):
    def __init__(self, side):
        super(RectangularLattice, self).__init__(side)
        self.cluster = SquareCluster(side)
        self.point_generators = [
            D4_GROUP_ELEMENTS["reflection_x"],
            D4_GROUP_ELEMENTS["reflection_y"],
        ]
        self.point_group_elements = [
            D4_GROUP_ELEMENTS["identity"],
            D4_GROUP_ELEMENTS["rotation_180"],
            D4_GROUP_ELEMENTS["reflection_x"],
            D4_GROUP_ELEMENTS["reflection_y"],
        ]
        self.group_elements = self.generate_group_elements()


class SquareLattice(BravaisLattice):
    def __init__(self, side):
        super(SquareLattice, self).__init__(side)
        self.cluster = SquareCluster(side)
        self.point_generators = D4_GENERATORS.values()
        self.point_group_elements = D4_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class HexagonalLattice(BravaisLattice):
    def __init__(self, side):
        super(HexagonalLattice, self).__init__(side)
        self.cluster = HexagonalCluster(side)
        self.point_generators = D6_GENERATORS.values()
        self.point_group_elements = D6_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveCubicLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveCubicLattice, self).__init__(side)
        self.cluster = CubicCluster(side)
        self.point_generators = OH_GENERATORS.values()
        self.point_group_elements = OH_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveTetragonalLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveTetragonalLattice, self).__init__(side)
        self.cluster = CubicCluster(side)
        self.point_generators = D4H_GENERATORS.values()
        self.point_group_elements = D4H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveOrthorhombicLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveOrthorhombicLattice, self).__init__(side)
        self.cluster = CubicCluster(side)
        self.point_generators = D2H_GENERATORS.values()
        self.point_group_elements = D2H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveMonoclinicLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveMonoclinicLattice, self).__init__(side)
        self.cluster = CubicCluster(side)
        self.point_generators = C2H_GENERATORS.values()
        self.point_group_elements = C2H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveTriclinicLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveTriclinicLattice, self).__init__(side)
        self.cluster = CubicCluster(side)
        self.point_generators = CI_GENERATORS.values()
        self.point_group_elements = CI_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveRhombohedralLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveRhombohedralLattice, self).__init__(side)
        self.cluster = CubicCluster(side)
        self.point_generators = D3D_GENERATORS.values()
        self.point_group_elements = D3D_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class PrimitiveHexagonalLattice(BravaisLattice):
    def __init__(self, side):
        super(PrimitiveHexagonalLattice, self).__init__(side)
        self.cluster = HexagonalPrismCluster(side)
        self.point_generators = D6H_GENERATORS.values()
        self.point_group_elements = D6H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class CMonoclinicLattice(BravaisLattice):
    def __init__(self, side):
        super(CMonoclinicLattice, self).__init__(side)
        self.cluster = CCCubicCluster(side)
        self.point_generators = C2H_GENERATORS.values()
        self.point_group_elements = C2H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class COrthorhombicLattice(BravaisLattice):
    def __init__(self, side):
        super(COrthorhombicLattice, self).__init__(side)
        self.cluster = CCCubicCluster(side)
        self.point_generators = D2H_GENERATORS.values()
        self.point_group_elements = D2H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class BOrthorhombicLattice(BravaisLattice):
    def __init__(self, side):
        super(BOrthorhombicLattice, self).__init__(side)
        self.cluster = BCCubicCluster(side)
        self.point_generators = D2H_GENERATORS.values()
        self.point_group_elements = D2H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class BTetragonalLattice(BravaisLattice):
    def __init__(self, side):
        super(BTetragonalLattice, self).__init__(side)
        self.cluster = BCCubicCluster(side)
        self.point_generators = D4H_GENERATORS.values()
        self.point_group_elements = D4H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class BCubicLattice(BravaisLattice):
    def __init__(self, side):
        super(BCubicLattice, self).__init__(side)
        self.cluster = BCCubicCluster(side)
        self.point_generators = OH_GENERATORS.values()
        self.point_group_elements = OH_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class FOrthorhombicLattice(BravaisLattice):
    def __init__(self, side):
        super(FOrthorhombicLattice, self).__init__(side)
        self.cluster = FCCubicCluster(side)
        self.point_generators = D2H_GENERATORS.values()
        self.point_group_elements = D2H_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class FCubicLattice(BravaisLattice):
    def __init__(self, side):
        super(FCubicLattice, self).__init__(side)
        self.cluster = FCCubicCluster(side)
        self.point_generators = OH_GENERATORS.values()
        self.point_group_elements = OH_GROUP_ELEMENTS.values()
        self.group_elements = self.generate_group_elements()


class SetLattice(PrimitiveCubicLattice):
    def __init__(self, side):
        super(SetLattice, self).__init__(side)

    def generate_first_layer_permutations(self):
        permutations = super().generate_first_layer_permutations()
        symmetric_generator = list(range(max(permutations[0]) + 1))
        symmetric_generator[0], symmetric_generator[1] = (
            symmetric_generator[1],
            symmetric_generator[0],
        )
        permutations.append(symmetric_generator)

        return permutations

    def generate_permutations(self):
        return None


LATTICE_NAMES = {
    "ptriclinic": PrimitiveTriclinicLattice(2),
    "pmonoclinic": PrimitiveMonoclinicLattice(2),
    "cmonoclinic": CMonoclinicLattice(2),
    "porthorhombic": PrimitiveOrthorhombicLattice(2),
    "corthorhombic": COrthorhombicLattice(2),
    "borthorhombic": BOrthorhombicLattice(2),
    "forthorhombic": FOrthorhombicLattice(2),
    "ptetragonal": PrimitiveTetragonalLattice(2),
    "btetragonal": BTetragonalLattice(2),
    "prhombohedral": PrimitiveRhombohedralLattice(2),
    "phexagonal": PrimitiveHexagonalLattice(2),
    "pcubic": PrimitiveCubicLattice(2),
    "bcubic": BCubicLattice(2),
    "fcubic": FCubicLattice(2),
    "set": SetLattice(2),
}


def get_bravais_lattice(structure):
    short_name, number = structure.get_space_group_info()
    if number < 3:
        bravais_lattice = PrimitiveTriclinicLattice(2)
        lattice_string = "ptriclinic"
    elif 2 < number < 16:
        if short_name[0] == "P":
            bravais_lattice = PrimitiveMonoclinicLattice(2)
            lattice_string = "pmonoclinic"
        elif short_name[0] == "C":
            bravais_lattice = CMonoclinicLattice(2)
            lattice_string = "cmonoclinic"
    elif 15 < number < 75:
        if short_name[0] == "P":
            bravais_lattice = PrimitiveOrthorhombicLattice(2)
            lattice_string = "porthorhombic"
        elif short_name[0] == "C" or short_name[0] == "A":
            bravais_lattice = COrthorhombicLattice(2)
            lattice_string = "corthorhombic"
        elif short_name[0] == "I":
            bravais_lattice = BOrthorhombicLattice(2)
            lattice_string = "borthorhombic"
        elif short_name[0] == "F":
            bravais_lattice = FOrthorhombicLattice(2)
            lattice_string = "forthorhombic"
    elif 74 < number < 143:
        if short_name[0] == "P":
            bravais_lattice = PrimitiveTetragonalLattice(2)
            lattice_string = "ptetragonal"
        elif short_name[0] == "I":
            bravais_lattice = BTetragonalLattice(2)
            lattice_string = "btetragonal"
    elif 142 < number < 168:
        if short_name[0] == "R":
            bravais_lattice = PrimitiveRhombohedralLattice(2)
            lattice_string = "prhombohedral"
        if short_name[0] == "P":
            bravais_lattice = PrimitiveHexagonalLattice(2)
            lattice_string = "phexagonal"
    elif 167 < number < 195:
        bravais_lattice = PrimitiveHexagonalLattice(2)
        lattice_string = "phexagonal"
    elif 194 < number:
        if short_name[0] == "P":
            bravais_lattice = PrimitiveCubicLattice(2)
            lattice_string = "pcubic"
        elif short_name[0] == "I":
            bravais_lattice = BCubicLattice(2)
            lattice_string = "bcubic"
        elif short_name[0] == "F":
            bravais_lattice = FCubicLattice(2)
            lattice_string = "fcubic"

    return bravais_lattice, lattice_string


def build_generators(lattice, option="restricted"):
    print(f"Building layer for lattice {lattice}")
    bravais_lattice = LATTICE_NAMES[lattice]
    if option == "restricted":
        generators = bravais_lattice.generate_first_layer_permutations()
    elif option == "full":
        generators = bravais_lattice.generate_permutations()
    else:
        raise ValueError("Type must be one of 'restricted' or 'full'.")

    return generators


def save_generators(output_path):
    generators = {}
    for lattice in LATTICE_NAMES.keys():
        generators[lattice] = {option: build_generators(lattice, option) for option in ["restricted", "full"]}

    utils.save_json(generators, output_path)


def main():
    LAYER_GENERATORS = utils.SRC_PATH / "equivariant" / "layer_generators_prism.json"

    # for lattice_name, lattice in LATTICE_NAMES.items():
    #     print(
    #         f"Generators for {lattice_name}: {lattice.generate_permutations()}"
    #     )
    # save_generators(LAYER_GENERATORS)
    # lattice = BOrthorhombicLattice(2)
    # lattice.cluster.plot_grid()

    # lattice = RectangularLattice(2)
    # print(lattice.generate_permutations())
    # print(lattice.generate_first_layer_permutations())

    # lattice = RectangularLattice(2)
    # print(lattice.generate_permutations())
    # print(lattice.generate_first_layer_permutations())

    # lattice = HexagonalLattice(2)
    # print(lattice.generate_permutations())
    # print(lattice.generate_first_layer_permutations())

    # lattice = PrimitiveHexagonalLattice(2)
    # print(lattice.generate_permutations())
    # print(lattice.generate_first_layer_permutations())
    # lattice.cluster.plot_grid()

    lattice = PrimitiveHexagonalLattice(2)
    lattice.cluster.plot_grid()

    # lattice = PrimitiveTriclinicLattice(2)
    # print(len(lattice.cluster.points))
    # lattice.cluster.plot_grid()

    # permutations = lattice.generate_permutations()
    first_layer_permutations = lattice.generate_first_layer_permutations()

    # print(permutations)
    # print(first_layer_permutations)
    # parameters1 = np.random.rand(8)
    # parameters2 = np.random.rand(8)
    # colored_vector = create_colored_vector(first_layer_permutations)
    # first_colored_matrix = create_colored_matrix(first_layer_permutations, permutations)
    intermediate_colored_matrix = create_colored_matrix(first_layer_permutations, first_layer_permutations)
    # last_colored_matrix = create_colored_matrix(permutations, first_layer_permutations)

    # restricted = create_colored_matrix(
    #     [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
    #     [[1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
    # )
    # set_dict = {(0, 0): 1, (0, 1): 5, (1, 0): 5, (1, 1): 1}

    # def dict_to_matrix(dictionary, size):
    #     matrix = np.zeros((size, size))
    #     for key, value in dictionary.items():
    #         matrix[key[0]][key[1]] = value

    #     return matrix

    # p2_matrix = dict_to_matrix(restricted, 4) + 1
    # set_matrix = dict_to_matrix(set_dict, 2)

    # total_matrix = np.kron(p2_matrix, set_matrix)
    # u, inv = np.unique(total_matrix, return_inverse=True)
    # new_matrix = np.arange(len(u))[inv].reshape(total_matrix.shape)

    # adj_matrix = [
    #     [1, 1, 0, 1, 0, 1, 0, 0],
    #     [1, 1, 1, 0, 1, 0, 0, 0],
    #     [0, 1, 1, 1, 0, 0, 0, 1],
    #     [1, 0, 1, 1, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 1, 1, 0, 1],
    #     [1, 0, 0, 0, 1, 1, 1, 0],
    #     [0, 0, 0, 1, 0, 1, 1, 1],
    #     [0, 0, 1, 0, 1, 0, 1, 1],
    # ]

    # def matrix_to_dict(matrix):
    #     dictionary = {}
    #     for i, row in enumerate(matrix):
    #         for j, element in enumerate(row):
    #             dictionary[(i, j)] = element

    #     return dictionary

    # total_dict = matrix_to_dict(p2_matrix)

    # first_matrix = dict_to_matrix(first_colored_matrix, True)
    # intermediate_matrix = dict_to_matrix(intermediate_colored_matrix, True)
    # last_matrix = dict_to_matrix(last_colored_matrix, True)

    # matrix = (
    #     parameters1[first_matrix.astype(int)]
    #     @ (intermediate_matrix)
    #     @ parameters2[last_matrix.astype(int)]
    # )
    # print((set(np.unique(first_colored_matrix))))

    draw_colored_matrix(8, 8, intermediate_colored_matrix)
    # draw_colored_vector(8, colored_vector)

    # test_generators(D3D_GENERATORS, 8)

    # layer = LinearEquivDepth(
    #     first_layer_permutations, first_layer_permutations, 10, 10, fan="features"
    # )
    # x = torch.ones((41, 10, 8))
    # y = layer.forward(x)


if __name__ == "__main__":
    main()
