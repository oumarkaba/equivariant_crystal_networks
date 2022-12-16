import numpy as np

"""https://github.com/watchduck/full_octahedral_group/blob/master/projects/p03_subgroups/store_dicts.py"""

# 2D group matrices
D4_GENERATORS = {
    "rotation_90": np.array([[0, -1], [1, 0]]),
    "reflection_x": np.array([[-1, 0], [0, 1]]),
}

D4_GROUP_ELEMENTS = {
    "identity": np.eye(2),
    "rotation_90": D4_GENERATORS["rotation_90"],
    "rotation_180": D4_GENERATORS["rotation_90"] @ D4_GENERATORS["rotation_90"],
    "rotation_270": np.linalg.matrix_power(D4_GENERATORS["rotation_90"], 3),
    "reflection_x": D4_GENERATORS["reflection_x"],
    "reflection_anti": D4_GENERATORS["rotation_90"] @ D4_GENERATORS["reflection_x"],
    "reflection_y": np.linalg.matrix_power(D4_GENERATORS["rotation_90"], 2)
    @ D4_GENERATORS["reflection_x"],
    "reflection_diag": np.linalg.matrix_power(D4_GENERATORS["rotation_90"], 3)
    @ D4_GENERATORS["reflection_x"],
}

D6_GENERATORS = {
    "r_1": np.array(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3)],
            [np.sin(np.pi / 3), np.cos(np.pi / 3)],
        ]
    ),
    "f": np.array([[-1, 0], [0, 1]]),
}

D6_GROUP_ELEMENTS = {
    "e": np.eye(2),
    "r_1": D6_GENERATORS["r_1"],
    "r_2": D6_GENERATORS["r_1"] @ D6_GENERATORS["r_1"],
    "r_3": np.linalg.matrix_power(D6_GENERATORS["r_1"], 3),
    "r_4": np.linalg.matrix_power(D6_GENERATORS["r_1"], 4),
    "r_5": np.linalg.matrix_power(D6_GENERATORS["r_1"], 5),
    "f": D6_GENERATORS["f"],
    "rf": D6_GENERATORS["r_1"] @ D6_GENERATORS["f"],
    "r_2f": np.linalg.matrix_power(D6_GENERATORS["r_1"], 2) @ D6_GENERATORS["f"],
    "r_3f": np.linalg.matrix_power(D6_GENERATORS["r_1"], 3) @ D6_GENERATORS["f"],
    "r_4f": np.linalg.matrix_power(D6_GENERATORS["r_1"], 4) @ D6_GENERATORS["f"],
    "r_5f": np.linalg.matrix_power(D6_GENERATORS["r_1"], 5) @ D6_GENERATORS["f"],
}

# 3D group matrices
OH_GROUP_ELEMENTS = {
    (0, 0): np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (0, 1): np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    (0, 2): np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    (0, 3): np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
    (0, 4): np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    (0, 5): np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    (1, 0): np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (1, 1): np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    (1, 2): np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    (1, 3): np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]]),
    (1, 4): np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
    (1, 5): np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
    (2, 0): np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    (2, 1): np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    (2, 2): np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    (2, 3): np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]),
    (2, 4): np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]),
    (2, 5): np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    (3, 0): np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    (3, 1): np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
    (3, 2): np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    (3, 3): np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    (3, 4): np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
    (3, 5): np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]]),
    (4, 0): np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    (4, 1): np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
    (4, 2): np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    (4, 3): np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
    (4, 4): np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]]),
    (4, 5): np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    (5, 0): np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    (5, 1): np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
    (5, 2): np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    (5, 3): np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
    (5, 4): np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
    (5, 5): np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
    (6, 0): np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    (6, 1): np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
    (6, 2): np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]),
    (6, 3): np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
    (6, 4): np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
    (6, 5): np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]),
    (7, 0): np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    (7, 1): np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
    (7, 2): np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
    (7, 3): np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]]),
    (7, 4): np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
    (7, 5): np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
}

OH_GENERATORS = {
    (4, 0): OH_GROUP_ELEMENTS[(4, 0)],
    (0, 1): OH_GROUP_ELEMENTS[(0, 1)],
    (0, 3): OH_GROUP_ELEMENTS[(0, 3)],
}

D4H_GENERATORS = {
    (4, 0): OH_GROUP_ELEMENTS[(4, 0)],
    (0, 1): OH_GROUP_ELEMENTS[(0, 1)],
    (1, 0): OH_GROUP_ELEMENTS[(1, 0)],
}

D4H_GROUP_ELEMENTS = {
    (0, 0): OH_GROUP_ELEMENTS[(0, 0)],
    (0, 1): OH_GROUP_ELEMENTS[(0, 1)],
    (3, 1): OH_GROUP_ELEMENTS[(3, 1)],
    (3, 0): OH_GROUP_ELEMENTS[(3, 0)],
    (5, 0): OH_GROUP_ELEMENTS[(5, 0)],
    (5, 1): OH_GROUP_ELEMENTS[(5, 1)],
    (6, 1): OH_GROUP_ELEMENTS[(6, 1)],
    (6, 0): OH_GROUP_ELEMENTS[(6, 0)],
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
    (7, 1): OH_GROUP_ELEMENTS[(7, 1)],
    (4, 1): OH_GROUP_ELEMENTS[(4, 1)],
    (4, 0): OH_GROUP_ELEMENTS[(4, 0)],
    (2, 0): OH_GROUP_ELEMENTS[(2, 0)],
    (2, 1): OH_GROUP_ELEMENTS[(2, 1)],
    (1, 1): OH_GROUP_ELEMENTS[(1, 1)],
    (1, 0): OH_GROUP_ELEMENTS[(1, 0)],
}

D2H_GENERATORS = {
    (4, 0): OH_GROUP_ELEMENTS[(4, 0)],
    (2, 0): OH_GROUP_ELEMENTS[(2, 0)],
    (1, 0): OH_GROUP_ELEMENTS[(1, 0)],
}

D2H_GROUP_ELEMENTS = {
    (0, 0): OH_GROUP_ELEMENTS[(0, 0)],
    (3, 0): OH_GROUP_ELEMENTS[(3, 0)],
    (5, 0): OH_GROUP_ELEMENTS[(5, 0)],
    (6, 0): OH_GROUP_ELEMENTS[(6, 0)],
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
    (4, 0): OH_GROUP_ELEMENTS[(4, 0)],
    (2, 0): OH_GROUP_ELEMENTS[(2, 0)],
    (1, 0): OH_GROUP_ELEMENTS[(1, 0)],
}

C2H_GENERATORS = {
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
    (1, 0): OH_GROUP_ELEMENTS[(1, 0)],
}

C2H_GROUP_ELEMENTS = {
    (0, 0): OH_GROUP_ELEMENTS[(0, 0)],
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
    (1, 0): OH_GROUP_ELEMENTS[(1, 0)],
    (6, 0): OH_GROUP_ELEMENTS[(6, 0)],
}

CI_GENERATORS = {
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
}

CI_GROUP_ELEMENTS = {
    (0, 0): OH_GROUP_ELEMENTS[(0, 0)],
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
}

D3D_GENERATORS = {
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
    (0, 2): OH_GROUP_ELEMENTS[(0, 2)],
    (0, 3): OH_GROUP_ELEMENTS[(0, 3)],
}

D3D_GROUP_ELEMENTS = {
    (0, 0): OH_GROUP_ELEMENTS[(0, 0)],
    (0, 1): OH_GROUP_ELEMENTS[(0, 1)],
    (0, 2): OH_GROUP_ELEMENTS[(0, 2)],
    (0, 3): OH_GROUP_ELEMENTS[(0, 3)],
    (0, 4): OH_GROUP_ELEMENTS[(0, 4)],
    (0, 5): OH_GROUP_ELEMENTS[(0, 5)],
    (7, 0): OH_GROUP_ELEMENTS[(7, 0)],
    (7, 1): OH_GROUP_ELEMENTS[(7, 1)],
    (7, 2): OH_GROUP_ELEMENTS[(7, 2)],
    (7, 3): OH_GROUP_ELEMENTS[(7, 3)],
    (7, 4): OH_GROUP_ELEMENTS[(7, 4)],
    (7, 5): OH_GROUP_ELEMENTS[(7, 5)],
}

D6H_GENERATORS = {
    "r_1": np.array(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0],
            [np.sin(np.pi / 3), np.cos(np.pi / 3), 0],
            [0, 0, 1],
        ]
    ),
    "f": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "h": np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
}

D6H_GROUP_ELEMENTS = {
    "e": np.eye(3),
    "r_1": D6H_GENERATORS["r_1"],
    "r_2": D6H_GENERATORS["r_1"] @ D6H_GENERATORS["r_1"],
    "r_3": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 3),
    "r_4": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 4),
    "r_5": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 5),
    "f": D6H_GENERATORS["f"],
    "rf": D6H_GENERATORS["r_1"] @ D6H_GENERATORS["f"],
    "r_2f": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 2) @ D6H_GENERATORS["f"],
    "r_3f": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 3) @ D6H_GENERATORS["f"],
    "r_4f": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 4) @ D6H_GENERATORS["f"],
    "r_5f": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 5) @ D6H_GENERATORS["f"],
    "h": D6H_GENERATORS["h"],
    "r_1h": D6H_GENERATORS["r_1"] @ D6H_GENERATORS["h"],
    "r_2h": D6H_GENERATORS["r_1"] @ D6H_GENERATORS["r_1"] @ D6H_GENERATORS["h"],
    "r_3h": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 3) @ D6H_GENERATORS["h"],
    "r_4h": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 4) @ D6H_GENERATORS["h"],
    "r_5h": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 5) @ D6H_GENERATORS["h"],
    "fh": D6H_GENERATORS["f"] @ D6H_GENERATORS["h"],
    "rfh": D6H_GENERATORS["r_1"] @ D6H_GENERATORS["f"] @ D6H_GENERATORS["h"],
    "r_2fh": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 2)
    @ D6H_GENERATORS["f"]
    @ D6H_GENERATORS["h"],
    "r_3fh": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 3)
    @ D6H_GENERATORS["f"]
    @ D6H_GENERATORS["h"],
    "r_4fh": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 4)
    @ D6H_GENERATORS["f"]
    @ D6H_GENERATORS["h"],
    "r_5fh": np.linalg.matrix_power(D6H_GENERATORS["r_1"], 5)
    @ D6H_GENERATORS["f"]
    @ D6H_GENERATORS["h"],
}


def test_generators(generators_dict, max_lenght=4):
    generators = list(generators_dict.values())
    generators_tuples = [tuple(map(tuple, generator)) for generator in generators]
    elements = set(generators_tuples)
    for lenght in range(2, max_lenght):
        generator_lists = [generators for i in range(lenght)]
        combinaisons = itertools.product(*generator_lists)
        for combinaison in combinaisons:
            combinaison = np.array(combinaison)
            element = np.linalg.multi_dot(combinaison)
            element_tuple = tuple(map(tuple, element))
            elements.add(element_tuple)

    print(len(elements))
