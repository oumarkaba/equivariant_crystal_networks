import numpy as np
from scipy import special
from scipy.optimize import brentq
import torch
import torch.nn.functional as F
from learn_materials.equivariant.lattice import LATTICE_NAMES


def swish(x):
    return x * torch.sigmoid(x)


def bessel(x, order):
    return np.sqrt(np.pi / (2 * x)) * special.jv(order + 0.5, x)


def compute_cells_sparse(indices1, indices2, indices_cell):
    # for each index compute cell
    indices1_cells = torch.index_select(indices_cell, 0, indices1)
    indices2_cells = torch.index_select(indices_cell, 0, indices2)
    # for each index compute one hot for cell
    indices2_cells = F.one_hot(indices2_cells)

    return indices1_cells, indices2_cells


# Constants


def bessel_zeros(num_functions, num_zeroes):
    zeros = np.zeros((num_functions, num_zeroes))
    zeros[0] = np.arange(1, num_zeroes + 1) * np.pi
    points = np.arange(1, num_functions + num_zeroes + 1) * np.pi
    for order in range(1, num_functions):
        for root_order in range(num_functions + num_zeroes - order):
            points[root_order] = brentq(
                bessel, points[root_order], points[root_order + 1], (order,)
            )
        zeros[order][:num_zeroes] = points[:num_zeroes]
    return zeros


# Basis


def position_encoding(positions, dimension, max_distance=2.0):
    frequenties = (
        2 * np.pi / torch.pow(max_distance, torch.arange(0, dimension).float())
    )
    frequenties = frequenties.to(positions.device)
    positions = positions.repeat(dimension, 1, 1).permute(1, 2, 0)
    arguments = positions * frequenties
    encodings = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)
    encodings = torch.flatten(encodings, start_dim=1)
    return encodings


def gaussian_basis(
    distance, max_distance, num_centers, width, min_distance=0, to_device=True
):
    centers = torch.linspace(min_distance, max_distance, num_centers)
    if to_device:
        centers = centers.to(distance.device)
    positions = centers - distance
    gaussian_expansion = torch.exp(-positions * positions / (width * width))
    return gaussian_expansion


def bessel_basis(distance, max_distance, num_functions):
    freqs = torch.arange(1, num_functions + 1, 1.0) * np.pi / max_distance
    freqs = freqs.to(distance.device)
    bessel_expansion = np.sqrt(2 / np.pi) * torch.sin(freqs * distance) / distance
    return bessel_expansion


def spherical_harmonics_basis(
    distance, angle, max_distance, number_spherical, number_radial
):
    # FIXME: introduire zeros comme argument de la fonction qui pourra être calculé une seule fois
    device = distance.device
    distance = distance.cpu().numpy()
    angle = angle.cpu().numpy()
    zeros = bessel_zeros(number_spherical, number_radial)
    harmonics = np.arange(number_spherical)
    spherical_harmonics = np.real(special.sph_harm(0, harmonics, 0, angle))
    bessel_values = []
    for i, zeros_order in enumerate(zeros):
        bessel_values.append(bessel(zeros_order / max_distance * distance, i))
    normalization = []
    for i, zeros_order in enumerate(zeros):
        normalization.append(
            1 / (0.5 * bessel(zeros_order, i + 1) ** 2 * max_distance ** 3) ** 0.5
        )
    basis = (
        (
            np.transpose(np.array(bessel_values), (1, 0, 2)) * np.array(normalization)
        ).transpose(2, 0, 1)
        * spherical_harmonics
    ).transpose((1, 0, 2))

    harmonics_basis = torch.FloatTensor(basis).view(
        -1, number_radial * number_spherical
    )
    harmonics_basis = harmonics_basis.to(device)
    return harmonics_basis


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, batch_norm=False):
        super(EmbeddingLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_norm = batch_norm

        self.linear = torch.nn.Linear(self.input_size, self.output_size)
        if self.batch_norm:
            self.batch_norm_layer = torch.nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        embedding = self.linear(x)
        embedding = self.batch_norm_layer(embedding) if self.batch_norm else embedding
        return embedding


def main():
    print(spherical_harmonics_basis(2, 2, 5, 5, 3))


if __name__ == "__main__":
    main()
