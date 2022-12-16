import warnings
from typing import List, Optional, Union, Dict, Any
import math

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from pymatgen import PeriodicSite, Structure, Lattice
from pymatgen.core.structure import StructureError
from pymatgen.util.coord import lattice_points_in_supercell
import pymatgen.analysis.local_env as env
from pymatgen.analysis.local_env import _get_default_radius, _get_radius


class CustomPeriodicSite(PeriodicSite):
    def __init__(self, cell, **kwargs):
        self.cell = cell
        super(CustomPeriodicSite, self).__init__(**kwargs)

    def as_dict(self, verbosity=0):
        "Overloading original method"
        species_list = []
        for spec, occu in self._species.items():
            d = spec.as_dict()
            del d["@module"]
            del d["@class"]
            d["occu"] = occu
            species_list.append(d)

        d = {
            "species": species_list,
            "abc": [float(c) for c in self._frac_coords],
            "cell": self.cell,
            "lattice": self._lattice.as_dict(verbosity=verbosity),
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

        if verbosity > 0:
            d["xyz"] = [float(c) for c in self.coords]
            d["label"] = self.species_string

        d["properties"] = self.properties
        return d


class CustomStructure(Structure):
    def __init__(
        self,
        lattice,
        species,
        coords,
        charge=None,
        validate_proximity=False,
        to_unit_cell=False,
        coords_are_cartesian=False,
        site_properties=None,
        cells=None,
    ):
        if len(species) != len(coords):
            raise StructureError(
                "The list of atomic species must be of the" " same length as the list of fractional" " coordinates."
            )

        if isinstance(lattice, Lattice):
            self._lattice = lattice
        else:
            self._lattice = Lattice(lattice)

        sites = []
        for i, sp in enumerate(species):
            prop = None
            if site_properties:
                prop = {k: v[i] for k, v in site_properties.items()}

            sites.append(
                CustomPeriodicSite(
                    species=sp,
                    coords=coords[i],
                    lattice=self._lattice,
                    to_unit_cell=to_unit_cell,
                    coords_are_cartesian=coords_are_cartesian,
                    properties=prop,
                    cell=cells[i],
                )
            )
        self._sites = list(sites)
        if validate_proximity and not self.is_valid():
            raise StructureError(("Structure contains sites that are ", "less than 0.01 Angstrom apart!"))

    @classmethod
    def from_sites(cls, sites, charge=None, validate_proximity=False, to_unit_cell=False):
        """
        Convenience constructor to make a Structure from a list of sites.

        Args:
            sites: Sequence of PeriodicSites. Sites must have the same
                lattice.
            charge: Charge of structure.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 0.01 Ang apart. Defaults to False.
            to_unit_cell (bool): Whether to translate sites into the unit
                cell.

        Returns:
            (Structure) Note that missing properties are set as None.
        """
        if len(sites) < 1:
            raise ValueError("You need at least one site to construct a %s" % cls)
        prop_keys = []
        props = {}
        lattice = sites[0].lattice
        for i, site in enumerate(sites):
            if site.lattice != lattice:
                raise ValueError("Sites must belong to the same lattice")
            for k, v in site.properties.items():
                if k not in prop_keys:
                    prop_keys.append(k)
                    props[k] = [None] * len(sites)
                props[k][i] = v
        for k, v in props.items():
            if any((vv is None for vv in v)):
                warnings.warn("Not all sites have property %s. Missing values " "are set to None." % k)
        return cls(
            lattice,
            [site.species for site in sites],
            [site.frac_coords for site in sites],
            charge=charge,
            site_properties=props,
            validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell,
            cells=[site.cell for site in sites],
        )

    def __mul__(self, scaling_matrix):
        "Overloading original method"
        scale_matrix = np.array(scaling_matrix, np.int16)
        assert abs(scaling_matrix - scale_matrix).sum() < 1e-5
        if scale_matrix.shape != (3, 3):
            scale_matrix = np.array(scale_matrix * np.eye(3), np.int16)
        new_lattice = Lattice(np.dot(scale_matrix, self._lattice.matrix))

        f_lat = lattice_points_in_supercell(scale_matrix)
        c_lat = new_lattice.get_cartesian_coords(f_lat)

        new_sites = []
        for site in self:
            for i, v in enumerate(c_lat):
                s = CustomPeriodicSite(
                    species=site.species,
                    coords=site.coords + v,
                    lattice=new_lattice,
                    properties=site.properties,
                    coords_are_cartesian=True,
                    to_unit_cell=False,
                    skip_checks=True,
                    cell=f_lat[i],
                )
                new_sites.append(s)

        new_charge = self._charge * np.linalg.det(scale_matrix) if self._charge else None
        return CustomStructure.from_sites(new_sites, charge=new_charge)

    def as_dict(self, verbosity=1, fmt=None, **kwargs):
        "Overloading original method"
        if fmt == "abivars":
            """Returns a dictionary with the ABINIT variables."""
            from pymatgen.io.abinit.abiobjects import structure_to_abivars

            return structure_to_abivars(self, **kwargs)

        latt_dict = self._lattice.as_dict(verbosity=verbosity)
        del latt_dict["@module"]
        del latt_dict["@class"]

        d = {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "charge": self._charge,
            "lattice": latt_dict,
            "sites": [],
        }
        for site in self:
            site_dict = site.as_dict(verbosity=verbosity)
            del site_dict["lattice"]
            del site_dict["@module"]
            del site_dict["@class"]
            d["sites"].append(site_dict)
        return d

    # @classmethod
    # def from_structure(cls, structure):
    #     return cls(**vars(structure))


class CustomVoronoiNN(env.VoronoiNN):
    def get_all_voronoi_polyhedra(self, structure):
        """Get the Voronoi polyhedra for all site in a simulation cell

        Args:
            structure (Structure): Structure to be evaluated
        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """

        # Special case: For atoms with 1 site, the atom in the root image is not
        # included in the get_all_neighbors output. Rather than creating logic to add
        # that atom to the neighbor list, which requires detecting whether it will be
        # translated to reside within the unit cell before neighbor detection, it is
        # less complex to just call the one-by-one operation
        if len(structure) == 1:
            return [self.get_voronoi_polyhedra(structure, 0)]

        # Assemble the list of neighbors used in the tessellation
        if self.targets is None:
            targets = structure.composition.elements
        else:
            targets = self.targets

        # Initialize the list of sites with the atoms in the origin unit cell
        # The `get_all_neighbors` function returns neighbors for each site's image in
        # the original unit cell. We start off with these central atoms to ensure they
        # are included in the tessellation

        if self.to_primitive:
            sites = [x.to_unit_cell() for x in structure]
        else:
            sites = [x for x in structure]
        indices = [(i, 0, 0, 0) for i, _ in enumerate(structure)]

        # Get all neighbors within a certain cutoff
        #   Record both the list of these neighbors, and the site indices
        all_neighs = structure.get_all_neighbors(self.cutoff, include_index=True, include_image=True, sites=sites)
        for neighs in all_neighs:
            sites.extend([x[0] for x in neighs])
            indices.extend([(x[2],) + x[3] for x in neighs])

        # Get the non-duplicates (using the site indices for numerical stability)
        if self.to_primitive:
            sites_frac_coords = [site.frac_coords for site in sites]
            sites_frac_coords = np.array(sites_frac_coords)
            _, uniq_inds = np.unique(sites_frac_coords, return_index=True, axis=0)
            sites = [sites[i] for i in sorted(uniq_inds)]
        else:
            indices = np.array(indices, dtype=np.int)
            indices, uniq_inds = np.unique(indices, return_index=True, axis=0)
            sites = [sites[i] for i in uniq_inds]

        # Sort array such that atoms in the root image are first
        #   Exploit the fact that the array is sorted by the unique operation such that
        #   the images associated with atom 0 are first, followed by atom 1, etc.
        # (root_images,) = np.nonzero(np.abs(indices[:, 1:]).max(axis=1) == 0)
        root_images = np.arange(len(structure))

        del indices  # Save memory (tessellations can be costly)

        ######
        # root_sites = [sites[root_image] for root_image in root_images]
        # root_sites_coords = [site.frac_coords for site in root_sites]
        # root_sites_coords = np.array(root_sites_coords)
        # x_root = root_sites_coords.T[0]
        # y_root = root_sites_coords.T[1]
        # z_root = root_sites_coords.T[2]

        # sites_frac_coords = [site.frac_coords for site in sites]
        # sites_frac_coords = np.array(sites_frac_coords)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # x_coordinates = sites_frac_coords.T[0]
        # y_coordinates = sites_frac_coords.T[1]
        # z_coordinates = sites_frac_coords.T[2]
        # ax.scatter(x_coordinates, y_coordinates, z_coordinates)
        # ax.scatter(x_root, y_root, z_root, c="r", s=100)
        # for i, _ in enumerate(sites):
        #     ax.text(x_coordinates[i], y_coordinates[i], z_coordinates[i], '%s' % (str(i)))

        # plt.show()
        ######

        # Run the tessellation
        qvoronoi_input = [s.coords for s in sites]
        voro = Voronoi(qvoronoi_input)

        # Get the information for each neighbor
        return [
            self._extract_cell_info(structure, i, sites, targets, voro, self.compute_adj_neighbors)
            for i in root_images.tolist()
        ]

    def get_all_nn_info(self, structure: Structure) -> List[List[Dict[str, Any]]]:
        """
        Args:
            structure (Structure): input structure.

        Returns:
            List of near neighbor information for each site. See get_nn_info for the
            format of the data for each site.
        """
        if self.to_primitive:
            for site in structure:
                site.to_unit_cell(in_place=True)
        all_nns = self.get_all_voronoi_polyhedra(structure)
        return [self._filter_nns(structure, n, nns) for n, nns in enumerate(all_nns)]


class CustomIsayevNN(env.IsayevNN):
    def get_all_voronoi_polyhedra(self, structure):
        """Get the Voronoi polyhedra for all site in a simulation cell

        Args:
            structure (Structure): Structure to be evaluated
        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """

        # Special case: For atoms with 1 site, the atom in the root image is not
        # included in the get_all_neighbors output. Rather than creating logic to add
        # that atom to the neighbor list, which requires detecting whether it will be
        # translated to reside within the unit cell before neighbor detection, it is
        # less complex to just call the one-by-one operation
        if len(structure) == 1:
            return [self.get_voronoi_polyhedra(structure, 0)]

        # Assemble the list of neighbors used in the tessellation
        if self.targets is None:
            targets = structure.composition.elements
        else:
            targets = self.targets

        # Initialize the list of sites with the atoms in the origin unit cell
        # The `get_all_neighbors` function returns neighbors for each site's image in
        # the original unit cell. We start off with these central atoms to ensure they
        # are included in the tessellation

        if self.to_primitive:
            sites = [x.to_unit_cell() for x in structure]
        else:
            sites = [x for x in structure]
        indices = [(i, 0, 0, 0) for i, _ in enumerate(structure)]

        # Get all neighbors within a certain cutoff
        #   Record both the list of these neighbors, and the site indices
        all_neighs = structure.get_all_neighbors(self.cutoff, include_index=True, include_image=True, sites=sites)
        for neighs in all_neighs:
            sites.extend([x[0] for x in neighs])
            indices.extend([(x[2],) + x[3] for x in neighs])

        # Get the non-duplicates (using the site indices for numerical stability)
        if self.to_primitive:
            sites_frac_coords = [site.frac_coords for site in sites]
            sites_frac_coords = np.array(sites_frac_coords)
            _, uniq_inds = np.unique(sites_frac_coords, return_index=True, axis=0)
            sites = [sites[i] for i in sorted(uniq_inds)]
            root_images = np.arange(len(structure))
        else:
            indices = np.array(indices, dtype=np.int)
            indices, uniq_inds = np.unique(indices, return_index=True, axis=0)
            sites = [sites[i] for i in uniq_inds]
            (root_images,) = np.nonzero(np.abs(indices[:, 1:]).max(axis=1) == 0)

        # Sort array such that atoms in the root image are first
        #   Exploit the fact that the array is sorted by the unique operation such that
        #   the images associated with atom 0 are first, followed by atom 1, etc.

        del indices  # Save memory (tessellations can be costly)

        ######
        # root_sites = [sites[root_image] for root_image in root_images]
        # root_sites_coords = [site.coords for site in root_sites]
        # root_sites_coords = np.array(root_sites_coords)
        # x_root = root_sites_coords.T[0]
        # y_root = root_sites_coords.T[1]
        # z_root = root_sites_coords.T[2]

        # sites_frac_coords = [site.coords for site in sites]
        # sites_frac_coords = np.array(sites_frac_coords)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # x_coordinates = sites_frac_coords.T[0]
        # y_coordinates = sites_frac_coords.T[1]
        # z_coordinates = sites_frac_coords.T[2]
        # ax.scatter(x_coordinates, y_coordinates, z_coordinates)
        # ax.scatter(x_root, y_root, z_root, c="r", s=100)
        # for i, _ in enumerate(sites):
        #     ax.text(x_coordinates[i], y_coordinates[i], z_coordinates[i], "%s" % (str(i)))

        # plt.show()
        ######

        # Run the tessellation
        qvoronoi_input = [s.coords for s in sites]
        voro = Voronoi(qvoronoi_input)

        # Get the information for each neighbor
        return [
            self._extract_cell_info(structure, i, sites, targets, voro, self.compute_adj_neighbors)
            for i in root_images.tolist()
        ]

    def get_all_nn_info(self, structure: Structure) -> List[List[Dict[str, Any]]]:
        """
        Args:
            structure (Structure): input structure.

        Returns:
            List of near neighbor information for each site. See get_nn_info for the
            format of the data for each site.
        """
        if self.to_primitive:
            for site in structure:
                site.to_unit_cell(in_place=True)
        all_nns = self.get_all_voronoi_polyhedra(structure)
        return [self._filter_nns(structure, n, nns) for n, nns in enumerate(all_nns)]


class CustomCrystallNN(env.CrystalNN):
    def get_nn_data(self, structure, n, length=None):
        """
        The main logic of the method to compute near neighbor.

        Args:
            structure: (Structure) enclosing structure object
            n: (int) index of target site to get NN info for
            length: (int) if set, will return a fixed range of CN numbers

        Returns:
            a namedtuple (NNData) object that contains:
                - all near neighbor sites with weights
                - a dict of CN -> weight
                - a dict of CN -> associated near neighbor sites
        """

        length = length or self.fingerprint_length

        # determine possible bond targets
        target = None
        if self.cation_anion:
            target = []
            m_oxi = structure[n].specie.oxi_state
            for site in structure:
                if site.specie.oxi_state * m_oxi <= 0:  # opposite charge
                    target.append(site.specie)
            if not target:
                raise ValueError("No valid targets for site within cation_anion constraint!")

        # get base VoronoiNN targets
        cutoff = self.search_cutoff
        vnn = CustomVoronoiNN(weight="solid_angle", targets=target, cutoff=cutoff)
        nn = vnn.get_nn_info(structure, n)

        # solid angle weights can be misleading in open / porous structures
        # adjust weights to correct for this behavior
        if self.porous_adjustment:
            for x in nn:
                x["weight"] *= x["poly_info"]["solid_angle"] / x["poly_info"]["area"]

        # adjust solid angle weight based on electronegativity difference
        if self.x_diff_weight > 0:
            for entry in nn:
                X1 = structure[n].specie.X
                X2 = entry["site"].specie.X

                if math.isnan(X1) or math.isnan(X2):
                    chemical_weight = 1
                else:
                    # note: 3.3 is max deltaX between 2 elements
                    chemical_weight = 1 + self.x_diff_weight * math.sqrt(abs(X1 - X2) / 3.3)

                entry["weight"] = entry["weight"] * chemical_weight

        # sort nearest neighbors from highest to lowest weight
        nn = sorted(nn, key=lambda x: x["weight"], reverse=True)
        if nn[0]["weight"] == 0:
            return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)

        # renormalize weights so the highest weight is 1.0
        highest_weight = nn[0]["weight"]
        for entry in nn:
            entry["weight"] = entry["weight"] / highest_weight

        # adjust solid angle weights based on distance
        if self.distance_cutoffs:
            r1 = _get_radius(structure[n])
            for entry in nn:
                r2 = _get_radius(entry["site"])
                if r1 > 0 and r2 > 0:
                    d = r1 + r2
                else:
                    warnings.warn(
                        "CrystalNN: cannot locate an appropriate radius, "
                        "covalent or atomic radii will be used, this can lead "
                        "to non-optimal results."
                    )
                    d = _get_default_radius(structure[n]) + _get_default_radius(entry["site"])

                dist = np.linalg.norm(structure[n].coords - entry["site"].coords)
                dist_weight = 0

                cutoff_low = d + self.distance_cutoffs[0]
                cutoff_high = d + self.distance_cutoffs[1]

                if dist <= cutoff_low:
                    dist_weight = 1
                elif dist < cutoff_high:
                    dist_weight = (math.cos((dist - cutoff_low) / (cutoff_high - cutoff_low) * math.pi) + 1) * 0.5
                entry["weight"] = entry["weight"] * dist_weight

        # sort nearest neighbors from highest to lowest weight
        nn = sorted(nn, key=lambda x: x["weight"], reverse=True)
        if nn[0]["weight"] == 0:
            return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)

        for entry in nn:
            entry["weight"] = round(entry["weight"], 3)
            del entry["poly_info"]  # trim

        # remove entries with no weight
        nn = [x for x in nn if x["weight"] > 0]

        # get the transition distances, i.e. all distinct weights
        dist_bins = []
        for entry in nn:
            if not dist_bins or dist_bins[-1] != entry["weight"]:
                dist_bins.append(entry["weight"])
        dist_bins.append(0)

        # main algorithm to determine fingerprint from bond weights
        cn_weights = {}  # CN -> score for that CN
        cn_nninfo = {}  # CN -> list of nearneighbor info for that CN
        for idx, val in enumerate(dist_bins):
            if val != 0:
                nn_info = []
                for entry in nn:
                    if entry["weight"] >= val:
                        nn_info.append(entry)
                cn = len(nn_info)
                cn_nninfo[cn] = nn_info
                cn_weights[cn] = self._semicircle_integral(dist_bins, idx)

        # add zero coord
        cn0_weight = 1.0 - sum(cn_weights.values())
        if cn0_weight > 0:
            cn_nninfo[0] = []
            cn_weights[0] = cn0_weight

        return self.transform_to_length(self.NNData(nn, cn_weights, cn_nninfo), length)
