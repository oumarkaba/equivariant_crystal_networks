import uuid

import numpy as np
from pypif.pif import dump, dumps
from pypif.obj import ChemicalSystem, Property, Scalar
from pypif.util.case import keys_to_snake_case

import learn_materials.prepare.chemistry as chem
from learn_materials.utils import load_json

np.seterr("raise")


class Material(ChemicalSystem):
    def __init__(
        self,
        names=None,
        ids=None,
        references=None,
        chemical_formula=None,
        properties=None,
        **kwargs,
    ):
        super().__init__(
            names=names,
            ids=ids,
            references=references,
            chemical_formula=chemical_formula,
            properties=properties,
            **kwargs,
        )
        self.uid = str(uuid.uuid1()) if not self.uid else self.uid

    def update_composition_descriptors(self, rounding=4):
        # print(f"Computing descriptors for material with id {self.uid}")
        if self.chemical_formula is None:
            return
        atomic_descriptors = chem.compute_atomic_descriptors(self.chemical_formula)
        stats_functions = {
            "Mean ": lambda values: np.sum(
                values
                * np.array(atomic_descriptors["number of atoms"])
                / np.sum(np.array(atomic_descriptors["number of atoms"]))
            ),
            "Maximum ": np.max,
            "Minimum ": np.min,
            "Maximum difference ": lambda values: np.max(values) - np.min(values),
            "Std ": np.std,
            # TODO: Marc-Antoine also had  "Magnetic atom", "Mag aba", "Mag abl"
            # TODO: Benjamin also had operations
            # "Mean among mag":
            # "Std among mag":
            # TODO: benjamin also used some custom descriptors:
            # ratio mag/total, number of species, number of different mag,
            # fraction_s, fraction_p, fraction_d, fraction_f, where
            # fraction_s = sum(valence_s) / sum(valence_s+valence_p+... )
        }

        composition_properties = []
        for stat, function in stats_functions.items():
            for descriptor, values in atomic_descriptors.items():
                # TODO: allow for skipping atoms without properties without replacing by zero, or provoking nan for the mean value, max or min...
                if stat == "Mean " and descriptor == "number of atoms":
                    scalar = Scalar(np.mean(values))
                else:
                    scalar = Scalar(round(function(values), rounding))
                if not np.isnan(scalar.value):
                    # FIXME: our periodic table does not use the units from Mendeleev
                    units = None  # chem.PERIODICTABLE.descriptor_units[descriptor]
                    prop = Property(
                        name=stat + descriptor, scalars=[scalar], units=units
                    )
                    composition_properties.append(prop)

        if not self.properties:
            self.properties = []
        self.properties.extend(composition_properties)

    # def add_magnetic_moment(self):
    #     ratio, formula_len = chem.cell_formula_ratio(self.chemical_formula)
    #     if ratio != 1.0:
    #         print(ratio)
    #     formula_moment = self.get_property_value("Total magnetization")
    #     cell_moment = Scalar(formula_moment * ratio)
    #     moment_per_atom = Scalar(formula_moment / formula_len)

    #     magnetic_moment = Property(
    #         name="Magnetic moment per unit cell",
    #         scalars=[cell_moment],
    #         units="$\\mu_B$",
    #     )
    #     magnetic_moment_per_atom = Property(
    #         name="Magnetic moment per unit cell per atom",
    #         scalars=[moment_per_atom],
    #         units="$\\mu_B$",
    #     )

    #     self.properties.extend([magnetic_moment, magnetic_moment_per_atom])

    def get_mp_id(self, source="Materials Project"):
        mp_ids = [
            single_id.value
            for single_id in self.ids
            if single_id.name == source
        ]
        return mp_ids[0] if mp_ids else None

    def property_names(self):
        return [prop.name for prop in self.properties] if self.properties else None

    def set_property(self, prop_name, new_prop):
        if self.properties:
            for i, prop in enumerate(self.properties):
                if prop.name == prop_name:
                    self.properties[i] = new_prop
                    return
            print(
                f'Property "{prop_name}" missing for material with id {self.uid}. Adding it.'
            )
            self.properties.append(new_prop)
        else:
            print(
                f'Property "{prop_name}" missing for material with id {self.uid}. Adding it.'
            )
            self.properties = [new_prop]

    def get_property(self, prop_name):
        if self.properties:
            for prop in self.properties:
                if prop.name == prop_name:
                    return prop
        print(f'Missing property "{prop_name}" for material with id {self.uid}')
        return None

    def get_property_value(
        self, prop_name, estimator="mean", uncert=False, nan=None, uncert_nan=None
    ):
        prop = self.get_property(prop_name)
        if prop:
            if prop.scalars[0].value is None:
                return None
            values = [scalar.value for scalar in prop.scalars]
            # FIXME: Calculation of the uncertainty when there are multiple data points not properly implemented
            uncertainties = [
                scalar.uncertainty for scalar in prop.scalars if scalar.uncertainty
            ]
            if estimator == "mean":
                estimated_value = np.mean(values)
                uncert_value = (
                    np.std(values) + np.mean(uncertainties) if uncert else None
                )
            elif estimator == "median":
                estimated_value = np.median(values)
                uncert_value = (
                    (np.median(np.abs(np.array(values) - estimated_value)))
                    + np.median(uncertainties)
                    if uncert
                    else None
                )
            else:
                raise ValueError(f"estimator has to be mean or median")
            if uncert:
                return (
                    (estimated_value, uncert_value)
                    if uncert_value
                    else (estimated_value, uncert_nan)
                    if uncert_nan
                    else (estimated_value, None)
                )
            else:
                return estimated_value
        if nan:
            if uncert:
                return (nan, uncert_nan) if uncert_nan else (nan, None)
            else:
                return nan
        return None

    def get_properties_values(
        self, props=None, estimator="mean", uncert=False, nan=None, uncert_nan=None
    ):
        if not self.properties:
            return [nan for prop in range(len(props))] if (nan and props) else None
        if props:
            return [
                self.get_property_value(prop, estimator, uncert, nan, uncert_nan)
                for prop in props
            ]
        return (
            [
                self.get_property_value(prop, estimator, uncert, nan, uncert_nan)
                for prop in self.property_names()  # pylint: disable=E1133
            ]
            if self.property_names()
            else None
        )

    @staticmethod
    def combine(original_material, added_material):
        combined_material = Material(
            names=original_material.names,
            tags=original_material.tags,
            ids=original_material.ids,
            chemical_formula=original_material.chemical_formula,
            references=original_material.references,
            properties=original_material.properties,
        )
        if combined_material.tags:
            combined_material.tags += added_material.tags if added_material.tags else []
        if combined_material.names:
            combined_material.names += (
                added_material.names if added_material.names else []
            )
        if combined_material.ids:
            combined_material.ids += added_material.ids if added_material.ids else []
        if combined_material.references:
            combined_material.references += (
                added_material.references if added_material.references else []
            )

        if combined_material.properties and added_material.properties:
            same_properties = list(
                set(combined_material.property_names()).intersection(
                    set(added_material.property_names())
                )
            )
            for same_property in same_properties:
                original_prop = combined_material.get_property(same_property)
                added_prop = added_material.get_property(same_property)
                combined_material.set_property(
                    same_property, combine_properties(original_prop, added_prop)
                )

            different_properties = list(
                set(added_material.property_names()).difference(
                    set(combined_material.property_names())
                )
            )
            for different_property in different_properties:
                added_prop = added_material.get_property(different_property)
                combined_material.set_property(different_property, added_prop)
        elif added_material.properties:
            combined_material.properties = added_material.properties

        return combined_material

    @staticmethod
    def all_descriptors():
        stats = ["Mean ", "Maximum ", "Minimum ", "Maximum difference "]
        descriptors = [
            "atomic number",
            "number of atoms",
            "atomic mass",
            "atomic magnetic moment",
            "atomic electronegativity",
            "ionization energy",
            "atomic radius",
            "electron affinity",
            "atomic electronegativity",
            "covalent radius",
            "cohesive energy",
            "electron binding energy" "atomic valence number",
            "atomic valence radius",
        ]
        return [stat + descriptor for stat in stats for descriptor in descriptors]


def combine_properties(original_property, added_property):
    combined_property = Property(
        name=original_property.name,
        units=original_property.units,
        scalars=original_property.scalars,
        vectors=original_property.vectors,
        matrices=original_property.matrices,
        conditions=original_property.conditions,
        files=original_property.files,
    )
    if combined_property.scalars:
        combined_property.scalars += (
            added_property.scalars if added_property.scalars is not None else []
        )
    else:
        combined_property.scalars = added_property.scalars
    if combined_property.vectors:
        combined_property.vectors += (
            added_property.vectors if added_property.vectors is not None else []
        )
    else:
        combined_property.vectors = added_property.vectors
    if combined_property.matrices:
        combined_property.matrices += (
            added_property.matrices if added_property.matrices is not None else []
        )
    else:
        combined_property.matrices = added_property.matrices
    if combined_property.conditions:
        combined_property.conditions += (
            added_property.conditions if added_property.conditions is not None else []
        )
    else:
        combined_property.conditions = added_property.conditions
    if combined_property.files:
        combined_property.files += (
            added_property.files if added_property.files is not None else []
        )
    else:
        combined_property.files = added_property.files
    return combined_property


def load_material_file(pif_path):
    print("Starting to load material file...")
    materials_dicts = load_json(pif_path)
    print("File loaded")
    return load_material_dicts(materials_dicts)


def save_material_file(materials, path):
    with open(path, "w", encoding="utf-8") as file:
        dump(materials, file)


def load_material_dicts(obj):
    if isinstance(obj, list):
        return [Material(**keys_to_snake_case(i)) for i in obj]
    elif isinstance(obj, dict):
        return Material(**keys_to_snake_case(obj))
    else:
        raise ValueError("expecting list or dictionary as outermost structure")


def main():
    # mat1 = Material(chemical_formula="FeO3")
    # mat2 = Material(chemical_formula="FeO3")
    # mat1.update_composition_descriptors()

    # prop1 = Property(
    #     name="velocity",
    #     scalars=[Scalar(value=0), Scalar(value=2), Scalar(value=10)],
    #     units="K",
    # )
    # mat1.set_property("velocity", prop1)
    # print(vars(mat1))
    # print(pif.dumps(mat1))

    # prop2 = Property(name="velocity", scalars=[Scalar(value=1)])
    # mat2.set_property("velocity", prop2)

    # mat = Material.combine(mat1, mat2)
    # print(pif.dumps(mat))
    mat = Material(chemical_formula="H3Ru")
    mat.update_composition_descriptors()
    print(dumps(mat.properties))


if __name__ == "__main__":
    main()
