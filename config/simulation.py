import numpy as np
import netCDF4

from refractor import framework as rf

class DataGroupValues(object):

    def __init__(self, file_contents, index, group_name):
        self.index = index
        self.group_data = file_contents[group_name]

    def __getattr__(self, name):

        dataset = self.group_data[name]
        data_value = dataset[self.index, ...]

        if hasattr(dataset, "unit"):
            ndim = len(dataset.shape) - 1
            if ndim > 0:
                ArrayAdClass = getattr(rf, "ArrayWithUnit_double_%d" % ndim)
                return ArrayAdClass(data_value, dataset.unit)
            else:
                return rf.DoubleWithUnit(float(data_value), dataset.unit)
        else:
            return data_value

class ScenarioValues(DataGroupValues):
    
    def __init__(self, file_contents, index):
        super().__init__(file_contents, index, "Scenario")

    @property
    def altitude(self):
        return self.surface_height

    @property
    def time(self):
        times = []
        for pgs_time_val in self.group_data['time'][self.index, :]:
            times.append(rf.Time.time_pgs(float(pgs_time_val)))
        return times

class AtmosphereValues(DataGroupValues):

    def __init__(self, file_contents, index):
        super().__init__(file_contents, index, "Atmosphere")

    @property
    def surface_pressure(self):
        # Creator expects an array
        psurf = self.group_data["surface_pressure"][self.index]
        return np.array([psurf])

    @property
    def pressure_levels(self):
        # Creator expects just array without units
        return self.group_data["pressure_levels"][self.index, :]

    @property
    def temperature(self):
        # Creator expects just array without units
        return self.group_data["temperature"][self.index, :]

class AtmosphereElements(object):

    def __init__(self, file_contents, index, group_name, name_field="name"):
        self.index = index
        self.group_data = file_contents[group_name]
        self.name_field = name_field

    def _extract_names(self, field_name):
        return list(netCDF4.chartostring(self.group_data[field_name][self.index, :]))

    @property
    def element_names(self):
        return self._extract_names(self.name_field)

    def element(self, element_name, field_name):

        if element_name in self.element_names:
            elem_idx = self.element_names.index(element_name)
            return self.group_data[field_name][self.index, elem_idx, :]
        else:
            raise Exception("Element not found: %s" % element_name)

class AbsorberValues(AtmosphereElements):

    def __init__(self, file_contents, index):
        super().__init__(file_contents, index, "Atmosphere/Absorber")

    @property
    def molecule_names(self):
        return self.element_names

    def vmr(self, molecule_name):
        return self.element(molecule_name, "vmr")

class AerosolValues(AtmosphereElements):

    def __init__(self, file_contents, index):
        super().__init__(file_contents, index, "Atmosphere/Aerosol")

    @property
    def particle_names(self):
        return self.element_names

    def property_name(self, aer_name):
        prop_names = self._extract_names("property_name")

        if aer_name in self.particle_names:
            aer_idx = self.particle_names.index(aer_name)
            return prop_names[aer_idx]
        else:
            raise Exception("Aerosol property not found: %s" % aer_name)

    def gaussian_param(self, aer_name):
        return self.element(aer_name, "gaussian_params")

class GroundValues(DataGroupValues):

    def __init__(self, file_contents, index):
        super().__init__(file_contents, index, "Atmosphere/Ground")

class SimulationFile(object):

    def __init__(self, filename, index):
        file_contents = netCDF4.Dataset(filename)

        self.scenario = ScenarioValues(file_contents, index)
        self.instrument = DataGroupValues(file_contents, index, "Instrument")
        self.atmosphere = AtmosphereValues(file_contents, index)
        self.absorber = AbsorberValues(file_contents, index)
        self.aerosol = AerosolValues(file_contents, index)
        self.ground = GroundValues(file_contents, index)
