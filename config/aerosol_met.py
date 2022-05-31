from refractor.framework.factory import param, Creator

import oco
import refractor.framework as rf

class AerosolMetPrior(Creator):

    aerosols = param.Iterable()
    pressure = param.InstanceOf(rf.Pressure)
    relative_humidity = param.InstanceOf(rf.RelativeHumidity)
    reference_wn = param.Scalar(int, default=1e4 / 0.755)

    met = param.InstanceOf(oco.OcoMetFile)
    aerosol_prop_file = param.Scalar(str)

    exp_aod = param.Scalar(float)
    min_types = param.Scalar(int)
    max_types = param.Scalar(int)
    linear_aod = param.Scalar(bool)
    relative_humidity_aerosol = param.Scalar(bool)
    max_residual = param.Scalar(float)

    def create(self, **kwargs):

        aer_prop = rf.HdfFile(self.aerosol_prop_file())

        aerosol_met_prior = oco.AerosolMetPrior(self.met(), aer_prop, self.pressure(), self.relative_humidity(),
            self.exp_aod(), self.min_types(), self.max_types(), self.linear_aod(), self.relative_humidity_aerosol(), self.max_residual(),
            self.reference_wn())

        # Manually dispatch these objects created within the AerosolMetPrior so they get registered into the state vector
        for aer_ext in aerosol_met_prior.retrieved_extinction:
            self._dispatch(aer_ext)

        for aerosol_name in self.aerosols():
            self.register_parameter(aerosol_name, param.Dict())
            aerosol_def = self.param(aerosol_name, aerosol_name=aerosol_name)

            if not "extinction" in aerosol_def:
                raise param.ParamError("exitinction value not in aerosol definition for aerosol: %s" % aerosol_name)

            if not "properties" in aerosol_def:
                raise param.ParamError("peroperties value not in aerosol definition for aerosol: %s" % aerosol_name)

            aerosol_met_prior.add_aerosol(aerosol_def['extinction'], aerosol_def['properties'])

        return aerosol_met_prior.aerosol
