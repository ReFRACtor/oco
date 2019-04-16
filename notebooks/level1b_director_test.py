import numpy as np

from refractor import framework as rf
from refractor.factory.creator.base import Creator
from refractor.factory import param

spec_coeffs = np.array([
    [ 7.57646001e-01,  1.74935138e-05, -2.73614810e-09, -7.51159779e-14, 1.32166257e-16, -7.27469045e-20],
    [ 1.59053617e+00,  3.64469725e-05, -5.95489446e-09,  9.39329211e-13, -9.47239929e-16,  3.16308468e-19],
    [ 2.04299043e+00,  4.69338016e-05, -7.53637453e-09,  9.08004109e-14, 3.85120599e-16, -1.92863621e-19],
])

num_samples = 1016

class Level1bDirector(rf.Level1bSampleCoefficient):

    def __init__(self):
        # This call is essential so the class gets connected to its director class
        super().__init__()

    # Methods needed by rf.Level1b

    def number_spectrometer(self) -> int:
        return 3

    def latitude(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(-27, "deg")

    def longitude(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(44, "deg")

    def sounding_zenith(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(0.1, "deg")

    def sounding_azimuth(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(295, "deg")

    def solar_zenith(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(55, "deg")

    def solar_azimuth(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(330, "deg")

    def stokes_coefficient(self, chan_index: int) -> np.ndarray:
        return np.array([1, 0, 0, 0], dtype=float)

    def altitude(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(0, "m")

    def relative_velocity(self, chan_index: int) -> rf.DoubleWithUnit:
        return rf.DoubleWithUnit(-14, "m/s")

    def sample_grid(self, chan_index: int) -> rf.SpectralDomain: 

        poly = np.poly1d(spec_coeffs[chan_index, ::-1])
        grid = poly(np.arange(1, num_samples+1))

        return rf.SpectralDomain(rf.ArrayWithUnit(grid, "micron"))

    def time(self, chan_index: int) -> rf.Time:
        return rf.Time.parse_time("2017-07-11T10:54:24.564724Z")

    def radiance(self, chan_index: int) -> rf.SpectralRange:
        uncertainty = np.random.rand(1016)
        rad_units = "ph / s / m^2 / micron sr^-1"
        return rf.SpectralRange(np.random.rand(1016)*1e19, rad_units, uncertainty)

    # Additional methods needed by rf.Level1bSampleCoefficient
    def number_sample(self, chan_index: int) -> int:
        return num_samples

    def spectral_coefficient(self, chan_index: int) -> rf.ArrayWithUnit:
        return rf.ArrayWithUnit(spec_coeffs[chan_index, :], "micron")

class Level1bDirectorCreator(Creator):

    def create(self, **kwargs):

        return Level1bDirector()
