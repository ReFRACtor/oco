%module(directors="1", allprotected="1") level_1b_hdf
%include "fp_common.i"

%{
#include "level_1b_hdf.h"
#include "noise_model.h"
%}

%base_import(level_1b_sample_coefficient)
%import(module="oco.hdf_sounding_id") "hdf_sounding_id.i"

%import "hdf_file.i"
%import "noise_model.i"
%import "array_with_unit.i"
%import "double_with_unit.i"

%fp_shared_ptr(FullPhysics::Level1bHdf);

namespace FullPhysics {
class Level1bHdf: public Level1bSampleCoefficient {
public:
  virtual ~Level1bHdf();
  virtual DoubleWithUnit latitude(int i) const;
  virtual DoubleWithUnit longitude(int i) const;
  virtual DoubleWithUnit sounding_zenith(int i) const;
  virtual DoubleWithUnit sounding_azimuth(int i) const;
  virtual blitz::Array<double, 1> stokes_coefficient(int i) const;
  virtual DoubleWithUnit solar_zenith(int i) const;
  virtual DoubleWithUnit solar_azimuth(int i) const;
  virtual DoubleWithUnit altitude(int i) const;
  virtual DoubleWithUnit relative_velocity(int i) const;
  virtual ArrayWithUnit<double, 1> spectral_coefficient(int i) const;
  virtual blitz::Array<double, 1> spectral_variable(int channel_index) const;
  virtual Time time(int i) const;
  %python_attribute_with_set(noise_model, boost::shared_ptr<FullPhysics::NoiseModel>);
protected:
  Level1bHdf();
  Level1bHdf(const std::string& Fname, 
             const boost::shared_ptr<HdfSoundingId>& Sounding_id);
  Level1bHdf(const boost::shared_ptr<HdfFile>& Hfile, 
             const boost::shared_ptr<HdfSoundingId>& Sounding_id);
};
}
