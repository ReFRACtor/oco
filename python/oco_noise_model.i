%module(directors="1", allprotected="1") oco_noise_model
%include "fp_common.i"

%{
#include "oco_noise_model.h"
%}

%base_import(noise_model)
%import(module="oco.hdf_sounding_id") "hdf_sounding_id.i"

%import "hdf_file.i"

%fp_shared_ptr(FullPhysics::OcoNoiseModel);

namespace FullPhysics {
class OcoNoiseModel : public NoiseModel {
public:
  OcoNoiseModel(const HdfFile& Hfile, const HdfSoundingId& Sounding_id);
  OcoNoiseModel(const HdfFile& Hfile, const HdfSoundingId& Sounding_id, const blitz::Array<double, 1>& Max_meas_signal);
  virtual blitz::Array<double, 1> uncertainty(int Spec_index, const blitz::Array<double, 1>& Radiance) const;
};
}
