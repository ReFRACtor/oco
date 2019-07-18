%module(directors="1", allprotected="1") oco_sounding_id
%include "fp_common.i"

%{
#include "oco_sounding_id.h"
%}

%import(module="oco.hdf_sounding_id") "hdf_sounding_id.i"

%import "hdf_file.i"

%fp_shared_ptr(FullPhysics::OcoSoundingId);

namespace FullPhysics {
class OcoSoundingId : public HdfSoundingId {
public:
  OcoSoundingId(const std::string& Fname, const std::string& Sounding_id);
  OcoSoundingId(const HdfFile& File, const std::string& Sounding_id);
  virtual ~OcoSoundingId();
};
}
