%module(directors="1", allprotected="1")  "oco"

// Do this only once for all classes, swig_array must be included
// first to have the swig setup defined in this repository
%include "swig_array.i"
%include "fp_common.i"

%include "hdf_sounding_id.i"
%include "oco_noise_model.i"
%include "level_1b_hdf.i"
%include "level_1b_oco.i"
%include "oco_met_file.i"
%include "oco_sounding_id.i"
