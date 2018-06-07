#ifndef OCO_MET_FILE_H
#define OCO_MET_FILE_H

#include "meteorology.h"
#include "hdf_sounding_id.h"
#include <boost/algorithm/string/case_conv.hpp>

namespace FullPhysics {
/****************************************************************//**
  This class implements the OCO specific ECMWF reading 
  functionality.
*******************************************************************/

class OcoMetFile : public Meteorology {
public:
    OcoMetFile(const std::string& Fname, const boost::shared_ptr<HdfSoundingId>& Hdf_sounding_id);
    ~OcoMetFile() {}

    // Define how to read various items
    using Meteorology::pressure_levels;
    blitz::Array<double, 1> pressure_levels() const
        { return read_array(group_name + "/vector_pressure_levels" + name_suffix); }

    using Meteorology::specific_humidity;
    blitz::Array<double, 1> specific_humidity() const
        { return read_array(group_name + "/specific_humidity_profile" + name_suffix); }

    using Meteorology::vmr;
    blitz::Array<double, 1> vmr(const std::string& Species) const;

    blitz::Array<double, 1> ozone_mmr() const
        { return read_array(group_name + "/ozone_profile" + name_suffix); }

    virtual blitz::Array<double, 1> ozone_vmr() const;

    double surface_pressure() const
        { return read_scalar(group_name + "/surface_pressure" + name_suffix); }

    double windspeed_u() const
        { return read_scalar(group_name + "/windspeed_u" + name_suffix); }

    double windspeed_v() const
        { return read_scalar(group_name + "/windspeed_v" + name_suffix); }

    using Meteorology::temperature;
    blitz::Array<double, 1> temperature() const
        { return read_array(group_name + "/temperature_profile" + name_suffix); }

    void print(std::ostream& Os) const { Os << "OcoMetFile"; }

private:

    //-----------------------------------------------------------------------
    /// OCO specific ECMWF reader routines
    //-----------------------------------------------------------------------

    double read_scalar(const std::string& Field) const;
    blitz::Array<double, 1> read_array(const std::string& Field) const;

    HdfFile h;
    boost::shared_ptr<HdfSoundingId> hsid;
    bool average_sounding_number;

    // Name of the HDF group to read from, either "ECMWF" or "Meteorology"
    std::string group_name;

    // Suffix of the datasets, either "_ecmwf" or "_met"
    std::string name_suffix;
};
}
#endif
