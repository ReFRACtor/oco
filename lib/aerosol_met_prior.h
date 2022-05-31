#ifndef AEROSOL_MET_PRIOR_H
#define AEROSOL_MET_PRIOR_H
#include "aerosol.h"
#include "aerosol_extinction_imp_base.h"
#include "aerosol_property.h"
#include "hdf_file.h"
#include "pressure.h"
#include "oco_met_file.h"

namespace FullPhysics {
/****************************************************************//**
  This class is used to create the Aerosol from a Meteorology
  file.
*******************************************************************/
class AerosolMetPrior: public Printable<AerosolMetPrior> {
public:

  AerosolMetPrior(const OcoMetFile& Met_file,
                  const HdfFile& Aerosol_property,
                  const boost::shared_ptr<Pressure> &Press,
                  const boost::shared_ptr<RelativeHumidity> &Rh,
                  double Exp_aod = 0.8,
                  int Min_types = 2,
                  int Max_types = 2,
                  bool Linear_aod = false,
                  bool Relative_humidity_aerosol = false,
                  double Max_residual = 0.005,
                  double Reference_wn=1e4/0.755);

  virtual ~AerosolMetPrior() {}

  boost::shared_ptr<Aerosol> aerosol() const;

  void add_aerosol(const boost::shared_ptr<AerosolExtinction>& Aext,
                   const boost::shared_ptr<AerosolProperty>& Aprop);

  std::vector<boost::shared_ptr<AerosolExtinction> > extinction() const
  { return aext; }

  std::vector<boost::shared_ptr<AerosolExtinctionImpBase> > retrieved_extinction() const;

  std::vector<boost::shared_ptr<AerosolProperty> > property() const
  { return aprop; }

//-----------------------------------------------------------------------
/// Number of merra particles.
//-----------------------------------------------------------------------

  int number_merra_particle() const {return number_merra_particle_;}

  virtual void print(std::ostream& Os) const;
private:
  mutable boost::shared_ptr<Aerosol> aerosol_;
  bool linear_aod, rh_aerosol;
  int number_merra_particle_;
  std::vector<boost::shared_ptr<AerosolExtinction> > aext;
  std::vector<boost::shared_ptr<AerosolProperty> > aprop;
  boost::shared_ptr<Pressure> press;
  boost::shared_ptr<RelativeHumidity> rh;
  double ref_wn;
  std::string met_fname, prop_fname;
};
}
#endif
