/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(gpumbx, PairGPUMBX)

#else

#ifndef LMP_PAIR_GPUMBX_H
#define LMP_PAIR_GPUMBX_H

#include "pair_mbx.h"

namespace LAMMPS_NS {

class PairGPUMBX : public PairMBX {
 public:
  PairGPUMBX(class LAMMPS *);

  void compute(int, int) override;
  void settings(int, char **) override;

 protected:
  int energy_terms;    // 1 if individual energy terms are reported
};

}    // namespace LAMMPS_NS

#endif
#endif
