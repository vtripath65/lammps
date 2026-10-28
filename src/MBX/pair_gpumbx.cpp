/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_gpumbx.h"

#include "comm.h"
#include "error.h"
#include "fix_mbx.h"

#include "bblock/system.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   variant of pair style mbx that obtains energy, forces, and virial from a
   single MBX energy call; requires an MBX GPU library (libmbx_cuda or
   libmbx_hip, selected with -D MBX_GPU=cuda|hip)
------------------------------------------------------------------------- */

PairGPUMBX::PairGPUMBX(LAMMPS *lmp) : PairMBX(lmp), energy_terms(0)
{
  energy_full = 1;
}

/* ---------------------------------------------------------------------- */

void PairGPUMBX::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // all terms from one MBX call on the periodic system of local molecules

  bblock::System *ptr_mbx_local = mbx_local();
  const bool has_mbx_atoms = (fix_MBX->mbx_num_atoms_local + fix_MBX->mbx_num_ext_local > 0);

  for (int i = 0; i < 6; ++i) mbx_virial[i] = 0.0;
  mbx_total_energy = 0.0;

  if (has_mbx_atoms) {
    fix_MBX->mbxt_start(FixMBX::MBXT_LABELS::ENERGY_FULL);
    mbx_total_energy = ptr_mbx_local->Energy(true);
    fix_MBX->mbxt_stop(FixMBX::MBXT_LABELS::ENERGY_FULL);
    accumulate_f_local(true);
  }

  for (int i = 0; i < 6; ++i) virial[i] += mbx_virial[i];
  if (evflag) eng_vdwl = mbx_total_energy;

#ifdef MBX_HAS_ENERGY_TERMS
  // same layout as pair style mbx

  if (evflag && energy_terms) {
    for (int i = 0; i < nextra; ++i) pvector[i] = 0.0;
    if (!has_mbx_atoms) return;

    using Term = bblock::System::EnergyTerm;
    std::vector<double> e = ptr_mbx_local->GetEnergyTerms();
    pvector[0] = e[Term::E1B];
    pvector[1] = e[Term::E2B];
    pvector[2] = e[Term::E3B];
    pvector[3] = e[Term::E4B];
    pvector[4] = e[Term::EDISP];
    pvector[5] = e[Term::EBUCK] + e[Term::ELJ] + e[Term::EFF];
    pvector[6] = ptr_mbx_local->GetPermanentElectrostaticEnergy();
    pvector[7] = ptr_mbx_local->GetInducedElectrostaticEnergy();
    pvector[8] = e[Term::EELEC];
    pvector[9] = mbx_total_energy;
  }
#endif
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGPUMBX::settings(int narg, char **arg)
{
  if (narg < 1) utils::missing_cmd_args(FLERR, "pair_style gpumbx", error);

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  int terms = -1;    // -1: not set, use default
  int iarg = 1;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "terms") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style gpumbx terms", error);
      terms = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown pair_style gpumbx keyword: {}", arg[iarg]);
    }
  }

  // individual energy terms are reported by default if the MBX library provides them

#ifdef MBX_HAS_ENERGY_TERMS
  energy_terms = (terms != 0) ? 1 : 0;
#else
  if (terms == 1)
    error->all(FLERR, "Pair style gpumbx keyword terms requires an MBX library that provides "
                      "individual energy terms");
  energy_terms = 0;
#endif

  nextra = energy_terms ? 10 : 0;

  // gpumbx only makes sense with a GPU MBX library; the CPU libmbx reports "none",
  // and an MBX without GpuBackend() predates the GPU libraries

#ifdef MBX_HAS_GPU_BACKEND
  if (strcmp(bblock::System::GpuBackend(), "none") == 0)
    error->all(FLERR, "Pair style gpumbx requires an MBX GPU library (libmbx_cuda or libmbx_hip); "
                      "reconfigure LAMMPS with -D MBX_GPU=cuda or -D MBX_GPU=hip");
  // write MBX's device line to the LAMMPS log instead of letting MBX print it to stderr;
  // other ranks keep MBX's own stderr line

  if (comm->me == 0) {
    utils::logmesg(lmp, "Pair style gpumbx: MBX GPU backend {}\n", bblock::System::GpuBackend());
#ifdef MBX_HAS_GPU_DEVICE_INFO
    utils::logmesg(lmp, "{}\n", bblock::System::GpuDeviceInfo());
#endif
  }
#else
  error->all(FLERR, "Pair style gpumbx requires an MBX library with GPU support");
#endif
}
