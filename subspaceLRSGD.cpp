/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** This program is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
** 
** You should have received a copy of the GNU General Public License
** along with this program. If not, see <http://www.gnu.org/licenses/>.
**
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "utils.h"
#include "mtrand.h"
#include "subspaceLRSGD.h"

#include "learnerRegistry.h"

static LearnerRegistrar registrar("subspace-lrsgd", constructor<subspaceLRSGD>);

subspaceLRSGD::subspaceLRSGD(char*const*& argv, char*const* end)
{
  name_ = "LRSGD";
  
  
  // defaults
  regularisation_rate_ = 0.001;
  update_rate_ = 0.001;

  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (argv[0][1] == 'r') {
      getFloatFromStr(argv[0]+2, regularisation_rate_, "Regularisation Rate");
    }
    else if (argv[0][1] == 'u') {
      getFloatFromStr(argv[0]+2, update_rate_, "Update Rate");
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}


subspaceLRSGD::~subspaceLRSGD(void)
{
}


void  subspaceLRSGD::getCapabilities(capabilities &c){
  c.setCatAtts(true);
  c.setNumAtts(true);
}

void subspaceLRSGD::reset(InstanceStream &is) {
  trainingIsFinished_ = false;

  metaData_ = is.getMetaData();

  update_num_ = 0;

  alpha_.assign(is.getNoClasses()-1, 0.0);
  
  numweights_.resize(is.getNoClasses()-1);
  for (CatValue y = 0; y < is.getNoClasses()-1; y++) {
    numweights_[y].assign(is.getNoNumAtts(), 0.0);
  }
  nummin_.assign(is.getNoNumAtts(), std::numeric_limits<double>::max());
  nummax_.assign(is.getNoNumAtts(), -std::numeric_limits<double>::max());
  mean_.assign(is.getNoNumAtts(), 0.0);
  catweights_.clear();
  last_update_cat_.clear();
  last_update_cat_.resize(is.getNoCatAtts());
  catweights_.resize(is.getNoClasses()-1);
  for (CatValue y = 0; y < is.getNoClasses()-1; y++) {
    catweights_[y].resize(is.getNoCatAtts());
  }
  for (CategoricalAttribute i = 0; i < is.getNoCatAtts(); i++) {
    for (CatValue y = 0; y < is.getNoClasses()-1; y++) {
      catweights_[y][i].assign(is.getNoValues(i), 0.0);
    }
    last_update_cat_[i].assign(is.getNoValues(i), 0);
  }
    
  //csscatweights_[s][sv][ym1][i][v]
  //cssnumweights_[s][sv][ym1][i]
  csscatweights_.resize(is.getNoCatAtts());
  cssnumweights_.resize(is.getNoCatAtts());
  for (CategoricalAttribute s = 0; s < is.getNoCatAtts(); s++) {
    csscatweights_[s].resize(is.getNoValues(s));
    cssnumweights_[s].resize(is.getNoValues(s));
    for (CatValue sv = 0; sv < is.getNoValues(s); sv++) {
      csscatweights_[s][sv].resize(is.getNoClasses()-1);
      cssnumweights_[s][sv].resize(is.getNoClasses()-1);
      for (CatValue y = 0; y < is.getNoClasses()-1; y++) {
        csscatweights_[s][sv][y].resize(s);
        for (CategoricalAttribute i = 0; i < s; i++) {
          csscatweights_[s][sv][y][i].assign(is.getNoValues(i), 0.0);
        }
        cssnumweights_[s][sv][y].assign(is.getNoNumAtts(), 0.0);
      }
    }
  }

  //nsscatweights_[s][sv][ym1][i][v]
  //nssnumweights_[s][sv][ym1][i]
  //nsslast_update_[s][sv]
  nsscatweights_.resize(is.getNoNumAtts());
  nssnumweights_.resize(is.getNoNumAtts());
  nsslast_update_.resize(is.getNoNumAtts());
  for (NumericAttribute s = 0; s < is.getNoNumAtts(); s++) {
    nsscatweights_[s].resize(noOfSubspaces_);
    nssnumweights_[s].resize(noOfSubspaces_);
    nsslast_update_[s].assign(noOfSubspaces_, 0);
    for (CatValue sv = 0; sv < noOfSubspaces_; sv++) {
      nsscatweights_[s][sv].resize(is.getNoClasses()-1);
      nssnumweights_[s][sv].resize(is.getNoClasses()-1);
      for (CatValue y = 0; y < is.getNoClasses()-1; y++) {
        nsscatweights_[s][sv][y].resize(is.getNoCatAtts());
        for (CategoricalAttribute i = 0; i < is.getNoCatAtts(); i++) {
          nsscatweights_[s][sv][y][i].assign(is.getNoValues(i), 0.0);
        }
        nssnumweights_[s][sv][y].assign(is.getNoNumAtts(), 0.0);
      }
    }

  }

  logMax_ = log(std::numeric_limits<double>::max()/is.getNoClasses());

  numsample_.resize(is.getNoNumAtts());
  cutpoints_.resize(is.getNoNumAtts());
}


void subspaceLRSGD::train(const instance &inst) {
  // make these three local variables static so as to save the overhead of repeated allocation and deallocation
  static std::vector<double> classDist;
  static std::vector<double> ymp;
  static std::vector<double> adjust;

  classDist.resize(metaData_->getNoClasses());
  ymp.resize(metaData_->getNoClasses()-1);
  adjust.resize(metaData_->getNoNumAtts());

  update_num_++;

  classify(inst, classDist);

  // First do standard sgd on individual attributes
  for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) { ///< ym1 is the class value minus 1
    const CatValue y = ym1+1;   ///< this is the class value
    if (inst.getClass() == y) ymp[ym1] = (1.0-classDist[y]);
    else ymp[ym1] = -classDist[y];
    alpha_[ym1] *= 1.0-regularisation_rate_*update_rate_;
    alpha_[ym1] += update_rate_ * ymp[ym1];
    for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
      const CatValue v = inst.getCatVal(i);
      catweights_[ym1][i][v] *= pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-last_update_cat_[i][v])); // regularise
      catweights_[ym1][i][v] += update_rate_ * ymp[ym1];
    }
  }

  for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
    if (inst.isMissing(i)) {
      // just need to update for regularisation
      for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
        numweights_[ym1][i] *= 1.0-regularisation_rate_*update_rate_; // regularise - note numeric atts are updated every cycle
      }
    }
    else {
      const double oldmean = mean_[i];
      double v = inst.getNumVal(i);
      const double newmean = oldmean + (v-oldmean)/update_num_;

      // update the sample used for maintaining the cutpoints
      if (numsample_[i].size() < sampleSize_) {
        numsample_[i].push_back(v);

        if (numsample_[i].size() == sampleSize_) {
          std::sort(numsample_[i].begin(), numsample_[i].end());
          cutpoints_[i].clear();
          for (unsigned int c=1; c < noOfSubspaces_; c++) {
            cutpoints_[i].push_back(numsample_[i][c*sampleSize_/noOfSubspaces_-1]);
          }
        }
      }
      else {
        static MTRand_int32 rand;

        const int gapIndex = rand(sampleSize_);
        std::vector<NumValue>::iterator gap = numsample_[i].begin() + gapIndex;  // the element to remove

        std::vector<NumValue>::iterator insertionPoint = std::lower_bound(numsample_[i].begin(), numsample_[i].end(), v);

        if (gap == insertionPoint) {
          *gap = v;
        }
        else if (gap < insertionPoint) {
          std::copy(gap+1, insertionPoint, gap);
          *(insertionPoint-1) = v;
        }
        else {
          std::copy_backward(insertionPoint, gap, gap+1);
          *insertionPoint = v;
        }

        // update cutpoints
        for (unsigned int c=1; c < noOfSubspaces_; c++) {
          cutpoints_[i][c-1] = numsample_[i][c*sampleSize_/noOfSubspaces_-1];
        }
      }

      // update the numeric attribute weights in the global space
      //alpha_ += (newmean-oldmean) * numweights_[i];   // rescale alpha_ for the change in means
      mean_[i] = newmean;
      const double oldRange = nummax_[i]-nummin_[i];
      if (v > nummax_[i]) nummax_[i] = v;
      if (v < nummin_[i]) nummin_[i] = v;
      if (nummax_[i]>nummin_[i]) {
        // if only one value has been observed then there is nothing to be gained by weighting this attribute
        const double newRange = nummax_[i]-nummin_[i];
        adjust[i] = oldRange/newRange;
        for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
          numweights_[ym1][i] *= adjust[i] * (1.0-regularisation_rate_*update_rate_); // regularise - note numeric atts are updated every cycle - (oldRange/newRange) rescales
          numweights_[ym1][i] += update_rate_ * ymp[ym1] * ((v-newmean)/newRange);
        }
      }
    }
  }

  // now do the subspaces of the categorical attributes
  for (CategoricalAttribute s = 0; s < metaData_->getNoCatAtts(); s++) {
    const CatValue sv = inst.getCatVal(s);

    for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) { ///< ym1 is the class value minus 1
      for (CategoricalAttribute i = 0; i < s; i++) {
        const CatValue v = inst.getCatVal(i);
        csscatweights_[s][sv][ym1][i][v] *= pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-min(last_update_cat_[i][v],last_update_cat_[s][sv]))); // regularise
        csscatweights_[s][sv][ym1][i][v] += update_rate_ * ymp[ym1];
      }
    }

    for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
      if (nummax_[i]>nummin_[i]) {
        if (inst.isMissing(i)) {
          // just need to update for regularisation
          for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
            cssnumweights_[s][sv][ym1][i] *= 1.0-regularisation_rate_*update_rate_; // regularise - note numeric atts are updated every cycle
          }
        }
        else {
          // if only one value has been observed then there is nothing to be gained by weighting this attribute
          const double newRange = nummax_[i]-nummin_[i];
          double v = inst.getNumVal(i);

          for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
              // note, we are not doing the adjustments for rescaling of the numeric attributes because we do not know what historical adjustments have taken place since the last update
            cssnumweights_[s][sv][ym1][i] *= pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-last_update_cat_[s][sv])); // regularise - note numeric atts are updated every cycle - (oldRange/newRange) rescales
            cssnumweights_[s][sv][ym1][i] += update_rate_ * ymp[ym1] * ((v-mean_[i])/newRange);
          }
        }
      }
    }
  }

  // now do the subspaces of the numeric attributes
  for (NumericAttribute s = 0; s < metaData_->getNoNumAtts(); s++) {
    if (cutpoints_[s].size() > 0) {
      // only start the subspaces for numeric attributes once there are sufficient examples to find moderately stable cutpoints

      if (!inst.isMissing(s)) {
        const NumValue nv = inst.getNumVal(s);
        CatValue sv = 0;
        while (sv < noOfSubspaces_-1 && nv > cutpoints_[s][sv]) sv++; 

        for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) { ///< ym1 is the class value minus 1
          for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
            const CatValue v = inst.getCatVal(i);
            nsscatweights_[s][sv][ym1][i][v] *= pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-min(last_update_cat_[i][v],nsslast_update_[s][sv]))); // regularise
            nsscatweights_[s][sv][ym1][i][v] += update_rate_ * ymp[ym1];
          }
        }

        for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
          if (inst.isMissing(i)) {
            // just need to update for regularisation
            for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
              nssnumweights_[s][sv][ym1][i] *= pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-nsslast_update_[s][sv])); // regularise - note numeric atts are updated every cycle
            }
          }
          else {
            if (nummax_[i]>nummin_[i]) {
              // if only one value has been observed then there is nothing to be gained by weighting this attribute
              const double newRange = nummax_[i]-nummin_[i];
              double v = inst.getNumVal(i);

              for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
                // note, we are not doing the adjustments for rescaling of the numeric attributes because we do not know what historical adjustments have taken place since the last update
                nssnumweights_[s][sv][ym1][i] *= pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-nsslast_update_[s][sv])); // regularise - note numeric atts are updated every cycle - (oldRange/newRange) rescales
                nssnumweights_[s][sv][ym1][i] += update_rate_ * ymp[ym1] * ((v-mean_[i])/newRange);
              }
            }
          }
        }

        nsslast_update_[s][sv] = update_num_;
      }
    }
  }

  // record update time for cat values that have been updated
  for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
    last_update_cat_[i][inst.getCatVal(i)] = update_num_;
  }
}


void subspaceLRSGD::initialisePass() {
}


void subspaceLRSGD::finalisePass() {
  trainingIsFinished_ = true;
}


bool subspaceLRSGD::trainingIsFinished() {
  return trainingIsFinished_;
}

void subspaceLRSGD::classify(const instance &inst, std::vector<double> &classDist) {
  classDist[0] = 1.0;

  for (CatValue y = 1; y < classDist.size(); y++) {
    double sum = alpha_[y-1];

    for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
      if (!inst.isMissing(i)) {
        if (nummax_[i]>nummin_[i]) {
          sum += numweights_[y-1][i]*((inst.getNumVal(i)-mean_[i])/(nummax_[i]-nummin_[i]));
        }
      }
    }

    for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
      const CatValue v = inst.getCatVal(i);
      sum += catweights_[y-1][i][v] * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-last_update_cat_[i][v]));
    }

    // now do the categorical subspaces
    for (CategoricalAttribute s = 0; s < metaData_->getNoCatAtts(); s++) {
      const CatValue sv = inst.getCatVal(s);

      for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
        if (!inst.isMissing(i)) {
          if (nummax_[i]>nummin_[i]) {
            sum += cssnumweights_[s][sv][y-1][i]*((inst.getNumVal(i)-mean_[i])/(nummax_[i]-nummin_[i])) * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-last_update_cat_[s][sv]));
          }
        }
      }

      for (CategoricalAttribute i = 0; i < s; i++) {
        const CatValue v = inst.getCatVal(i);
        sum += csscatweights_[s][sv][y-1][i][v] * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-min(last_update_cat_[s][sv],last_update_cat_[i][v])));
      }
    }

    // now do the numeric subspaces
    for (NumericAttribute s = 0; s < metaData_->getNoNumAtts(); s++) {
      if (cutpoints_[s].size() > 0) {
        // only start the subspaces for numeric attributes once there are sufficient examples to find moderately stable cutpoints
        const NumValue nv = inst.getNumVal(s);
        CatValue sv = 0;
        while (sv < noOfSubspaces_-1 && nv > cutpoints_[s][sv]) sv++; 

        for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
          if (!inst.isMissing(i)) {
            if (nummax_[i]>nummin_[i]) {
              sum += nssnumweights_[s][sv][y-1][i]*((inst.getNumVal(i)-mean_[i])/(nummax_[i]-nummin_[i])) * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-nsslast_update_[s][sv]));
            }
          }
        }

        for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
          const CatValue v = inst.getCatVal(i);
          sum += nsscatweights_[s][sv][y-1][i][v] * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-min(nsslast_update_[s][sv],last_update_cat_[i][v])));
        }
      }
    }

    if (sum > logMax_) {
      static bool noWarnings = true;
      sum = logMax_;
      if (noWarnings) {
        printf("WARNING: WEIGHTS ARE BEING ROUNDED DUE TO NUMERIC OVERFLOW\n");
        noWarnings = false;
      }
    }
    classDist[y] = exp(sum);
  }

  normalise(classDist);
}
