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
#include <cmath>
#include <limits>

#include "utils.h"
#include "LR_SGD.h"

#include "learnerRegistry.h"

static LearnerRegistrar registrar("lrsgd", constructor<LR_SGD>);

LR_SGD::LR_SGD(char*const*& argv, char*const* end)
{
  name_ = "LRSGD";
  normalise_ = true;
  weightsForMissing_ = false;
  
  // defaults
  regularisation_rate_ = 0.001;
  update_rate_ = 0.001;

  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (streq(argv[0]+1, "nonorm")) {
      normalise_ = false;
    }
    else if (argv[0][1] == 'r') {
      getFloatFromStr(argv[0]+2, regularisation_rate_, "Regularisation Rate");
    }
    else if (argv[0][1] == 'u') {
      getFloatFromStr(argv[0]+2, update_rate_, "Update Rate");
    }
    else if (streq(argv[0]+1, "weights-for-missing")) {
      weightsForMissing_ = true;
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}


LR_SGD::~LR_SGD(void)
{
}


void  LR_SGD::getCapabilities(capabilities &c){
  c.setCatAtts(true);
  c.setNumAtts(true);
}

void LR_SGD::reset(InstanceStream &is) {
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

  if (weightsForMissing_) {
    missingWeights_.resize(is.getNoClasses()-1);
    for (CatValue ym1 = 0; ym1 < is.getNoClasses()-1; ym1++) {
      missingWeights_[ym1].assign(is.getNoNumAtts(), 0.0);
    }
    lastMissingUpdate_.assign(is.getNoNumAtts(), 0);
  }

  logMax_ = log(std::numeric_limits<double>::max()/is.getNoClasses());
}


void LR_SGD::train(const instance &inst) {
  static std::vector<double> classDist;
  static std::vector<double> ymp;

  classDist.resize(metaData_->getNoClasses());
  ymp.resize(metaData_->getNoClasses()-1);

  update_num_++;

  classify(inst, classDist);

  for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) { ///< ym1 is the class value minus 1
    const CatValue y = ym1+1;   ///< this is the class value
    if (inst.getClass() == y) ymp[ym1] = (1.0-classDist[y]);
    else ymp[ym1] = -classDist[y];
    alpha_[ym1] *= 1.0-regularisation_rate_*update_rate_;
    alpha_[ym1] += update_rate_ * ymp[ym1];
    for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
      const CatValue v = inst.getCatVal(i);
      catweights_[ym1][i][v] *= pow(1.0-regularisation_rate_*update_rate_, update_num_-last_update_cat_[i][v]); // regularise
      catweights_[ym1][i][v] += update_rate_ * ymp[ym1];
    }
  }

  for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
    last_update_cat_[i][inst.getCatVal(i)] = update_num_;
  }

  for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
    if (inst.isMissing(i)) {
      if (weightsForMissing_) {
        // update using the missing weights
        for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
          missingWeights_[ym1][i] *= pow(1.0-regularisation_rate_*update_rate_, update_num_-lastMissingUpdate_[i]); // regularise
          missingWeights_[ym1][i] += update_rate_ * ymp[ym1];
        }
        lastMissingUpdate_[i] = update_num_;
      }

      // need to update for regularisation
      for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
        numweights_[ym1][i] *= 1.0-regularisation_rate_*update_rate_; // regularise - note numeric atts are updated every cycle
      }
    }
    else {
      const double oldmean = mean_[i];
      const double v = inst.getNumVal(i);
      const double newmean = oldmean + (v-oldmean)/update_num_;
      //alpha_ += (newmean-oldmean) * numweights_[i];   // rescale alpha_ for the change in means
      mean_[i] = newmean;
      const double oldRange = nummax_[i]-nummin_[i];
      if (v > nummax_[i]) nummax_[i] = v;
      if (v < nummin_[i]) nummin_[i] = v;
      if (nummax_[i]>nummin_[i]) {
        // if only one value has been observed then there is nothing to be gained by weighting this attribute
        if (normalise_) {
          const double newRange = nummax_[i]-nummin_[i];
          const double adjust = oldRange/newRange;
          for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
            numweights_[ym1][i] *= adjust * (1.0-regularisation_rate_*update_rate_); // regularise - note numeric atts are updated every cycle - (oldRange/newRange) rescales
            numweights_[ym1][i] += update_rate_ * ymp[ym1] * ((v-newmean)/newRange);
          }
        }
        else {
          for (CatValue ym1 = 0; ym1 < metaData_->getNoClasses()-1; ym1++) {
            numweights_[ym1][i] *= 1.0-regularisation_rate_*update_rate_; // regularise - note numeric atts are updated every cycle
            numweights_[ym1][i] += update_rate_ * ymp[ym1] * v;
          }
        }
      }
    }
  }
}


void LR_SGD::initialisePass() {
}


void LR_SGD::finalisePass() {
  trainingIsFinished_ = true;
}


bool LR_SGD::trainingIsFinished() {
  return trainingIsFinished_;
}

void LR_SGD::classify(const instance &inst, std::vector<double> &classDist) {
  classDist[0] = 1.0;

  for (CatValue y = 1; y < classDist.size(); y++) {
    double sum = alpha_[y-1];
    for (NumericAttribute i = 0; i < metaData_->getNoNumAtts(); i++) {
      if (inst.isMissing(i)) {
        if (weightsForMissing_) {
          sum += missingWeights_[y-1][i] * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-lastMissingUpdate_[i]));
        }
      }
      else {
        if (nummax_[i]>nummin_[i]) {
          // don't update weights if only one value has been seen as the attribute cannot have carried any information
          NumValue v = inst.getNumVal(i);
          if (normalise_) v = (v-mean_[i])/(nummax_[i]-nummin_[i]);
          sum += numweights_[y-1][i]*v;
        }
      }
    }
    for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
      const CatValue v = inst.getCatVal(i);
      sum += catweights_[y-1][i][v] * pow(1.0-regularisation_rate_*update_rate_, static_cast<long>(update_num_-last_update_cat_[i][v]));
    }

    if (sum > logMax_) {
      static bool noWarnings = true;  // use this flag to ensure the error message is only issued once
      if (noWarnings) {
        printf("WARNING: WEIGHTS ARE BEING ROUNDED DUE TO NUMERIC OVERFLOW\n");
        noWarnings = false;
      }

      sum = logMax_;
    }
    classDist[y] = exp(sum);
  }

  normalise(classDist);
}
