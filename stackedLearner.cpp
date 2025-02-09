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

#include "stackedLearner.h"
#include "learnerRegistry.h"
#include "utils.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>

StackedLearner::StackedLearner(char*const*& argv, char*const* end)
{  name_ = "STACKED";

  // defaults
  capabilities_.setCatAtts(true);
  capabilities_.setNumAtts(true);

  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (argv[0][1] == 'b') {
      // add a base learner
      char* blname = argv[0]+2;
      base_learners_.push_back(dynamic_cast<IncrementalLearner*>(createLearner(blname, ++argv, end)));
      
      if (base_learners_.back() == NULL) {
        error("Learner %s is not supported", argv[0]+2);
      }

      name_ += "_";
      name_ += *base_learners_.back()->getName();

      capabilities c;
      base_learners_.back()->getCapabilities(c);
      if (!c.getCatAtts()) capabilities_.setCatAtts(false);
      if (!c.getNumAtts()) capabilities_.setNumAtts(false);
    }
    else if (argv[0][1] == 'm') {
      // create the meta learner
      char* mlname = argv[0]+2;
      meta_learner_ = dynamic_cast<IncrementalLearner*>(createLearner(mlname, ++argv, end));
      if (meta_learner_ == NULL) {
        error("Learner %s is not supported", argv[0]+2);
      }
    }
    else {
      break;
    }
  }

  if (base_learners_.size() == 0) error("No base learner specified");
  if (meta_learner_ == NULL) error("No meta learner specified");

  name_ += ">";
  name_ += *meta_learner_->getName();
}


StackedLearner::~StackedLearner(void) {
  for (unsigned int i = 0; i < base_learners_.size(); ++i) {
    delete base_learners_[i];
  }

  delete meta_learner_;
}

void  StackedLearner::getCapabilities(capabilities &c){
  c = capabilities_;
}

// reset the learner prior to training
void StackedLearner::reset(InstanceStream &is) {
  base_outcomes_.setNoClasses(is.getNoClasses());
  base_outcomes_.setNoNumAtts(is.getNoClasses()*base_learners_.size()); // one attribute per class per base learner
  base_outcomes_.reset();
  //metaData_ = base_outcomes_.getMetaData();

  for (std::vector<IncrementalLearner*>::iterator it = base_learners_.begin(); it != base_learners_.end(); it++) (*it)->reset(base_outcomes_);

  meta_learner_->reset(base_outcomes_);

  meta_inst_.init(base_outcomes_);
}

// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void StackedLearner::initialisePass() {
}

// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void StackedLearner::finalisePass() {
}

// true iff no more passes are required. updated by finalisePass()
bool StackedLearner::trainingIsFinished() {
  bool result = true;

  for (std::vector<IncrementalLearner*>::iterator it = base_learners_.begin(); it != base_learners_.end(); it++) 
    if (!(*it)->trainingIsFinished()) result = false;

  return result;
}

// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
void StackedLearner::train(const instance &inst) {
  std::vector<IncrementalLearner*>::iterator it;
  std::vector<double> classDist(base_outcomes_.getNoClasses());

  base_outcomes_.setClass(meta_inst_, inst.getClass());

  NumericAttribute n = 0;

  for (it = base_learners_.begin(); it != base_learners_.end(); it++) {
    (*it)->classify(inst, classDist);

    for (std::vector<double>::const_iterator v = classDist.begin(); v != classDist.end(); v++) {
      base_outcomes_.setNumVal(meta_inst_, n++, static_cast<NumValue>(*v));
    }

    (*it)->train(inst);
  }

  meta_learner_->train(meta_inst_);
}


void StackedLearner::classify(const instance &inst, std::vector<double> &classDist) {
  std::vector<IncrementalLearner*>::iterator it;
  std::vector<double> baseClassDist(base_outcomes_.getNoClasses());

  base_outcomes_.setClass(meta_inst_, inst.getClass());

  NumericAttribute n = 0;

  for (it = base_learners_.begin(); it != base_learners_.end(); it++) {
    (*it)->classify(inst, baseClassDist);

    for (std::vector<double>::const_iterator v = baseClassDist.begin(); v != baseClassDist.end(); v++) {
      base_outcomes_.setNumVal(meta_inst_, n++, static_cast<NumValue>(*v));
    }
  }

  meta_learner_->classify(meta_inst_, classDist);
}



