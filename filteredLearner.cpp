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


#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
//#ifndef DBG_NEW
//#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
//#define new DBG_NEW
//#endif
#include <stdlib.h>
#include <crtdbg.h>
#endif
#endif

#include "filteredLearner.h"
#include "RFDTree.h"
#include "learnerRegistry.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "utils.h"
#include "mtrand.h"

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "instanceStreamQuadraticFilter.h"
#include "learnerRegistry.h"

static LearnerRegistrar registrar("filtered", constructor<FilteredLearner>);

FilteredLearner::FilteredLearner(char*const*& argv, char*const* end) 
{ name_ = "FILTERED";

  // defaults
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (argv[0][1] == 'b') {
      // create the base learner
      char* learnerName = argv[0]+2;
      learner* l = createLearner(learnerName, ++argv, end);
      
      if (l == NULL) {
        error("Learner %s is not supported", learnerName);
      }

      base_learner_ = dynamic_cast<IncrementalLearner*>(l);
      
      if (base_learner_ == NULL) {
        error("Filtered classifier must be incremental");
      }

      name_ += "_";
      name_ += *(base_learner_->getName());
    }
    else if (argv[0][1] == 'f') {
      if (streq(argv[0]+2, "quadratic", false)) {
        filter_ = new InstanceStreamQuadraticFilter(++argv, end);
        name_ += "_quadratic";
      }
      else {
        error("Filter %s is not supported", argv[0]+2);
      }
    }
    else {
      break;
    }
  }

  if (base_learner_ == NULL) error("No base learner specified");
  if (filter_ == NULL) error("No filter specified");
}


FilteredLearner::~FilteredLearner(void) {
  delete base_learner_;
  delete filter_;
}

void  FilteredLearner::getCapabilities(capabilities &c){
  base_learner_->getCapabilities(c);
}

void FilteredLearner::train(const instance &inst) {
  filter_->convert(inst, filtered_inst_);
  base_learner_->train(filtered_inst_);
}


void FilteredLearner::classify(const instance &inst, std::vector<double> &classDist) {
  filter_->convert(inst, filtered_inst_);
  base_learner_->classify(filtered_inst_, classDist);
}


// reset the learner prior to training
void FilteredLearner::reset(InstanceStream &is) {
  filter_->setSource(is);
  base_learner_->reset(*filter_);
  filtered_inst_.init(*filter_);
}

// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void FilteredLearner::initialisePass() {
  base_learner_->initialisePass();
}

// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void FilteredLearner::finalisePass() {
  base_learner_->finalisePass();
}
 
// true iff no more passes are required. updated by finalisePass()
bool FilteredLearner::trainingIsFinished() {
  return base_learner_->trainingIsFinished();
}


