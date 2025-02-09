/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
** Implements Sahami's k-dependence Bayesian classifier
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

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "cbn.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "learnerRegistry.h"

static LearnerRegistrar registrar("cbn", constructor<cbn>);

cbn::cbn() : pass_(1)
{
}

cbn::cbn(char*const*& argv, char*const* end) : pass_(1)
{ name_ = "CBN";

  bool kSet = false;
  bool dSet = false;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (argv[0][1] == 'd') {
      getUIntFromStr(argv[0]+2, dof_, "d");
      dSet = true;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
      kSet = true;
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }

  // set defaults for k_ and dof_ if not specified
  if (!kSet && !dSet) {
    error("%s requires either +d or +k to be set", name_.c_str());
  }
  else {
    if (!kSet) {
      k_ = std::numeric_limits<unsigned int>::max();
    }

    if (!dSet) {
      dof_ = std::numeric_limits<unsigned int>::max();
    }
  }
}

cbn::~cbn(void)
{
}


void  cbn::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class
class miCmpClass {
public:
  miCmpClass(std::vector<float> *m) {
    mi = m;
  }

  bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
    return (*mi)[a] > (*mi)[b];
  }

private:
  std::vector<float> *mi;
};


void cbn::reset(InstanceStream &is) {
  metaData_ = is.getMetaData();
  const unsigned int noCatAtts = is.getNoCatAtts();
  noCatAtts_ = noCatAtts;
  noClasses_ = is.getNoClasses();

  k_ = min(k_, noCatAtts_-1);  // k cannot exceed the real number of categorical attributes - 1
  
  // initialise distributions
  dTree_.resize(noCatAtts);
  parents_.resize(noCatAtts);

  for (CategoricalAttribute a = 0; a < noCatAtts; a++) {
    parents_[a].clear();
    dTree_[a].init(is, a);
  }

  dist_.reset(is);

  classDist_.reset(is);

  pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
void cbn::train(const instance &inst) {
  if (pass_ == 1) {
    dist_.update(inst);
  }
  else {
    assert(pass_ == 2);

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      dTree_[a].update(inst, a, parents_[a]);
    }
    classDist_.update(inst);
  }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void cbn::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void cbn::finalisePass() {
  if (pass_ == 1 && k_!=0) {
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;  
    getMutualInformation(dist_.xyCounts, mi);
    
    if (verbosity >= 3) {
      printf("\nMutual information table\n");
      print(mi);
    }

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noCatAtts_);
    getCondMutualInf(dist_,cmi);
    
    dist_.clear();

    if (verbosity >= 3) {
      printf("\nConditional mutual information table\n");
      cmi.print();
    }

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      order.push_back(a);
    }

    unsigned int totalDOF = noClasses_ - 1;

    // assign the parents
    if (!order.empty()) {
      miCmpClass cmp(&mi);

      std::sort(order.begin(), order.end(), cmp);

      std::vector<CategoricalAttribute> orderedParents(order);

      if (verbosity >= 2) {
        printf("\n%s parents:\n", metaData_->getCatAttName(order[0]));
      }

      // proper KDB assignment of parents
      for (std::vector<CategoricalAttribute>::iterator it = order.begin()+1; it != order.end(); it++) {
        orderedParents.assign(order.begin(), it);

        miCmpClass cmiCmp(&cmi[*it]);

        std::sort(orderedParents.begin(), orderedParents.end(), cmiCmp);

        unsigned int dof = (metaData_->getNoValues(*it)-1) * noClasses_;

        if (dof <= dof_/2) {
          // check that it is possible to have a parent within the width constraint

          for (std::vector<CategoricalAttribute>::const_iterator p = orderedParents.begin(); dof <= dof_/2 && parents_[*it].size() < k_ && p != orderedParents.end(); p++) {
            const unsigned int nov = metaData_->getNoValues(*p);

            if (dof * nov <= dof_) {
              parents_[*it].push_back(*p);
              dof *= nov;
            }
          }
        }

        if (verbosity >= 2) {
          printf("%s parents: ", metaData_->getCatAttName(*it));
          for (unsigned int i = 0; i < parents_[*it].size(); i++) {
            printf("%s ", metaData_->getCatAttName(parents_[*it][i]));
          }
          putchar('\n');
        }

        totalDOF += dof;
      }
    }

    if (verbosity >= 1) {
      printf("\nComplexity: %d\n", totalDOF);
      printf("Classes: %d\n", noClasses_);
      printf("Attributes: %d\n", noCatAtts_);

      unsigned int totalK = 0;
      for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        totalK += parents_[a].size();
      }

      printf("Average k: %0.2f\n\n", static_cast<double>(totalK)/ noCatAtts_);
    }
  }

  ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()
bool cbn::trainingIsFinished() {
  return pass_ > 2;
}

void cbn::classify(const instance& inst, std::vector<double> &posteriorDist) {
  // calculate the class probabilities in parallel
  // P(y)
  for (CatValue y = 0; y < noClasses_; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  // P(x_i | x_p1, .. x_pk, y)
  for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
    dTree_[x].updateClassDistribution(posteriorDist, x, inst);
  }

  // normalise the results
  normalise(posteriorDist);
}



