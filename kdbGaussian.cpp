/* Open source system for classification learning from very large data
 * Copyright (C) 2012 Geoffrey I Webb
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#include "kdbGaussian.h"

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>
#include <cmath>

#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStreamDiscretiser.h"

kdbGaussian::kdbGaussian() : pass_(1)
{
}

kdbGaussian::kdbGaussian(char*const*& argv, char*const* end) : pass_(1)
{ name_ = "KDB-GAUSSIAN";
  discretisingStream_ = NULL;

  useMean4Miss_ = false; //missing values on numeric attributes are ignored by default.

  // defaults
  k_ = 1;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }
    else if (argv[0][1] == 'd') {
      char* const discName = argv[0]+2;
      discretisingStream_ = new InstanceStreamDiscretiser(discName, ++argv, end);
    }
    else if(argv[0][1] == 'M'){
        useMean4Miss_ = true;
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
  
  noOrigCatAtts_ = 0;
  noNumAtts_ = 0;

  if (discretisingStream_==NULL)
    error("Error: Learner %s must be called in combination with a discretization option, e.g. -dmdl", name_.c_str());
}

kdbGaussian::~kdbGaussian(void)
{
}


void  kdbGaussian::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
  c.setNumAtts(true);
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


void kdbGaussian::reset(InstanceStream &is) {
  instanceStream_ = &is;
  const unsigned int noAtts = is.getNoCatAtts();
  
  noOrigCatAtts_ = is.getMetaData()->getNoCatAtts();
  noNumAtts_ = noAtts - noOrigCatAtts_;
  noClasses_ = is.getNoClasses();

  k_ = min(k_, noAtts-1);  // k cannot exceed the real number of categorical attributes - 1
  
  // initialise distributions
  dCatTree_.resize(noOrigCatAtts_);
  dNumTree_.resize(noNumAtts_);
  parents_.resize(noAtts);

  for (CategoricalAttribute a = 0; a < noOrigCatAtts_; a++) {
    parents_[a].clear();
    dCatTree_[a].init(is, a);
  }
  
  for (NumericAttribute a = 0; a < noNumAtts_; a++) {
    parents_[noOrigCatAtts_ + a].clear();
    dNumTree_[a].init(is);
  } 

  dist_.reset(is);

  classDist_.reset(is);

  pass_ = 1;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
void kdbGaussian::train(const instance &inst) {
    
  instance instDisc(*discretisingStream_);
  discretisingStream_->convert(inst, instDisc);
  
  if (pass_ == 1) {
    dist_.update(instDisc);
  }
  else {
    assert(pass_ == 2);

    for (CategoricalAttribute a = 0; a < noOrigCatAtts_; a++) {
      dCatTree_[a].update(instDisc, a, parents_[a]);
    }
    
    for (NumericAttribute a = 0; a < noNumAtts_; a++) {
      if(!inst.isMissing(a)){
        dNumTree_[a].update(instDisc, parents_[noOrigCatAtts_ + a], inst.getNumVal(a));
      }
    }
    
    classDist_.update(instDisc);
  }
}

void kdbGaussian::train(InstanceStream &is){
  instance inst;
  
  testCapabilities(is);
  
  reset(is);

  while (!trainingIsFinished()) {
    initialisePass();
    is.rewind();
    while (is.advance(inst)) {
      train(inst);
    }
    finalisePass();
  }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
void kdbGaussian::initialisePass() {
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
void kdbGaussian::finalisePass() {
  if (pass_ == 1 && k_!=0) {
      
    unsigned int noAtts = noOrigCatAtts_ + noNumAtts_;
    // calculate the mutual information from the xy distribution
    std::vector<float> mi;  
    getMutualInformation(dist_.xyCounts, mi);
    
    if (verbosity >= 3) {
      printf("\nMutual information table\n");
      print(mi);
    }

    // calculate the conditional mutual information from the xxy distribution
    crosstab<float> cmi = crosstab<float>(noAtts);
    getCondMutualInf(dist_,cmi);
    
    dist_.clear();

    if (verbosity >= 3) {
      printf("\nConditional mutual information table\n");
      cmi.print();
    }

    // sort the attributes on MI with the class
    std::vector<CategoricalAttribute> order;

    for (CategoricalAttribute a = 0; a < noAtts; a++) {
      order.push_back(a);
    }

    // assign the parents
    if (!order.empty()) {
      miCmpClass cmp(&mi);

      std::sort(order.begin(), order.end(), cmp);

      if (verbosity >= 2) {
        printf("\n%s parents:\n", instanceStream_->getCatAttName(order[0]));
      }

      // proper KDB assignment of parents
      for (std::vector<CategoricalAttribute>::const_iterator it = order.begin()+1; it != order.end(); it++) {
        parents_[*it].push_back(order[0]);
        for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin()+1; it2 != it; it2++) {
          // make parents into the top k attributes on mi that precede *it in order
          if (parents_[*it].size() < k_) {
            // create space for another parent
            // set it initially to the new parent.
            // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
            parents_[*it].push_back(*it2);
          }
          for (unsigned int i = 0; i < parents_[*it].size(); i++) {
            if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it]) {
              // move lower value parents down in order
              for (unsigned int j = parents_[*it].size()-1; j > i; j--) {
                parents_[*it][j] = parents_[*it][j-1];
              }
              // insert the new att
              parents_[*it][i] = *it2;
              break;
            }
          }
        }

        if (verbosity >= 2) {
          printf("%s parents: ", instanceStream_->getCatAttName(*it));
          for (unsigned int i = 0; i < parents_[*it].size(); i++) {
            printf("%s ", instanceStream_->getCatAttName(parents_[*it][i]));
          }
          putchar('\n');
        }
      }
    }
  }

  ++pass_;
}

/// true iff no more passes are required. updated by finalisePass()
bool kdbGaussian::trainingIsFinished() {
  return pass_ > 2;
}

void kdbGaussian::classify(const instance& inst, std::vector<double> &posteriorDist) {
  // calculate the class probabilities in parallel
  // P(y)
    
  instance instDisc(*discretisingStream_);
  discretisingStream_->convert(inst, instDisc);
  
  for (CatValue y = 0; y < noClasses_; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  // P(x_i | x_p1, .. x_pk, y)
  for (CategoricalAttribute x = 0; x < noOrigCatAtts_; x++) {
    dCatTree_[x].updateClassDistribution(posteriorDist, x, instDisc);
  }
 
  for (NumericAttribute x = 0; x < noNumAtts_; x++) {
    if(!inst.isMissing(x)){
      dNumTree_[x].updateClassDistribution(posteriorDist, instDisc, inst.getNumVal(x));
    }else if(useMean4Miss_){        
        dNumTree_[x].updateClassDistribution(posteriorDist, instDisc);//use mean for missing values
    }        
  }

  // normalise the results
  normalise(posteriorDist);
}




