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

#pragma once

#include "kdb.h"
#include "xxxyDist.h"
#include "mtrand.h"

/**
<!-- globalinfo-start -->
 * Class for (k-)selective k-dependence Bayesian classifier, attribute selection using leave-one-out cross validation (loocv). Clean version with no options, use this one if the number of classes is large. <br/>
 <!-- globalinfo-end -->
 *
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */

class kdbSelectiveClean : public kdb
{
public:
  kdbSelectiveClean(char*const*& argv, char*const* end);
  ~kdbSelectiveClean(void);

  void reset(InstanceStream &is);   
   void trainPass3(const instance &inst);
  virtual void train(const instance &inst);
  virtual void finalisePass();
  bool trainingIsFinished();        
  void getCapabilities(capabilities &c);
  
  virtual void classify(const instance &inst, std::vector<double> &classDist);

  void printClassifier();

private:
  bool selectiveK_;          ///< selects the best k value
  bool onlyK_;               ///< only selects the best k value, not attribute selection
   
  std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
  unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest
  InstanceCount trainSize_;  ///< number of examples for training, to calculate the RMSE for each LOOCV -- flags: selective, selectiveTest
  std::vector<double> foldLossFunct_;  ///< loss function for every additional attribute (foldLossFunct_[noCatAtt]: only the class is considered) -- flags: selective, selectiveTest
  std::vector<std::vector<double> > foldLossFunctallK_;  ///< loss function for every additional attribute (foldSumRMSE[noCatAtt]: only the class is considered) for every k -- flags: selectiveK
  std::vector< std::vector< crosstab<InstanceCount> > > xtab_; ///< confusion matrix for all k values and all attributes (needed for selectiveMCC with selectiveK), only k=0 is used for plain selective
  std::vector<CategoricalAttribute> order_;        ///< record the attributes in order based on different criteria
  unsigned int bestK_;                ///< indicates the number of parents/links selected for each attribute (needed for selectiveLinks_)
  MTRand_open rand_;                  ///< random number generator for selecting cache_
  MTRand_int32 randInt_;              ///< random number generator for selecting cache_
  InstanceCount testSetSize_;         ///< the number of test cases to use in pass 3
  InstanceCount testSetSoFar_;        ///< used in pass 3 to count how many test cases have been used so far
  InstanceCount seenSoFar_;           ///< used in pass 3 to count how many cases from the input stream have been seen so far
  bool useCache_;                     ///< cache a sample in pass 2 for use in pass 3
  std::vector<instance> cache_;       ///< a cache of instances to be collected in pass 2 and used in pass three
};

