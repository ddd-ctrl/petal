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

/**
<!-- globalinfo-start -->
 * Class for (k-)selective k-dependence Bayesian classifier, attribute selection using leave-one-out cross validation (loocv).<br/>
 <!-- globalinfo-end -->
 *
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */

class kdbSelective : public kdb
{
public:
  kdbSelective(char*const*& argv, char*const* end);
  ~kdbSelective(void);

  void reset(InstanceStream &is);   
  virtual void train(const instance &inst);
  virtual void finalisePass();
  bool trainingIsFinished();        
  void getCapabilities(capabilities &c);
  
  virtual void classify(const instance &inst, std::vector<double> &classDist);

  void printClassifier();

private:
  unsigned int minCount_;    ///< minimum count required for probability estimation
  bool selectiveTest_;       ///< attribute selection if significant difference (binomial test).
  bool selectiveWeighted_;   ///< use weighted RMSE for attribute selection.
  bool selectiveMCC_;        ///< used Matthews Correlation Coefficient for selection
  bool selectiveK_;          ///< selects the best k value
  bool selectiveLinks_;      ///< select the number of parents based on the best so far for each attribute, only with selectiveK
  
  std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
  unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest
  InstanceCount trainSize_;  ///< number of examples for training, to calculate the RMSE for each LOOCV -- flags: selective, selectiveTest
  std::vector<double> foldLossFunct_;  ///< loss function for every additional attribute (foldLossFunct_[noCatAtt]: only the class is considered) -- flags: selective, selectiveTest
  std::vector<std::vector<double> > foldLossFunctallK_;  ///< loss function for every additional attribute (foldSumRMSE[noCatAtt]: only the class is considered) for every k -- flags: selectiveK
  //unsigned int bestK_;      ///< number of parents selected by kdb selective with selectiveK option
  std::vector<InstanceCount> binomialTestCounts_;  ///< number of wins compared to considering all the atts -- flags:  selectiveTest
  std::vector<InstanceCount> sampleSizeBinomTest_; ///< Sample size for a binomial test, leaving out ties -- flags: selectiveTest
//  std::vector<InstanceCount> TP_;  ///< Store TP count (needed for selectiveMCC)
//  std::vector<InstanceCount> FP_;  ///< Store FP count (needed for selectiveMCC)
//  std::vector<InstanceCount> TN_;  ///< Store TN count (needed for selectiveMCC)
//  std::vector<InstanceCount> FN_;  ///< Store FN count (needed for selectiveMCC)
  //std::vector<std::vector<InstanceCount> > TPallK_;  ///< Store TP count (needed for selectiveMCC with selectiveK)
  //std::vector<std::vector<InstanceCount> > FPallK_;  ///< Store FP count (needed for selectiveMCC with selectiveK)
  //std::vector<std::vector<InstanceCount> > TNallK_;  ///< Store TN count (needed for selectiveMCC with selectiveK)
  //std::vector<std::vector<InstanceCount> > FNallK_;  ///< Store FN count (needed for selectiveMCC with selectiveK)
  std::vector< std::vector< crosstab<InstanceCount> > > xtab_; ///< confusion matrix for all k values and all attributes (needed for selectiveMCC with selectiveK), only k=0 is used for plain selective
  std::vector<double> prior_;///< the prior for each class, saved at end of first pass -- flags: selectiveWeighted
  std::vector<CategoricalAttribute> order_;        ///< record the attributes in order based on different criteria
  std::vector<unsigned int> bestKatt_;                ///< indicates the number of parents/links selected for each attribute (needed for selectiveLinks_)
};

