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

#pragma once

#include "incrementalLearner.h"
#include <vector>

/**
<!-- globalinfo-start -->
 * Multiclass Logistic Regression using stochastic gradient descent.<br/>
 * Features are dynamically normalised.<br>
 * <br/>
 <!-- globalinfo-end -->
 *
 * @author Geoff Webb (geoff.webb@monash.edu) 
 */

class LR_SGD :
  public IncrementalLearner
{
public:
  LR_SGD(char*const*& argv, char*const* end);
  ~LR_SGD(void);

  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c); 

  /**
   * Calculates the class membership probabilities for the given test instance. 
   * 
   * @param inst The instance to be classified
   * @param classDist Predicted class probability distribution
   */
  virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
  bool trainingIsFinished_;       ///< true iff the learner is trained
  bool normalise_;                ///< true iff the learner should dynamically normalise values
  std::vector<std::vector<double> > numweights_;  ///< the weight for each numeric variable, indexed by [y-1][att]
  std::vector<double> nummin_;    ///< the minimum value for each numeric att
  std::vector<double> nummax_;    ///< the maximum value for each numeric att
  std::vector<double> mean_;      ///< the mean of each numeric att
  std::vector<double> alpha_;     ///< the weight for the bias feature for each class other than 0
  std::vector<std::vector<std::vector<double> > > catweights_;  ///< the weight for each categorical variable value [y-1][att][val]
  long update_num_;               ///< the current operation
  std::vector<std::vector<int> > last_update_cat_; ///< the operation at which the corresponding weight was last updated indexed by categorical att then value
  double regularisation_rate_;    ///< mu - the regularisation rate
  double update_rate_;            ///< lambda - the update rate
  bool weightsForMissing_;        ///<  use a special weight for issing numeric values
  std::vector<std::vector<double> > missingWeights_;  ///< the weights for missing numeric values
  std::vector<int> lastMissingUpdate_;    ///< the operation at which the missing value wieght was last updated
  InstanceStream::MetaData* metaData_;
  double logMax_;                 ///< the maximum value that it is safe to exponentiate
};

