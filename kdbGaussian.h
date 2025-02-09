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

#include <limits>

#include "instanceStreamDiscretiser.h"
#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "yDist.h"

/**
<!-- globalinfo-start -->
 * Class for a kDB that uses Gaussian distributions for numeric attributes, 
 * it should always be run with a discretization option, e.g. dmdl.<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */


class kdbGaussian : public IncrementalLearner{
public:
  kdbGaussian();
  kdbGaussian(char*const*& argv, char*const* end);
  virtual ~kdbGaussian();
  
  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void train(InstanceStream &is);   ///< Require for a hybrid classifier
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c);

  virtual void classify(const instance &inst, std::vector<double> &classDist);

protected:
  unsigned int pass_;                                        ///< the number of passes for the learner
  unsigned int k_;                                           ///< the maximum number of parents
  unsigned int noNumAtts_;                                   ///< the number of categorical attributes.
  unsigned int noOrigCatAtts_;                               ///< the number of categorical attributes.
  unsigned int noClasses_;                                   ///< the number of classes
  xxyDist dist_;                                             // used in the first pass
  yDist classDist_;                                          // used in the second pass and for classification
  std::vector<distributionCatTree> dCatTree_;                      // used in the second pass and for classification
  std::vector<distributionNumTree> dNumTree_;
  std::vector<std::vector<CategoricalAttribute> > parents_;
  InstanceStream* instanceStream_;
  bool useMean4Miss_;                                        // if true, the mean of the (conditional) Gaussian dist. is used instead of ignoring the value
  InstanceStreamDiscretiser *discretisingStream_; ///< the InstanceStreamDiscretiser used to discretize instances 
};


