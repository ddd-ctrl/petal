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
#include "xyDist.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "xxxxyDist.h"
/**
<!-- globalinfo-start -->
 * (Not a true learner) Class for printing dataset's statistics such as # of instances, attributes, classes, class distribution, etc.<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 *
 *
 * @author Ana M Martinez (anam.martinez@monash.edu) 
 */


enum distType{
	dtXy,
	dtXxy,
	dtXxxy,
	dtXxxxy
};


class dataStatistics : public IncrementalLearner
{
public:
  
  /**
   * @param argv Options for the dataStatistics classifier
   * @param argc Number of options for dataStatistics
   */
  dataStatistics(char*const*& argv, char*const* end);
  
  
  ~dataStatistics(void);
  
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
  bool trainingIsFinished_; ///< true iff the learner is trained
  std::vector<InstanceCount >  classCount;     ///< count of the number of instances seen so far
  InstanceCount noInstances_; ///< the number of instances.
  unsigned int noCatAtts_;  ///< the number of categorical attributes.
  unsigned int noClasses_;  ///< the number of classes
  unsigned int noNumAtts_ ;     ///< number of discrete attributes in the original dataset
  double avNoAttValues_;    ///<average number of attribute values
  std::string dataName_; //</name of the dataset

  distType dt_;

  xyDist xyDist_;///< the xy distribution
  xxyDist xxyDist_; ///< the xxy distribution that aode learns from the instance stream and uses for classification
  xxxyDist xxxyDist_;///< the xxxy distribution
  xxxxyDist xxxxyDist_;///< the xxxxy distribution
};

