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
#include "instanceStreamFilter.h"

/**
<!-- globalinfo-start -->
 * Class for a filtered classifier.<br/>
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */
class FilteredLearner : public IncrementalLearner
{
public:
  
  /**
   * @param argv Options for the filtered classifier
   * @param end the first option after the options of rthe filtered learner
   */
  FilteredLearner(char*const*& argv, char*const* end);
  
  ~FilteredLearner(void);

  void getCapabilities(capabilities &c); 


  /**
   * Calculates the class membership probabilities for the given test instance. 
   * 
   * @param inst The instance to be classified
   * @param classDist Predicted class probability distribution
   */
  virtual void classify(const instance &inst, std::vector<double> &classDist);
  
  virtual void reset(InstanceStream &is);   ///< reset the learner prior to training
  virtual void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  virtual void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  virtual void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  virtual bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()

  
private:
  IncrementalLearner* base_learner_; ///< the base learner
  InstanceStreamFilter* filter_; ///< the filter to apply
  instance filtered_inst_;  ///< maintain one instance so as to save overhead of construction and destruction
};

