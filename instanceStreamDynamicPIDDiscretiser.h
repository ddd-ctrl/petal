/* Open source system for classification learning from very large data
** Class for a discretisation filter for instance streams
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


/**
<!-- globalinfo-start -->
 * Class for an instance stream filter that dynamically dicretises numeric attributes.<p>
 * Dynamic discretisation creates the cut points incrementally and modifies them as more data are seen.<p>
 * This version uses Partition Incremental Discretization algorithm (PiD), 
 *    Gama, J. & Pinto, C. Discretization from data streams: applications to histograms and data mining.
*     Proceedings of the 2006 ACM symposium on Applied computing, 2006, 662-667.
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

#pragma once
#include "instanceStreamDiscretiser.h"
#include "discretiser.h"
#include "utils.h"
#include <set>

class InstanceStreamDynamicPIDDiscretiser :
  public InstanceStreamDiscretiser
{
public:
  InstanceStreamDynamicPIDDiscretiser(char*const*& argv, char*const* end);
  ~InstanceStreamDynamicPIDDiscretiser(void);

  virtual void setSource(InstanceStream &source);   ///< set the source for the filter
  void rewind();                                    ///< return to the first instance in the stream
  void reset();                                     ///< rreset discretisation variables after setSource or rewind

  virtual bool advance(instance &inst);             ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 

  virtual void printStats();                        ///< print stats on the number of bins produced

  class MetaData : public InstanceStreamDiscretiser::MetaData {
  public:
    MetaData(unsigned int nBins) : noOfBins_(nBins) {}
    virtual unsigned int getNoValues(const CategoricalAttribute att) const;   ///< return the number of values for a categorical attribute
    virtual bool hasCatMissing(const CategoricalAttribute att) const;          ///< return whether a categorical attribute contains missing values
    unsigned int noOfBins_;                           ///< the number of bins for each discretised attribute
  };

private:
  double alfa_;                                 ///< magic number of splitting a level 1 bin
  unsigned int Nr_;                             ///< total count
  unsigned int noOfBins_;                       ///< the number of bins for each discretised attribute
  unsigned int level1Multiplier_;               ///< level 1 consists of level1Multiplier_ * noOfBins_ bins.
  unsigned int noOfLevel1Bins_;                 ///< the number of level 1 bins = noOfBins_ * level1Multiplier_
  std::vector<NumValue> estMin_;                ///< an initial estimate of the min
  std::vector<NumValue> estMax_;                ///< an initial estimate of the max
  std::vector<NumValue> min_;                   ///< initialised to estMin then updated as examples seen
  std::vector<NumValue> max_;                   ///< initialised to estMax then updated as examples seen
  std::vector<std::vector<NumValue> > level1Cuts_;            ///< the cuts at level 1 - the level 2 cuts are cuts_ inherited from InstanceStreamDiscretiser
  std::vector<std::vector<unsigned int> > level1Counts_;      ///< the counts of the number of examples in each level 1 bin
};
