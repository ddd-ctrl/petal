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
#pragma once
#include "instanceStreamFilter.h"
#include "discretiser.h"
#include <vector>


/**
<!-- globalinfo-start -->
 * Class for an instance stream filter that normalises all numeric attributes.<p>
 * Requires a pass through the data to identify min and max value for each attribute.<p>
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

class InstanceStreamNormalisationFilter :
  public ClassSafeInstanceStreamFilter
{
public:
  InstanceStreamNormalisationFilter(char*const*& argv, char*const* end);
  InstanceStreamNormalisationFilter(InstanceStream *src);
  ~InstanceStreamNormalisationFilter(void);

  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool advanceNumeric(instance &inst);                        ///< advance to the next instance in the stream without discretizing numeric values.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd() const;                                       ///< true if we have advanced past the last instance
  InstanceCount size();                                       /// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

  void setSource(InstanceStream &source);                     ///< set the source for the filter
  
  virtual void convert(const instance &inst, instance &instDisc);   ///< return the discretised version of the instance. @param inst the instance to discretise.
  
  /// return the normalise value for an attribute
  inline NumValue normalise(const NumValue v, const NumericAttribute a) const {
    const NumValue min = min_[a];
    const NumValue max = max_[a];

    if (min >= max) return 0.0;
    else if (minIsMinusOne_) return static_cast<NumValue>(2 * (v - min) / (max - min) - 1.0);
    else return static_cast<NumValue>((v - min) / (max - min));
  }

  inline MetaData* getMetaData() { return source_->getMetaData(); } // normalisation does not change the metaData

private:
  bool minIsMinusOne_;          ///< if true the minimum normalised value is -1, otherwise it is 0
  std::vector<NumValue> min_;  ///< the minimum value for each original attribute
  std::vector<NumValue> max_;  ///< the maximum value for each original attribute
};
