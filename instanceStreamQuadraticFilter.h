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

// creates a stream with additional attributes which are the product of each of the original attributes

class InstanceStreamQuadraticFilter :
  public ClassSafeInstanceStreamFilter
{
public:
  InstanceStreamQuadraticFilter(char*const*& argv, char*const* end);
  InstanceStreamQuadraticFilter(InstanceStream *src);
  ~InstanceStreamQuadraticFilter(void);

  void rewind();                                              ///< return to the first instance in the stream
  bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool advanceNumeric(instance &inst);                        ///< advance to the next instance in the stream without discretizing numeric values.  Return true iff successful. @param inst the instance record to receive the new instance. 
  bool isAtEnd() const;                                             ///< true if we have advanced past the last instance
  InstanceCount size();                                       /// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

  void setSource(InstanceStream &source);                     ///< set the source for the filter
  
  virtual void convert(const instance &inst, instance &instDisc);                      ///< return the discretised version of the instance. @param inst the instance to discretise. 

  class QFMetaData : public InstanceStream::MetaDataFilter {
  public:
    QFMetaData();
    virtual unsigned int getNoCatAtts() const;                          ///< return the number of categorical attributes
    virtual bool hasCatMissing(const CategoricalAttribute att) const;          ///< return whether a categorical attribute contains missing values
    virtual bool hasNumMissing(const NumericAttribute att) const;              ///< return whether a numeric attribute contains missing values
    virtual unsigned int getNoValues(const CategoricalAttribute att) const;   ///< return the number of values for a categorical attribute
    virtual const char* getCatAttName(const CategoricalAttribute att) const;  ///< return the name for a categorical Attribute
    virtual const char* getCatAttValName(const CategoricalAttribute att, const CatValue val) const; ///< return the name for a categorical attribute value
    virtual unsigned int getNoNumAtts() const;                          ///< return the number of numeric attributes
    virtual const char* getNumAttName(const NumericAttribute att) const;      ///< return the name for a numeric attribute
    virtual unsigned int getPrecision(const NumericAttribute att) const;      ///< return the precision to which values of a numeric attribute should be output
    virtual const char* getName() const ;                                      ///< return a string that gives a meaningful name for the stream

  public:
    std::vector<unsigned int> precision_;  ///< the precision for each numeric attribute
    std::vector<unsigned int> no_values_;  ///< the number of values for each categorical attribute
    std::vector<bool> hasCatMissing_; ///< whether each categorical attribute has missing values
    std::vector<bool> hasNumMissing_; ///< whether each numeric attribute has missing values
  };

  inline MetaData* getMetaData() { return &metaData_; }

private:
  unsigned int noOrigNum_;               ///< the number of original numeric atts
  std::vector<unsigned int> noOrigVals_; ///< the number of values for the original categorical atts
  instance sourceInst_;                  ///< the current instance from the source stream. Maintain one instance record to save repeated construction/destruction.
  QFMetaData metaData_;                  ///< the revised metadata
};
