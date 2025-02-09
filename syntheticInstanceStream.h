/* Open source system for classification learning from very large data
** Class for an input source for instances
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

//Produce a stream of instances of the changing STAGGER concepts:
// M. A. Maloof and R. S. Michalski. Selecting examples for partial memory learning. Machine
// Learning, 41:27–52, 2000.

#pragma once

#include "instanceStream.h"
#include "mtrand.h"
#include "utils.h"

#include <assert.h>

class SyntheticInstanceStream : public InstanceStream
{
public:
  SyntheticInstanceStream(char*const*& argv, char*const* argvEnd);
  ~SyntheticInstanceStream(void);

  virtual void rewind();                                              ///< return to the first instance in the stream
  virtual bool advance();                                             ///< advance, discarding the next instance in the stream.  Return true iff successful.
  virtual bool advance(instance &inst);                               ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  virtual bool isAtEnd() const;                                             ///< true if we have advanced past the last instance
  virtual InstanceCount size();                                       ///< the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.
  
  inline void advanceTime() { time_++; }                              ///< advance time - concepts change over time

  class SyntheticMetaData : public InstanceStream::MetaData {
  public:
    virtual unsigned int getNoClasses() const;                          ///< return the number of classes
    virtual const char* getClassName(const CatValue y) const;               ///< return the name for a class
    virtual const char* getClassAttName() const;                        ///< return the name for the class attribute
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
    virtual bool areNamesCaseSensitive() const;                               ///< true iff name comparisons are case sensitive
    virtual void setAllAttsMissing();

    std::vector<char*> classNames_;
    char* classAttName_;
    std::vector<char*> catAttNames_;
    std::vector<std::vector<char*> > catAttValNames_;
    std::vector<char*> numAttNames_;
    char* name_;
  };

private:
  typedef enum {abruptDrift, gradualDrift, incrementalDrift, noDrift} DriftType;
  typedef enum {normalDistribution, skewedDistribution} DistributionType;
  DriftType driftType_;
  DistributionType distributionType_;
  bool subclasses_;
  unsigned int time_;
  unsigned int seed_;
  SyntheticMetaData metaData_;
  MTRand_closed rand_;
  RandNorm randNorm_;
  //static const int theSize_ = 120;
  unsigned int noAtts_;
  std::vector<std::vector<double> > means_;       ///< the mean for each class and att
  std::vector<std::vector<double> > stddev_;      ///< the standard deviation for each class and att
  double inflation_;            ///< the total amount by which to inflate each mean over the period of the stream
  unsigned int duration_;           ///< the length of the stream
  double p_;  ///< the probability of class 1
};
