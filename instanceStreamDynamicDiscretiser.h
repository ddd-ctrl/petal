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
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

#pragma once
#include "instanceStreamDiscretiser.h"
#include "discretiser.h"
#include "utils.h"
#include "DEPQ.h"
#include <set>

class InstanceStreamDynamicDiscretiser :
  public InstanceStreamDiscretiser
{
public:
  InstanceStreamDynamicDiscretiser(char*const*& argv, char*const* end);
  ~InstanceStreamDynamicDiscretiser(void);

  virtual void setSource(InstanceStream &source);                                 ///< set the source for the filter

  void rewind();                                                                  ///< return to the first instance in the stream
  void reset();                                                                   ///< rreset discretisation variables after setSource or rewind
  virtual bool advance(instance &inst);                                           ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  
  virtual CatValue discretise(const NumValue v, const NumericAttribute a) const;  ///< return the discretised value of a numeric attribute value. @param v the value to discretise. @param a the attribute

  void update(instance &inst);                                                    ///< update the samples using the new instance

  class MetaData : public InstanceStreamDiscretiser::MetaData {
  public:
    MetaData(unsigned int nBins) : noOfBins_(nBins) {}
    virtual unsigned int getNoValues(const CategoricalAttribute att) const;   ///< return the number of values for a categorical attribute
    virtual bool hasCatMissing(const CategoricalAttribute att) const;          ///< return whether a categorical attribute contains missing values
    unsigned int noOfBins_;                           ///< the number of bins for each discretised attribute
  };

private:

  /// uses a vector of multisets to store the (partially) sorted instances
  class sortedIndexesMultiset {
  public:
    void reset(const unsigned int nbins, const bool degenerates);   ///< reset sorted indexes
    inline unsigned int size() const { return size_; }
    void insert(const NumValue v);        ///< insert an index
    void remove(const NumValue v);        ///< remove an index
    void replace(const NumValue oldV, const NumValue nweV);        ///< replace oldV with newV
    NumValue cut(const unsigned int index) const;       ///< return a cut value
    inline CatValue discretise(const NumValue v) const;  ///< return the discretised value

  private:
    inline NumValue maxVal(const unsigned int index) const { return *values_[index].rbegin(); }

    unsigned int nBins_;                                  ///< the number of bins
    bool degenerates_;                                    ///< if true then degenerate intervals are identified and created
    unsigned int size_;                                   ///< the number of indexes currently stored
    std::vector<std::multiset<NumValue> > values_;        ///< the values in the intervals, indexed by [interval]
  };

  /// uses a vector of multisets to store the (partially) sorted instances
  class sortedIndexesDEPQ {
  public:
    void reset(const unsigned int nbins, const bool degenerates); ///< reset sorted indexes
    inline unsigned int size() const { return size_; }
    void insert(const NumValue v);                                ///< insert an value
    void insert(const NumValue v, unsigned int targetbin);        ///< insert an value shuffling across bins to increase the size of targetbin
    void replace(unsigned int index, const NumValue v);           ///< replace the indexed value with the specified value
    NumValue cut(const unsigned int index) const;                 ///< return a cut value
    inline CatValue discretise(const NumValue v) const;           ///< return the discretised value

  private:
    void replace(unsigned int targetbin, const unsigned int index, const NumValue v); ///< auxiliary function for replace(index, v)

    unsigned int nBins_;                                    ///< the number of bins
    bool degenerates_;                                      ///< if true then degenerate intervals are identified and created
    unsigned int size_;                                     ///< the number of indexes currently stored
    std::vector<DEPQ<NumValue> > values_;                   ///< the values in the intervals, indexed by [interval]
  };

  class sample {
  public:
    sample(const unsigned int noOfBins, const bool degenerates) : noOfBins_(noOfBins), degenerates_(degenerates) {}
    virtual ~sample() {}

    virtual void reset(const unsigned int noNumAtts) = 0;
    virtual void insert(const instance &inst) = 0;                                        ///< insert an instance into the sample
    virtual NumValue cut(const NumericAttribute att, const unsigned int index) const = 0; ///< return the discretised value for the attribute
    virtual CatValue discretise(const NumericAttribute att, const NumValue v) const = 0;  ///< return the discretised value for hte attribute value

  protected:
    unsigned int noOfBins_;                                 ///< the number of bins for each discretised attribute
    bool degenerates_;                                      ///< if true then degenerate intervals are identified and created
  };

  class windowSample : public sample {
  public:
    windowSample(const unsigned int noOfBins, const bool degenerates, const unsigned int sampleSize)
      : sample(noOfBins, degenerates), sampleSize_(sampleSize) {}
    ~windowSample() {}

    void reset(const unsigned int noNumAtts);
    void insert(const instance &inst);                                        ///< insert an instance into the sample
    NumValue cut(const NumericAttribute att, const unsigned int index) const; ///< return the discretised value for the attribute
    CatValue discretise(const NumericAttribute att, const NumValue v) const;  ///< return the discretised value for hte attribute value

  private:
    unsigned int sampleSize_;                               ///< the size of the window
    std::vector<sortedIndexesMultiset> sortedValues_;       ///< the sample
    std::vector<std::vector<NumValue> > insertionOrder_;    ///< a circular queue of the values in the window for each att
    std::vector<unsigned int> next_;                        ///< the head of the circular queue
  };

  class randomSample : public sample {
  public:
    randomSample(const unsigned int noOfBins, const bool degenerates, const unsigned int sampleSize, const bool alwaysUpdate)
      : sample(noOfBins, degenerates), sampleSize_(sampleSize), alwaysUpdate_(alwaysUpdate)
      {}
    ~randomSample() {}

    void reset(const unsigned int noNumAtts);
    void insert(const instance &inst);                                        ///< insert an instance into the sample
    NumValue cut(const NumericAttribute att, const unsigned int index) const; ///< return the discretised value for the attribute
    CatValue discretise(const NumericAttribute att, const NumValue v) const;  ///< return the discretised value for hte attribute value

  private:
    unsigned int sampleSize_;                               ///< the size of the window
    bool alwaysUpdate_;                                     ///< if true then every new object is added to the sample
    std::vector<InstanceCount> count_;                      ///< the number of instances observed in the stream to date, indexed by att to allow for missing values
    std::vector<sortedIndexesDEPQ> sortedValues_;
  };

  class completeSample : public sample {
  public:
    completeSample(const unsigned int noOfBins, const bool degenerates) : sample(noOfBins, degenerates) {}
    ~completeSample() {}

    void reset(const unsigned int noNumAtts);
    void insert(const instance &inst);                                        ///< insert an instance into the sample
    NumValue cut(const NumericAttribute att, const unsigned int index) const; ///< return the discretised value for the attribute
    CatValue discretise(const NumericAttribute att, const NumValue v) const;  ///< return the discretised value for hte attribute value

  private:
    std::vector<sortedIndexesDEPQ> sortedValues_;
  };

  sample* sample_;                                        ///< the sample
  unsigned int noOfBins_;                                 ///< the number of bins for each discretised attribute
  bool preUpdate_;                                        ///< true iff the intervals should be updated before discretizing a new instance
  bool degenerates_;                                      ///< if true then degenerate intervals are identified and created
};
