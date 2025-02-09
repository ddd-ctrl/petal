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

/**
<!-- globalinfo-start -->
 * Class for a bagged classifier.<br/>
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */
class StackedLearner : public IncrementalLearner
{
public:
  
  /**
   * @param argv Options for the NB classifier
   * @param argc Number of options for NB
   */
  StackedLearner(char*const*& argv, char*const* end);
  
  ~StackedLearner(void);

  virtual void reset(InstanceStream &is);   ///< reset the learner prior to training
  virtual void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  virtual void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  virtual void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  virtual bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()

  void getCapabilities(capabilities &c); 

  /**
   * Calculates the class membership probabilities for the given test instance. 
   * 
   * @param inst The instance to be classified
   * @param classDist Predicted class probability distribution
   */
  virtual void classify(const instance &inst, std::vector<double> &classDist);
  
  
private:
  /// A dummy instance stream to pass to the stacked learner
  /// All that it needs to do is create an appropriate instance and return the correct metadata
  class slInstanceStream: public InstanceStream {
  public:
    virtual void rewind() {};                                              ///< return to the first instance in the stream
    virtual bool advance() {return true;};                                 ///< advance, discarding the next instance in the stream.  Return true iff successful.
    virtual bool advance(instance &inst) {return true;};                   ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
    virtual bool isAtEnd() const {return false;};                          ///< true if we have advanced past the last instance
    virtual InstanceCount size() {return 0;};                              ///< the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

    //inline void setNumVal(instance &inst, const NumericAttribute att, const NumValue v) { InstanceStream:setNumVal(inst, att, v); }
    //inline void setClass(instance &inst, const CatValue v) { InstanceStream:setClass(inst, v); }
 
    //inline unsigned int getNoClasses() { return metaData_.getNoClasses(); };

    inline void setNoClasses(unsigned int n) {mfMetaData_.setNoClasses(n);};
    inline void setNoNumAtts(unsigned int n) {mfMetaData_.setNoNumAtts(n);};

    inline void reset() {metaData_ = &mfMetaData_;};

    class MetaDataFilter : public MetaData {
    public:

      unsigned int getNoClasses() const {return no_classes_;};                          ///< return the number of classes
      const char* getClassName(const CatValue y) const {return NULL;};               ///< return the name for a class
      const char* getClassAttName() const {return NULL;};                        ///< return the name for the class attribute
      unsigned int getNoCatAtts() const {return 0;};                          ///< return the number of categorical attributes
      virtual bool hasCatMissing(const CategoricalAttribute att) const {return false;};          ///< return whether a categorical attribute contains missing values
      virtual bool hasNumMissing(const NumericAttribute att) const {return false;};              ///< return whether a numeric attribute contains missing values
      virtual unsigned int getNoValues(const CategoricalAttribute att) const {return 0;};   ///< return the number of values for a categorical attribute
      virtual const char* getCatAttName(const CategoricalAttribute att) const {return NULL;};  ///< return the name for a categorical Attribute
      virtual const char* getCatAttValName(const CategoricalAttribute att, const CatValue val) const {return NULL;}; ///< return the name for a categorical attribute value
      virtual unsigned int getNoNumAtts() const {return noNumAtts_;};                          ///< return the number of numeric attributes
      virtual const char* getNumAttName(const NumericAttribute att) const {return NULL;};      ///< return the name for a numeric attribute
      virtual unsigned int getPrecision(const NumericAttribute att) const {return 5;};      ///< return the precision to which values of a numeric attribute should be output
      virtual const char* getName() const {return "stacked instances";};                                      ///< return a string that gives a meaningful name for the stream
      virtual bool areNamesCaseSensitive() const {return false;};                               ///< true iff name comparisons are case sensitive
      virtual void setAllAttsMissing() {}; 

      inline void setNoClasses(unsigned int n) {no_classes_=n;};
      inline void setNoNumAtts(unsigned int n) {noNumAtts_=n;};

    protected:
      unsigned int no_classes_;
      unsigned int noNumAtts_;
    };

    MetaDataFilter mfMetaData_;
  };

private:
  std::vector<IncrementalLearner*> base_learners_;            ///< the classifiers in the ensemble
  IncrementalLearner *meta_learner_;                          ///< the meta learner
  instance meta_inst_;                             ///< theinstance to be fed to the meta learner
  slInstanceStream base_outcomes_;                 ///< a stream formed from the outputs of the base learners
  capabilities capabilities_;
};

