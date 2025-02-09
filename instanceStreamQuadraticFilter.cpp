#include "instanceStreamQuadraticFilter.h"
#include "globals.h"
#include "mtrand.h"
#include "utils.h"

// discretisers
#include "eqDepthDiscretiser.h"
#include "MDLDiscretiser.h"
#include "MDLBinaryDiscretiser.h"

//RegisterFilter* r = new RegisterFilter("quadratic", InstanceStreamQuadraticFilter::InstanceStreamQuadraticFilter);

InstanceStreamQuadraticFilter::InstanceStreamQuadraticFilter(char*const*& argv, char*const* end)
{
  // get arguments
  // no arguments currently supported
  //while (argv != end) {
  //  if (**argv == '-' && argv[0][1] == 's') {
  //    getUIntFromStr(argv[0]+2, targetSampleSize_, "s");
  //    ++argv;
  //  }
  //  else {
  //    break;  // do not consume the remaining arguments
  //  }
  //}

  InstanceStream::metaData_ = &metaData_;
}

InstanceStreamQuadraticFilter::InstanceStreamQuadraticFilter(InstanceStream *src)
{
  InstanceStream::metaData_ = &metaData_;
  setSource(*src);
}

InstanceStreamQuadraticFilter::~InstanceStreamQuadraticFilter(void) {
}

/// set the source for the filter
void InstanceStreamQuadraticFilter::setSource(InstanceStream &src) {
	source_ = &src;
  metaData_.setSource(src.getMetaData());
  metaData_.precision_.clear();
  metaData_.no_values_.clear();
  metaData_.hasNumMissing_.clear();
  metaData_.hasCatMissing_.clear();

  noOrigNum_ = src.getNoNumAtts();

  noOrigVals_.clear();
  for (CategoricalAttribute i = 0; i < src.getNoCatAtts(); i++) {
    noOrigVals_.push_back(src.getNoValues(i));
  }

  // copy numeric atts
  for (NumericAttribute i = 0; i < noOrigNum_; i++) {
    metaData_.precision_.push_back(src.getPrecision(i));
    metaData_.hasNumMissing_.push_back(src.hasNumMissing(i));
  }

  // copy categorical atts
  for (CategoricalAttribute i = 0; i < src.getNoCatAtts(); i++) {
    metaData_.no_values_.push_back(src.getNoValues(i));
    metaData_.hasCatMissing_.push_back(src.hasCatMissing(i));
  }

  // create a quadratic numeric att for every pair of original numeric atts
  for (NumericAttribute a1 = 1; a1 < noOrigNum_; a1++) {
    for (NumericAttribute a2 = 0; a2 < a1; a2++) {
      metaData_.precision_.push_back(src.getPrecision(a1)+src.getPrecision(a2));
      metaData_.hasNumMissing_.push_back(src.hasNumMissing(a1)||src.hasNumMissing(a2));
    }
  }

  // create a quadratic numeric att for every pair of original cat att value and numeric att
  for (CategoricalAttribute a1 = 0; a1 < src.getNoCatAtts(); a1++) {
    for (NumericAttribute a2 = 0; a2 < noOrigNum_; a2++) {
      for (CatValue v = 0; v < src.getNoValues(a1); v++) {
        metaData_.precision_.push_back(src.getPrecision(a2));
        metaData_.hasNumMissing_.push_back(src.hasCatMissing(a1)||src.hasNumMissing(a2));
      }
    }
  }

  // create a quadratic categorical att for every pair of original cat atts
  for (CategoricalAttribute a1 = 1; a1 < src.getNoCatAtts(); a1++) {
    for (CategoricalAttribute a2 = 0; a2 < a1; a2++) {
      metaData_.no_values_.push_back(src.getNoValues(a1)*src.getNoValues(a2));
      metaData_.hasCatMissing_.push_back(src.hasCatMissing(a1)||src.hasCatMissing(a2));
    }
  }

	rewind();
}

/// return to the first instance in the stream
void InstanceStreamQuadraticFilter::rewind() {
	source_->rewind();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceStreamQuadraticFilter::advance() {
	return source_->advance();
}


void InstanceStreamQuadraticFilter::convert(const instance &in, instance &out) {
  out.setClass(in.getClass());

  // copy numeric atts
  for (NumericAttribute i = 0; i < noOrigNum_; i++) {
    out.setNumVal(i, in.getNumVal(i));
  }

  // copy categorical atts
  for (CategoricalAttribute i = 0; i < noOrigVals_.size(); i++) {
    out.setCatVal(i, in.getCatVal(i));
  }

  NumericAttribute ni = noOrigNum_;

  // create a quadratic numeric att for every pair of original numeric atts
  for (NumericAttribute a1 = 1; a1 < noOrigNum_; a1++) {
    for (NumericAttribute a2 = 0; a2 < a1; a2++) {
      if (in.isMissing(a1) || in.isMissing(a2)) {
        out.setMissing(ni++);
      }
      else {
        out.setNumVal(ni++, in.getNumVal(a1) * in.getNumVal(a2));
      }
    }
  }

  // create a quadratic numeric att for every pair of original cat att value and numeric att
  for (CategoricalAttribute a1 = 0; a1 < noOrigVals_.size(); a1++) {
    for (NumericAttribute a2 = 0; a2 < noOrigNum_; a2++) {
      CatValue v;
      for (v = 0; v < in.getCatVal(a1); v++) {
        out.setNumVal(ni++, 0); // values that are not present
      }
      out.setNumVal(ni++, in.getNumVal(a2)); // copy the numeric value to the value for the corresponding cat val
      for (; v < noOrigVals_[a1]; v++) {
        out.setNumVal(ni++, 0); // values that are not present
      }
    }
  }

  CategoricalAttribute ci = noOrigVals_.size(); // the start index into the generated categorical attributes of the quadratic categorical attributes

  // create a quadratic categorical att for every pair of original cat atts
  for (CategoricalAttribute a1 = 1; a1 < noOrigVals_.size(); a1++) {
    for (CategoricalAttribute a2 = 0; a2 < a1; a2++) {
      out.setCatVal(ci++, in.getCatVal(a1)*noOrigVals_[a2]+in.getCatVal(a2));
    }
  }
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamQuadraticFilter::advance(instance &inst) {
	if (!source_->advance(sourceInst_))
		return false;

	convert(sourceInst_, inst);

	return true;
}


/// true if we have advanced past the last instance
bool InstanceStreamQuadraticFilter::isAtEnd() const {
	return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.
InstanceCount InstanceStreamQuadraticFilter::size() {
	return source_->size();
}


InstanceStreamQuadraticFilter::QFMetaData::QFMetaData()  {
}

// return the number of categorical attributes
unsigned int InstanceStreamQuadraticFilter::QFMetaData::getNoCatAtts() const {
  return no_values_.size();
}

// return whether a categorical attribute contains missing values
bool InstanceStreamQuadraticFilter::QFMetaData::hasCatMissing(const CategoricalAttribute att) const {
  return hasCatMissing_[att];
}

// return whether a numeric attribute contains missing values
bool InstanceStreamQuadraticFilter::QFMetaData::hasNumMissing(const NumericAttribute att) const {
  return hasNumMissing_[att];
}

// return the number of values for a categorical attribute
unsigned int InstanceStreamQuadraticFilter::QFMetaData::getNoValues(const CategoricalAttribute att) const {
  return no_values_[att];
}

// return the name for a categorical Attribute
const char* InstanceStreamQuadraticFilter::QFMetaData::getCatAttName(const CategoricalAttribute att) const {
  return "Quadratic attribute";
}

// return the name for a categorical attribute value
const char* InstanceStreamQuadraticFilter::QFMetaData::getCatAttValName(const CategoricalAttribute att, const CatValue val) const {
  return "Quadratic attribute value";
}

// return the number of numeric attributes
unsigned int InstanceStreamQuadraticFilter::QFMetaData::getNoNumAtts() const {
  return precision_.size();
}

// return the name for a numeric attribute
const char* InstanceStreamQuadraticFilter::QFMetaData::getNumAttName(const NumericAttribute att) const {
  return "Quadratic attribute";
}

// return the precision to which values of a numeric attribute should be output
unsigned int InstanceStreamQuadraticFilter::QFMetaData::getPrecision(const NumericAttribute att) const {
  return precision_[att];
}

// return a string that gives a meaningful name for the stream
const char* InstanceStreamQuadraticFilter::QFMetaData::getName() const {
  return "Quadratic filter";
}
