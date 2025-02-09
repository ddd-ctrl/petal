#include "instanceStreamNormalisationFilter.h"
#include "globals.h"
#include "mtrand.h"
#include "utils.h"

// discretisers
#include "eqDepthDiscretiser.h"
#include "MDLDiscretiser.h"
#include "MDLBinaryDiscretiser.h"

//RegisterFilter* r = new RegisterFilter("quadratic", InstanceStreamNormalisationFilter::InstanceStreamNormalisationFilter);

InstanceStreamNormalisationFilter::InstanceStreamNormalisationFilter(char*const*& argv, char*const* end)
  : minIsMinusOne_(false)
{
  // get arguments
  while (argv != end) {
    if (**argv == '+' && argv[0][1] == 'm') {
      minIsMinusOne_ = true;
      ++argv;
    }
    else {
      break;  // do not consume the remaining arguments
    }
  }
}

InstanceStreamNormalisationFilter::InstanceStreamNormalisationFilter(InstanceStream *src)
{
  setSource(*src);
}

InstanceStreamNormalisationFilter::~InstanceStreamNormalisationFilter(void) {
}

/// set the source for the filter
void InstanceStreamNormalisationFilter::setSource(InstanceStream &src) {
	source_ = &src;
  metaData_ = src.getMetaData();


  // find min and max
  min_.assign(src.getNoNumAtts(), std::numeric_limits<NumValue>::max());
  max_.assign(src.getNoNumAtts(), std::numeric_limits<NumValue>::min());

	src.rewind();

  instance sourceInst_(src);

  while (src.advance(sourceInst_)) {
    for (NumericAttribute a = 0; a < metaData_->getNoNumAtts(); a++) {
      if (!sourceInst_.isMissing(a)) {
        const NumValue v = sourceInst_.getNumVal(a);
        if (v < min_[a]) min_[a] = v;
        if (v > max_[a]) max_[a] = v;
      }
    }
  }

	src.rewind();
}

/// return to the first instance in the stream
void InstanceStreamNormalisationFilter::rewind() {
	source_->rewind();
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool InstanceStreamNormalisationFilter::advance() {
	return source_->advance();
}


void InstanceStreamNormalisationFilter::convert(const instance &in, instance &out) {
  out.setClass(in.getClass());

  // convert numeric atts
  for (NumericAttribute a = 0; a < metaData_->getNoNumAtts(); a++) {
    if (in.isMissing(a)) {
      out.setNumVal(a, in.getNumVal(a));
    }
    else {
      out.setNumVal(a, normalise(in.getNumVal(a), a));
    }
  }

  // copy categorical atts
  for (CategoricalAttribute i = 0; i < metaData_->getNoCatAtts(); i++) {
    out.setCatVal(i, in.getCatVal(i));
  }
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamNormalisationFilter::advance(instance &inst) {
	if (!source_->advance(inst))
		return false;

  for (NumericAttribute a = 0; a < metaData_->getNoNumAtts(); a++) {
    if (!inst.isMissing(a)) {
      inst.setNumVal(a, normalise(inst.getNumVal(a), a));
    }
  }

	return true;
}


/// true if we have advanced past the last instance
bool InstanceStreamNormalisationFilter::isAtEnd() const {
	return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.
InstanceCount InstanceStreamNormalisationFilter::size() {
	return source_->size();
}
