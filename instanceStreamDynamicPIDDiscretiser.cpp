#include <algorithm>

#include "instanceStreamDynamicPIDDiscretiser.h"
#include "globals.h"
#include "mtrand.h"
#include "utils.h"


InstanceStreamDynamicPIDDiscretiser::InstanceStreamDynamicPIDDiscretiser(char* const *& argv, char* const * end)
  : Nr_(0), noOfBins_(5), level1Multiplier_(20) {
  
  // get arguments
	while (argv != end) {
		if (**argv == '+' && argv[0][1] == 'b') {
			getUIntFromStr(argv[0] + 2, noOfBins_, "b");

      if (noOfBins_ < 2) error("There must be at least 2 bins");

			++argv;
		}
		else if (**argv == '+' && argv[0][1] == 'w') {
			getUIntFromStr(argv[0] + 2, level1Multiplier_, "w");
			++argv;
		}
    else {
			break;  // do not consume the remaining arguments
		}
	}

  noOfLevel1Bins_ = level1Multiplier_ * noOfBins_;
  alfa_ = 2.0 / noOfLevel1Bins_;  // this means that a level 1 bin must grow to approximately twice its default size before it is split

  metaData_ = new InstanceStreamDynamicPIDDiscretiser::MetaData(noOfBins_);
  theDiscretiser = NULL;
}


InstanceStreamDynamicPIDDiscretiser::~InstanceStreamDynamicPIDDiscretiser(void) {
}

/// set the source for the filter
void InstanceStreamDynamicPIDDiscretiser::setSource(InstanceStream &src) {
  source_ = &src;
  metaData_->setSource(src.getMetaData());

  if (allNumWithMiss_) metaData_->setAllAttsMissing();

  InstanceStream::metaData_ = metaData_;

  sourceInst_.init(src);

  reset();
}

void InstanceStreamDynamicPIDDiscretiser::rewind() {
  InstanceStreamDiscretiser::rewind();
  reset();
}

void InstanceStreamDynamicPIDDiscretiser::reset() {
  noOfLevel1Bins_ = level1Multiplier_ * noOfBins_;

  // initialise estMin_ and estMax_ to the true min and max
  estMin_.assign(source_->getNoNumAtts(), std::numeric_limits<NumValue>::max());
  estMax_.assign(source_->getNoNumAtts(), std::numeric_limits<NumValue>::min());

  instance inst(*source_);

  while (source_->advance(inst)) {
    for (NumericAttribute a = 0; a < source_->getNoNumAtts(); a++) {
      if (!inst.isMissing(a)) {
        const NumValue v = inst.getNumVal(a);

        if (v < estMin_[a]) estMin_[a] = v;
        if (v > estMax_[a]) estMax_[a] = v;
      }
    }
  }

  source_->rewind();

  min_ = estMin_;
  max_ = estMax_;

  // initialise level1Cuts_, level1Counts_ and cuts_
  level1Cuts_.resize(source_->getNoNumAtts());
  level1Counts_.resize(source_->getNoNumAtts());

  metaData_->cuts.clear();
  metaData_->cuts.resize(source_->getNoNumAtts());  // initially all atts are discretised into a single interval
  for (NumericAttribute a = 0; a < source_->getNoNumAtts(); a++) {
    level1Cuts_[a].resize(noOfLevel1Bins_+1);
    level1Cuts_[a][0] = estMin_[a];
    level1Cuts_[a][noOfLevel1Bins_] = estMax_[a];
    const NumValue range = estMax_[a]-estMin_[a];
    for (unsigned int c = 1; c < noOfLevel1Bins_; c++) {
      level1Cuts_[a][c] = estMin_[a] + c * range / noOfLevel1Bins_;
    }

    level1Counts_[a].assign(noOfLevel1Bins_+1, 0);  // 1 extra because we start the bins at level1Counts_[1]

    metaData_->cuts[a].resize(noOfBins_);
    for (unsigned int c = 0; c < noOfBins_-1; c++) {
      metaData_->cuts[a][c] = estMin_[a] + c * range / noOfBins_;
    }
  }

  Nr_ = 0;
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamDynamicPIDDiscretiser::advance(instance &inst) {
static MTRand_open rand;
static MTRand_int32 randint;

	if (!source_->advance(sourceInst_))	return false;

  // update counts and reorganise if necessary
  Nr_++;

  for (NumericAttribute na = 0; na < source_->getNoNumAtts(); na++) {
    if (!sourceInst_.isMissing(na)) {
      CatValue cv = 1;
      NumValue nv = sourceInst_.getNumVal(na);

      if (nv < level1Cuts_[na][0]) {
        // a new minimum value
        level1Cuts_[na][0] = min_[na] = nv;
        cv = 1;
      }
      else if (nv > level1Cuts_[na].back()) {
        // a new maximum value
        level1Cuts_[na].back() = max_[na] = nv;
        cv = level1Cuts_[na].size()-1;          // There is one more cut than bin because level1Cuts_[na][0] is before the first bin and level1Cuts_[na].back() is after the last
      }
      else {
        while (cv < level1Cuts_[na].size()-1 && nv > level1Cuts_[na][cv]) cv++;
      }

      level1Counts_[na][cv]++;

      if (level1Counts_[na][cv] > 1 && (level1Counts_[na][cv]+1)/(Nr_+2.0) > alfa_) {
        // need to split
        level1Counts_[na].push_back(0);
        level1Cuts_[na].push_back(0.0);

        // copy cuts upward leaving level1Cuts_[na] free
        for (CatValue i = level1Cuts_[na].size() - 1; i > cv; i--) {
          level1Counts_[na][i] = level1Counts_[na][i-1];
          level1Cuts_[na][i] = level1Cuts_[na][i-1];
        }

        // place a new cut half way between the two old cuts
        level1Cuts_[na][cv] = level1Cuts_[na][cv-1] + (level1Cuts_[na][cv+1] - level1Cuts_[na][cv-1]) / 2;
        
        // divide the counts between the two new bins
        level1Counts_[na][cv+1] = level1Counts_[na][cv] / 2;
        level1Counts_[na][cv] -= level1Counts_[na][cv+1];
      }

      if (Nr_ % noOfBins_ == 0) {
        // reset cuts
        const unsigned int targetSize = Nr_ / noOfBins_;
        CatValue level1Cut = 1;
        unsigned int count = level1Counts_[na][1];
        for (CatValue cut = 0; cut < noOfBins_; cut++) {
          while (count < (cut+1)*targetSize && level1Cut < level1Cuts_[na].size()-1) {
            level1Cut++;
            count += level1Counts_[na][level1Cut];
          }
          metaData_->cuts[na][cut] = level1Cuts_[na][level1Cut];
        }
      }
    }
  }

  setClass(inst, sourceInst_.getClass());

	CategoricalAttribute ca;

	for (ca = 0; ca < source_->getNoCatAtts(); ca++) {
		setCatVal(inst, ca, sourceInst_.getCatVal(ca));
	}

	for (NumericAttribute na = 0; na < source_->getNoNumAtts(); na++) {
    if (sourceInst_.isMissing(na)) {
      setCatVal(inst, ca, metaData_->getNoValues(ca)-1);
    }
    else {
		  setCatVal(inst, ca, discretise(sourceInst_.getNumVal(na), na));
    }
		ca++;
	}

	return true;
}

void InstanceStreamDynamicPIDDiscretiser::printStats() {
  printf("\nPID Discretiser: No of level 1 bins by attribute:\n");
  for (std::vector<std::vector<unsigned int> >::const_iterator it = level1Counts_.begin(); it != level1Counts_.end(); it++) {
    printf("- %ld\n", it->size());
  }
}

/// return the number of values for a categorical attribute
unsigned int InstanceStreamDynamicPIDDiscretiser::MetaData::getNoValues(CategoricalAttribute att) const {
	if (att < source_->getNoCatAtts())
		return source_->getNoValues(att);
	else {
		return noOfBins_ + 1; // also allow for the missing value
	}
}

// all discretised attributes have missing vlaues because the value is set to missing before the window has grown large enough to set values
bool InstanceStreamDynamicPIDDiscretiser::MetaData::hasCatMissing(const CategoricalAttribute att) const {
	if (att < source_->getNoCatAtts())
	  return source_->hasCatMissing(att);
  else
    return true;
}
