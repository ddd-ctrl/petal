#include <algorithm>

#include "instanceStreamDynamicDiscretiser.h"
#include "globals.h"
#include "mtrand.h"
#include "utils.h"
#include <algorithm>    // std::binary_search
#include <cmath> 

InstanceStreamDynamicDiscretiser::InstanceStreamDynamicDiscretiser(char* const *& argv, char* const * end)
  : sample_(NULL), noOfBins_(5), preUpdate_(true), degenerates_(true) {

	unsigned int intervalSampleSize = 0;
  unsigned int sampleSize = 1000;
  bool alwaysUpdate = false;
  bool replaceAtRandom = true; 

  // get arguments
	while (argv != end) {
		if (**argv == '+' && argv[0][1] == 'b') {
			getUIntFromStr(argv[0] + 2, noOfBins_, "b");

      if (noOfBins_ < 2) error("There must be at least 2 bins");

			++argv;
		}
		else if (streq(*argv, "+drift")) {
			alwaysUpdate = true;
      replaceAtRandom = false;
			++argv;
		}
		else if (**argv == '+' && argv[0][1] == 'i') {
			getUIntFromStr(argv[0] + 2, intervalSampleSize, "i");
			++argv;
		}
    else if (streq(*argv, "+nodegenerates")) {
			degenerates_ = false;
			++argv;
		}
		else if (streq(*argv, "+postupdate")) {
			preUpdate_ = false;
			++argv;
		}
		else if (streq(*argv, "+slowdrift")) {
			alwaysUpdate = true;
      replaceAtRandom = true;
			++argv;
		}
		else if (**argv == '+' && argv[0][1] == 's') {
			getUIntFromStr(argv[0] + 2, sampleSize, "s");
			++argv;
		}
    else {
			break;  // do not consume the remaining arguments
		}
	}

  if (intervalSampleSize) {
    sampleSize = intervalSampleSize * noOfBins_;
  }

  metaData_ = new InstanceStreamDynamicDiscretiser::MetaData(noOfBins_);
  theDiscretiser = NULL;

  if (sampleSize == 0) {
    // a complete sample
    sample_ = new completeSample(noOfBins_, degenerates_);
  }
  else if (replaceAtRandom) {
    // random sample
    sample_ = new randomSample(noOfBins_, degenerates_, sampleSize, alwaysUpdate);
  }
  else {
    // window sample
    sample_ = new windowSample(noOfBins_, degenerates_, sampleSize);
  }
}


InstanceStreamDynamicDiscretiser::~InstanceStreamDynamicDiscretiser(void) {
  delete sample_;
}

/// set the source for the filter
void InstanceStreamDynamicDiscretiser::setSource(InstanceStream &src) {
  source_ = &src;
  metaData_->setSource(src.getMetaData());

  //if (allNumWithMiss_) metaData_->setAllAttsMissing();

  InstanceStream::metaData_ = metaData_;

  sourceInst_.init(src);

  reset();
}

void InstanceStreamDynamicDiscretiser::reset() {
  metaData_->cuts.clear();
  metaData_->cuts.resize(source_->getNoNumAtts());  // initially all atts are discretised into a single interval

  sample_->reset(source_->getNoNumAtts());
}

void InstanceStreamDynamicDiscretiser::rewind() {
  InstanceStreamDiscretiser::rewind();
  reset();
}

static MTRand_int32 randint;

void InstanceStreamDynamicDiscretiser::update(instance &inst) {
  sample_->insert(inst);
}

/// advance to the next instance in the stream. Return true iff successful. @param inst the instance record to receive the new instance. 
bool InstanceStreamDynamicDiscretiser::advance(instance &inst) {
	if (!source_->advance(sourceInst_))
		return false;

  setClass(inst, sourceInst_.getClass());

  if (preUpdate_) update(sourceInst_);

	CategoricalAttribute ca;

	for (ca = 0; ca < source_->getNoCatAtts(); ca++) {
		setCatVal(inst, ca, sourceInst_.getCatVal(ca));
	}

	for (NumericAttribute na = 0; na < source_->getNoNumAtts(); na++) {
		setCatVal(inst, ca, discretise(sourceInst_.getNumVal(na), na));
		ca++;
	}

  if (!preUpdate_) update(sourceInst_);

	return true;
}

CatValue InstanceStreamDynamicDiscretiser::discretise(const NumValue val, const NumericAttribute na) const {
  if (source_->hasNumMissing(na) && val == MISSINGNUM) return noOfBins_;  // return a missing value
  else return sample_->discretise(na, val);
}

/// return the number of values for a categorical attribute
unsigned int InstanceStreamDynamicDiscretiser::MetaData::getNoValues(CategoricalAttribute att) const {
	if (att < source_->getNoCatAtts())
		return source_->getNoValues(att);
	else {
		return noOfBins_ + 1; // also allow for the missing value
	}
}

// all discretised attributes have missing vlaues because the value is set to missing before the window has grown large enough to set values
bool InstanceStreamDynamicDiscretiser::MetaData::hasCatMissing(const CategoricalAttribute att) const {
	if (att < source_->getNoCatAtts())
	  return source_->hasCatMissing(att);
  else
    return true;
}

void InstanceStreamDynamicDiscretiser::sortedIndexesMultiset::reset(const unsigned int nbins, const bool degenerates) {
  nBins_ = nbins;
  degenerates_ = degenerates;

  values_.resize(nbins);

  for (unsigned int i = 0; i < nBins_; i++) {
    values_[i].clear();
  }

  size_ = 0;
}

void InstanceStreamDynamicDiscretiser::sortedIndexesMultiset::insert(const NumValue v) {
  const unsigned int targetbin = size_ % nBins_;  ///< the bin needing to expand
  unsigned int loc = 0;                           ///< the bin into which this value goes

  // advance while v can't go into this bin
  while (loc < nBins_-1 && !values_[loc+1].empty() && v > *values_[loc+1].begin()) {
    loc++;
  }

  // no bin before targetbin can be empty
  while (loc < targetbin && v >= *values_[loc].rbegin()) {
    // v falls between intervals so insert into the one closer to the target
    loc++;
  }

  const unsigned int insertLoc = loc;

  if (targetbin >= loc) {
    // need to shuffle replaced value up
    while (loc < targetbin) {
      std::multiset<NumValue>::iterator valToMove = values_[loc].end();
      valToMove--;
      values_[loc+1].insert(*valToMove);
      values_[loc].erase(valToMove);
      loc++;
    }
  }
  else {
    // need to shuffle replaced value down
    while (loc > targetbin) {
      std::multiset<NumValue>::iterator valToMove = values_[loc].begin();
      values_[loc-1].insert(*valToMove);
      values_[loc].erase(valToMove);
      loc--;
    }
  }

  values_[insertLoc].insert(v);

  size_++;
}

void InstanceStreamDynamicDiscretiser::sortedIndexesMultiset::remove(const NumValue v) {
  const unsigned int targetbin = (size_-1) % nBins_;  ///< the bin needing to contract
  unsigned int loc = 0;                               ///< the bin containing this value

  // advance until the bin contains the value
  while (loc < nBins_-1 && !values_[loc+1].empty() && v >= *values_[loc+1].begin()) {
    loc++;
  }

  // need to use an iterator to ensure that only one instance of the value is removed
  std::multiset<NumValue>::iterator valToErase = values_[loc].find(v);

  if (valToErase == values_[loc].end()) error("Attempt to remove a value failed");

  values_[loc].erase(valToErase);

  if (targetbin < loc) {
    // need to shuffle replacement value up
    while (loc > targetbin) {
      std::multiset<NumValue>::iterator valToMove = values_[loc-1].end();
      valToMove--;
      values_[loc].insert(*valToMove);
      loc--;
      values_[loc].erase(valToMove);
    }
  }
  else {
    // need to shuffle replacement value down
    while (loc < targetbin) {
      std::multiset<NumValue>::iterator valToMove = values_[loc+1].begin();
      values_[loc].insert(*valToMove);
      loc++;
      values_[loc].erase(valToMove);
    }
  }

  size_--;
}

void InstanceStreamDynamicDiscretiser::sortedIndexesMultiset::replace(const NumValue oldV, const NumValue newV) {
  if (oldV == newV) return;

  unsigned int oldBin = 0;  ///< the bin containing the old value
  unsigned int newBin = 0;  ///< the bin to contain new value

  // advance until the bin contains the value
  while (oldBin < nBins_-1 && !values_[oldBin+1].empty() && oldV >= *values_[oldBin+1].begin()) {
    oldBin++;
  }

  // need to use an iterator to ensure that only one instance of the value is removed
  std::multiset<NumValue>::iterator valToErase = values_[oldBin].find(oldV);
  assert(valToErase != values_[oldBin].end());
  if (valToErase == values_[oldBin].end()) error("Attempt to replace a value failed");
  values_[oldBin].erase(valToErase);

  // advance while v can't go into this bin
  while (newBin < nBins_-1 && !values_[newBin+1].empty() && newV > *values_[newBin+1].begin()) {
    newBin++;
  }

  while (newBin < oldBin && newV >= *values_[newBin].rbegin()) {
    // v falls between intervals so insert into the one closer to the target
    newBin++;
  }

  unsigned int loc = newBin;

  if (oldBin >= newBin) {
    // need to shuffle replaced value up
    while (loc < oldBin) {
      std::multiset<NumValue>::iterator valToMove = values_[loc].end();
      valToMove--;
      values_[loc+1].insert(*valToMove);
      values_[loc].erase(valToMove);
      loc++;
    }
  }
  else {
    // need to shuffle replaced value down
    while (loc > oldBin) {
      std::multiset<NumValue>::iterator valToMove = values_[loc].begin();
      values_[loc-1].insert(*valToMove);
      values_[loc].erase(valToMove);
      loc--;
    }
  }

  values_[newBin].insert(newV);
}

/// return a cut value
NumValue InstanceStreamDynamicDiscretiser::sortedIndexesMultiset::cut(const unsigned int index) const {
  if (values_[index].empty()) return MISSINGNUM;
  return *values_[index].rbegin();
}

// return the discretised value
inline CatValue InstanceStreamDynamicDiscretiser::sortedIndexesMultiset::discretise(const NumValue v) const {
  CatValue cv = 0;

  while (cv < nBins_ && !values_[cv].empty() && v > *values_[cv].rbegin()) cv++;

  // if the value spans the entire next bin then return the next bin
  if (degenerates_ && cv < nBins_-1 && !values_[cv+1].empty() && v == maxVal(cv+1)) cv++;

  return cv;
}


///////////////////////////////////
// Methods for sortedIndexesDEPQ //
///////////////////////////////////
void InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::reset(const unsigned int nbins, const bool degenerates) {
  nBins_ = nbins;
  degenerates_ = degenerates;

  values_.resize(nbins);

  for (unsigned int i = 0; i < nBins_; i++) {
    values_[i].clear();
  }

  size_ = 0;
}

void InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::insert(const NumValue v) {
  // size_ % nBins_ is the bin needing to expand
  insert(v, size_ % nBins_);

  size_++;
}

void InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::insert(const NumValue v, unsigned int targetbin) {
  unsigned int loc = 0;                     ///< the bin into which this value goes

  // advance while v can't go into this bin
  while (loc < nBins_-1 && !values_[loc+1].empty() && v > values_[loc+1].min()) {
    loc++;
  }

  // check  whether v falls between intervals and if so insert into the one closer to the target
  while (loc < targetbin && v >= values_[loc].max()) {
    loc++;
  }

  const unsigned int insertLoc = loc;

  if (targetbin >= loc) {
    // need to shuffle replaced value up
    while (loc < targetbin) {
      NumValue valToMove = values_[targetbin-1].pop_max();
      values_[targetbin].push(valToMove);
      targetbin--;
    }
  }
  else {
    // need to shuffle replaced value down
    while (loc > targetbin) {
      NumValue valToMove = values_[targetbin+1].pop_min();
      values_[targetbin].push(valToMove);
      targetbin++;
    }
  }

  values_[insertLoc].push(v);
}

void InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::replace(unsigned int targetbin, const unsigned int index, NumValue v) {
  unsigned int loc = 0;                     ///< the bin into which this value goes

  // advance while v can't go into this bin
  while (loc < nBins_-1 && !values_[loc+1].empty() && v > values_[loc+1].min()) {
    loc++;
  }

  // check  whether v falls between intervals and if so insert into the one closer to the target
  while (loc < targetbin && v >= values_[loc].max()) {
    loc++;
  }

  if (targetbin > loc) {
    // need to shuffle replaced value up
    while (loc < targetbin) {
      const NumValue valToMove = values_[loc].max();
      values_[loc].replaceMax(v);
      loc++;
      v = valToMove;
    }
  }
  else if (targetbin < loc) {
    // need to shuffle replaced value down
    while (loc > targetbin) {
      const NumValue valToMove = values_[loc].min();
      values_[loc].replaceMin(v);
      loc--;
      v = valToMove;
    }
  }

  values_[targetbin].replace(index, v);
}

void InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::replace(unsigned int index, const NumValue v) {
  unsigned int replacementBin = 0;

  while (index >= values_[replacementBin].size()) {
    index -= values_[replacementBin].size();
    replacementBin++;
  }

  //values_[replacementBin].remove(index);

  //insert(v, replacementBin);
  replace(replacementBin, index, v);
}


/// return a cut value
NumValue InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::cut(const unsigned int index) const {
  if (values_[index].empty()) return MISSINGNUM;
  return values_[index].max();
}

// return the discretised value
inline CatValue InstanceStreamDynamicDiscretiser::sortedIndexesDEPQ::discretise(const NumValue v) const {
  CatValue cv = 0;

  while (cv < nBins_ && !values_[cv].empty() && v > values_[cv].max()) {
    cv++;
  }

  // if the value spans the entire next bin then return the next bin
  if (degenerates_ && cv < nBins_-1 && !values_[cv+1].empty() && v == values_[cv+1].max()) {
    cv++;
  }

  return cv;
}


////////////////////
/// windowSample ///
////////////////////


void InstanceStreamDynamicDiscretiser::windowSample::reset(const unsigned int noNumAtts) {
  sortedValues_.resize(noNumAtts);
  insertionOrder_.resize(noNumAtts);
  next_.resize(noNumAtts);

  for (NumericAttribute a = 0; a < noNumAtts; a++) {
    sortedValues_[a].reset(noOfBins_, degenerates_);
    insertionOrder_[a].clear();
    next_[a] = 0;
  }
}

void InstanceStreamDynamicDiscretiser::windowSample::insert(const instance &inst) {
  for (NumericAttribute a = 0; a < sortedValues_.size(); a++) {
    if (!inst.isMissing(a)) {
      const NumValue v = inst.getNumVal(a);

      if (sortedValues_[a].size() < sampleSize_) {
        // the sample is not full so need to add the value to the queue and the sample
        sortedValues_[a].insert(v);
        insertionOrder_[a].push_back(v);
      }
      else {
        // the sample is full so need to replace the oldest value with this one
        const NumValue r = insertionOrder_[a][next_[a]];

        if (r == v) {
          // if replacing the same value no need to do anything except advance the queue
          next_[a]++;
          if (next_[a] >= insertionOrder_[a].size()) next_[a] = 0;
        }
        else {
          insertionOrder_[a][next_[a]] = v;
          next_[a]++;
          if (next_[a] >= insertionOrder_[a].size()) next_[a] = 0;

          sortedValues_[a].replace(r, v);

          assert(sortedValues_[a].size() == sampleSize_);
        }
      }
    }
  }
}

NumValue InstanceStreamDynamicDiscretiser::windowSample::cut(const NumericAttribute att, const unsigned int index) const {
  return sortedValues_[att].cut(index);
}

CatValue InstanceStreamDynamicDiscretiser::windowSample::discretise(const NumericAttribute att, const NumValue v) const {
  return sortedValues_[att].discretise(v);
}

///////////////////////////////////
// Methods for randomSample //
///////////////////////////////////

void InstanceStreamDynamicDiscretiser::randomSample::reset(const unsigned int noNumAtts) {
  if (alwaysUpdate_) {
    // count is not used
    count_.clear();
  }
  else {
    count_.assign(noNumAtts, 0);
  }
  
  sortedValues_.resize(noNumAtts);

  for (NumericAttribute a = 0; a < noNumAtts; a++) {
    sortedValues_[a].reset(noOfBins_, degenerates_);
  }
}

void InstanceStreamDynamicDiscretiser::randomSample::insert(const instance &inst) {
  // insert the values into each sortedValues_
  for (NumericAttribute na = 0; na < sortedValues_.size(); na++) {
    if (!inst.isMissing(na)) {
      const NumValue v = inst.getNumVal(na);

      if (sortedValues_[na].size() < sampleSize_) {
        sortedValues_[na].insert(v);

        if (!alwaysUpdate_) {
          count_[na]++;
        }
      }
      else {
        if (alwaysUpdate_) {
          unsigned int randval = randint(sampleSize_);  // this is the index of the value that each sample should replace
          sortedValues_[na].replace(randval, v);
        }
        else {
          static MTRand_open randopen;

          if (randopen() <= static_cast<double>(sampleSize_)/count_[na]) {
            unsigned int randval = randint(sampleSize_);  // this is the index of the value that each sample should replace
            sortedValues_[na].replace(randval, v);
          }

          count_[na]++;
        }
      }
    }
  }
}

NumValue InstanceStreamDynamicDiscretiser::randomSample::cut(const NumericAttribute att, const unsigned int index) const {
  return sortedValues_[att].cut(index);
}

CatValue InstanceStreamDynamicDiscretiser::randomSample::discretise(const NumericAttribute att, const NumValue v) const {
  return sortedValues_[att].discretise(v);
}


///////////////////////////////////
// Methods for completeSample //
///////////////////////////////////

void InstanceStreamDynamicDiscretiser::completeSample::reset(const unsigned int noNumAtts) {
  sortedValues_.resize(noNumAtts);

  for (NumericAttribute a = 0; a < noNumAtts; a++) {
    sortedValues_[a].reset(noOfBins_, degenerates_);
  }
}

void InstanceStreamDynamicDiscretiser::completeSample::insert(const instance &inst) {
  // insert the values into each sortedValues_
  for (NumericAttribute na = 0; na < sortedValues_.size(); na++) {
    if (!inst.isMissing(na)) {
      const NumValue v = inst.getNumVal(na);
      sortedValues_[na].insert(v);
    }
  }
}

NumValue InstanceStreamDynamicDiscretiser::completeSample::cut(const NumericAttribute att, const unsigned int index) const {
  return sortedValues_[att].cut(index);
}

CatValue InstanceStreamDynamicDiscretiser::completeSample::discretise(const NumericAttribute att, const NumValue v) const {
  return sortedValues_[att].discretise(v);
}
