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

#include "syntheticInstanceStream.h"
#include "utils.h"

SyntheticInstanceStream::SyntheticInstanceStream(char*const*& argv, char*const* argvEnd) : driftType_(incrementalDrift), distributionType_(normalDistribution), subclasses_(false), time_(0), seed_(0), noAtts_(10), p_(0.5) {

  duration_ = 100000;
  inflation_ = rand() / 10.0;

  bool minsep = false;

  while (argv != argvEnd) {
    if (*argv[0] != '+') {
      break;
    }
    else if (streq(argv[0], "+abrupt")) {
      driftType_ = abruptDrift;
      inflation_ = 0.0;
      ++argv;
    }
    else if (streq(argv[0], "+gradual")) {
      driftType_ = gradualDrift;
      inflation_ = 0.0;
      ++argv;
    }
    else if (streq(argv[0], "+minsep")) {
      minsep = true;
      ++argv;
    }
    else if (streq(argv[0], "+no-drift")) {
      driftType_ = noDrift;
      inflation_ = 0.0;
      ++argv;
    }
    else if (streq(argv[0], "+skewed")) {
      distributionType_ = skewedDistribution;
      ++argv;
    }
    else if (streq(argv[0], "+subclasses")) {
      subclasses_ = true;
      ++argv;
    }
    else if (argv[0][1] == 'a') {
      getUIntFromStr(argv[0] + 2, noAtts_, "a");
      ++argv;
    }
    else if (argv[0][1] == 'd') {
      getUIntFromStr(argv[0] + 2, duration_, "d");
      ++argv;
    }
    else if (argv[0][1] == 'p') {
      getFloatFromStr(argv[0] + 2, p_, "p");
      ++argv;
    }
    else if (argv[0][1] == 'r') {
      getFloatFromStr(argv[0] + 2, inflation_, "r");
      ++argv;
    }
    else if (argv[0][1] == 's') {
      getUIntFromStr(argv[0] + 2, seed_, "s");
      ++argv;
    }
    else break;
  }

  means_.resize(8);
  stddev_.resize(8);

  // set up 8 subclasses
  // even subclasses are subclasses of 0
  // if there is graudal drift then each subclass is gradually replaced its index + 4
  for (NumericAttribute a = 0; a < noAtts_; a++) {
    char name[10];

    sprintf(name, "a%d", a);
    metaData_.numAttNames_.push_back(name);
    if (minsep) {
      means_[0].push_back(0.0);
      means_[1].push_back(1.0);
      means_[2].push_back(0.0);
      means_[3].push_back(1.0);
      means_[4].push_back(0.0);
      means_[5].push_back(1.0);
      means_[6].push_back(0.0);
      means_[7].push_back(1.0);
      stddev_[0].push_back(2.0);
      stddev_[1].push_back(2.0);
      stddev_[2].push_back(2.0);
      stddev_[3].push_back(2.0);
      stddev_[4].push_back(2.0);
      stddev_[5].push_back(2.0);
      stddev_[6].push_back(2.0);
      stddev_[7].push_back(2.0);
    }
    else {
      means_[0].push_back(rand_());
      means_[1].push_back(rand_());
      means_[2].push_back(rand_());
      means_[3].push_back(rand_());
      means_[4].push_back(rand_());
      means_[5].push_back(rand_());
      means_[6].push_back(rand_());
      means_[7].push_back(rand_());
      stddev_[0].push_back(rand_());
      stddev_[1].push_back(rand_());
      stddev_[2].push_back(rand_());
      stddev_[3].push_back(rand_());
      stddev_[4].push_back(rand_());
      stddev_[5].push_back(rand_());
      stddev_[6].push_back(rand_());
      stddev_[7].push_back(rand_());
    }
  }

  metaData_.classAttName_ = "class";
  metaData_.classNames_.push_back("negative");
  metaData_.classNames_.push_back("positive");
  metaData_.name_ = "Synthetic Stream";

  InstanceStream::metaData_ = &metaData_;

  randNorm_.reset(seed_);
  rand_.seed(seed_);
}


SyntheticInstanceStream::~SyntheticInstanceStream(void) {
}


// return to the first instance in the stream
// note that rewind does not reset the random number generator so the stream will differ each time through
void SyntheticInstanceStream::rewind() {
  time_ = 0;
  //randNorm_.reset(seed_);
  //rand_.seed(seed_);
}

// advance, discarding the next instance in the stream.  Return true iff successful.
bool SyntheticInstanceStream::advance() {
  return true;
}

// advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
bool SyntheticInstanceStream::advance(instance &inst) {
  time_++;

  if (time_ > duration_) return false;

  CatValue y;

  if (rand_() < p_) {
    inst.setClass(1);
    y = 1;
  }
  else {
    inst.setClass(0);
    y = 0;
  }

  if (subclasses_ && rand_() < 0.5) {
    y += 2;
  }

  if (driftType_ == gradualDrift && rand_() < time_ / static_cast<double>(duration_)) {
    y += 4;
  }
  else if (driftType_ == abruptDrift && time_ > duration_ / 2) {
    y += 4;
  }

  for (NumericAttribute a = 0; a < noAtts_; a++) {
    if (distributionType_ == skewedDistribution && y%2 == 1) {
      inst.setNumVal(a, means_[y-1][a] + inflation_ * time_ / static_cast<double>(duration_));
    }
    else {
      const double thisMean = means_[y][a] + inflation_ * time_ / static_cast<double>(duration_);

      inst.setNumVal(a, randNorm_(thisMean, stddev_[y][a]));
    }
  }

  return true;
}

// true if we have advanced past the last instance
bool SyntheticInstanceStream::isAtEnd() const {
  return time_ >= duration_;
}

// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.
InstanceCount SyntheticInstanceStream::size() {
  return duration_;
}


unsigned int SyntheticInstanceStream::SyntheticMetaData::getNoClasses() const {
  return classNames_.size();
}

// return the number of classes
const char* SyntheticInstanceStream::SyntheticMetaData::getClassName(const CatValue y) const {
  return classNames_[y];
}

// return the name for a class
const char* SyntheticInstanceStream::SyntheticMetaData::getClassAttName() const {
  return classAttName_;
}

// return the name for the class attribute
unsigned int SyntheticInstanceStream::SyntheticMetaData::getNoCatAtts() const {
  return catAttNames_.size();
}

// return the number of categorical attributes
bool SyntheticInstanceStream::SyntheticMetaData::hasCatMissing(const CategoricalAttribute att) const {
  return false;
}

// return whether a categorical attribute contains missing values
bool SyntheticInstanceStream::SyntheticMetaData::hasNumMissing(const NumericAttribute att) const {
  return false;
}

// return whether a numeric attribute contains missing values
unsigned int SyntheticInstanceStream::SyntheticMetaData::getNoValues(const CategoricalAttribute att) const {
  return catAttValNames_[att].size();
}

// return the number of values for a categorical attribute
const char* SyntheticInstanceStream::SyntheticMetaData::getCatAttName(const CategoricalAttribute att) const {
  return catAttNames_[att];
}

// return the name for a categorical Attribute
const char* SyntheticInstanceStream::SyntheticMetaData::getCatAttValName(const CategoricalAttribute att, const CatValue val) const {
  return catAttValNames_[att][val];
}

// return the name for a categorical attribute value
unsigned int SyntheticInstanceStream::SyntheticMetaData::getNoNumAtts() const {
  return numAttNames_.size();
}

// return the number of numeric attributes
const char* SyntheticInstanceStream::SyntheticMetaData::getNumAttName(const NumericAttribute att) const {
  return numAttNames_[att];
}

// return the name for a numeric attribute
unsigned int SyntheticInstanceStream::SyntheticMetaData::getPrecision(const NumericAttribute att) const {
  return 0;
}

// return the precision to which values of a numeric attribute should be output
const char* SyntheticInstanceStream::SyntheticMetaData::getName() const {
  return name_;
}

// return a string that gives a meaningful name for the stream
bool SyntheticInstanceStream::SyntheticMetaData::areNamesCaseSensitive()const  {
  return false;
}

// true iff name comparisons are case sensitive
void SyntheticInstanceStream::SyntheticMetaData::setAllAttsMissing() {
}
