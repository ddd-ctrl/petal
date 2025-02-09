/* Open source system for classification learning from very large data
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

#include "instanceFile.h"
#include "FilterSet.h"
#include "learner.h"


class StreamTestArgs {
public:
  StreamTestArgs() : calcAUPRC_(false), zoLoss_(true), smoothing_(10), freq_(1000), shuffledRepetitions_(0), unShuffledRepetitions_(0), plotfile_(NULL) {}

  void getArgs(char*const*& argv, char*const* end);  // get settings from command line arguments

  bool calcAUPRC_;
  bool zoLoss_;                       // true iff should plot 0-1 loss
  unsigned int smoothing_;            // the number of turns over which each output value should be average
  unsigned int freq_;                 // the frequency with which points should be output
  unsigned int shuffledRepetitions_;  // the number of times that the data should be shuffled and the test repeated - 0 means no shuffling
  unsigned int unShuffledRepetitions_;  // the number of times that the data should be shuffled and the test repeated - 0 means no shuffling
  FILE *plotfile_;
};

/// train a learner from a stream, classifying each instance before learning from it
/// @param theLearner the learner to test
/// @param instStream the instance stream to train from
/// @param instFile the underlying instance file from which the stream is fed.
/// @param filters the set of filters to apply before learning
void streamTest(learner *theLearner, InstanceStream &instStream, FilterSet &filters, const StreamTestArgs &args);
