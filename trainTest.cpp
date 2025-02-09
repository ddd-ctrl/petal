/* Petal: An open source system for classification learning from very large data
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


#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
//#ifndef DBG_NEW
//#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
//#define new DBG_NEW
//#endif
#include <stdlib.h>
#include <crtdbg.h>
#endif
#endif

#include "trainTest.h"
#include "instanceFile.h"
#include "utils.h"
#include "globals.h"
#include "instanceStreamDiscretiser.h"
#include "correlationMeasures.h"

#include <typeinfo>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#include <cmath>
#else
#include <float.h>
#endif

void TrainTestArgs::getArgs(char*const*& argv, char*const* end) {
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (streq(argv[0]+1, "auprc")) {
      calcAUPRC_ = true;
    }
    else if (argv[0][1] == 's') {
      scoreFile_ = argv[0]+2;
    }
    else break;

    ++argv;
  }
}


void trainTest(learner *theLearner, InstanceStream &sourceInstanceStream, InstanceStream &testInstanceStream, FilterSet &filters, const TrainTestArgs &args) {
  FILE* scoref = NULL;
  InstanceStream* instanceStream = filters.apply(&sourceInstanceStream);
  InstanceStream* testStream = filters.apply(&testInstanceStream);

  const unsigned int noClasses = instanceStream->getNoClasses();

  crosstab<InstanceCount> xtab(noClasses);
  
  long int trainTime = 0;
  long int testTime = 0;
  #ifdef __linux__
  struct rusage usage;
  #endif

  #ifdef __linux__
  getrusage(RUSAGE_SELF, &usage);
  trainTime = usage.ru_utime.tv_sec+usage.ru_stime.tv_sec;
  #endif


  if (verbosity >= 1) printf("Training from %s\n", instanceStream->getName());

  theLearner->train(*instanceStream);
  
  #ifdef __linux__
  getrusage(RUSAGE_SELF, &usage);
  trainTime = ((usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)-trainTime);
  #endif

  testStream->rewind();

  instance inst(*testStream);

  if (verbosity >= 1) printf("Testing against file %s\n", testStream->getName());
  
  if (args.scoreFile_ != NULL) {
	if (verbosity > 1) printf("Opening score file %s\n", args.scoreFile_);

    scoref = fopen(args.scoreFile_, "w");

    if (scoref == NULL) {
      error("Cannot open score file %s", args.scoreFile_);
    }
  }

  std::vector<double> classDist(noClasses);
  InstanceCount count = 0;
  unsigned int zeroOneLoss = 0;
  double squaredError = 0.0;
  double squaredErrorAll = 0.0;
  double logLoss = 0.0;
  std::vector<std::vector<float> > probs(testStream->getNoClasses()); //< the sequence of predicted probabilitys for each class
  std::vector<CatValue> trueClasses; //< the sequence of true classes
    
  #ifdef __linux__
  getrusage(RUSAGE_SELF, &usage);
  testTime= usage.ru_utime.tv_sec+usage.ru_stime.tv_sec;
  #endif

  //std::string out, lab;
  while (!testStream->isAtEnd()) {
    if (testStream->advance(inst)) {
      count++;

      theLearner->classify(inst, classDist);

      const CatValue prediction = indexOfMaxVal(classDist);
      const CatValue trueClass = inst.getClass();

      if (prediction != trueClass) zeroOneLoss++;

      const double error = 1.0-classDist[trueClass];
      squaredError += error * error;
      squaredErrorAll += error * error;
      logLoss += log2(classDist[trueClass]);
      for (CatValue y = 0; y < testStream->getNoClasses(); y++) {
        if (y != trueClass) {
          const double err = classDist[y];
          squaredErrorAll += err * err;
        }

        if (scoref != NULL) {
          // output the scores
          if (y != 0) {
            fputs(", ", scoref);
          }
          fprintf(scoref, "%f", classDist[y]);
        }
      }

      if (scoref != NULL) {
        fputc('\n', scoref);
      }

      xtab[trueClass][prediction]++;

      if (args.calcAUPRC_) {
        for (CatValue y = 0; y < testStream->getNoClasses(); y++) {
          probs[y].push_back(classDist[y]);
        }
        trueClasses.push_back(trueClass);
      }
    }
  }

  #ifdef __linux__
  getrusage(RUSAGE_SELF, &usage);
  testTime = ((usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)-testTime);
  #endif
    
  if (verbosity >= 1) {
    theLearner->printClassifier();
    printResults(xtab, *testStream);
      
    double MCC = calcMCC(xtab);
    printf("\nMCC:\n");
    printf("%0.4f\n", MCC);
  }

  if (args.calcAUPRC_) {
    calcAUPRC(probs, trueClasses, *testStream->getMetaData());
    //calcAUPRC_COFFIN(probs, trueClasses, *testStream->getMetaData());
    //out.erase(out.end()-1);
    //lab.erase(lab.end()-1);
    //printf("out=array([%s])\n",out.c_str());
    //printf("lab=array([%s])\n",lab.c_str());
  }

  printf("\n%" ICFMT " test cases\n0-1 loss = %0.6f\nRoot mean squared error = %0.3f\n"
          "Root mean squared error all classes = %0.3f\nLogarithmic loss = %0.3f\n"
          "Training time: %ld\nClassification time: %ld\n", 
          count, zeroOneLoss/static_cast<double>(count), sqrt(squaredError/count), 
          sqrt(squaredErrorAll/(count*testStream->getNoClasses())), -logLoss/count,
          trainTime, testTime);
}
