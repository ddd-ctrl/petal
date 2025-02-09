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

#include "dataStatisticsAction.h"
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

void DataStatisticsActionArgs::getArgs(char*const*& argv, char*const* end) {
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (streq(argv[0]+1, "runs")) {
      numericRuns_ = true;
    }
    else break;

    ++argv;
  }
}

InstanceCount getMaxRun(std::vector<NumValue>& vals, NumValue &val) {
  std::sort(vals.begin(), vals.end());

  InstanceCount thisRun = 0;
  InstanceCount longestRun = 0;
  NumValue lastV = MISSINGNUM;
  for (std::vector<NumValue>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    if (*it == lastV) thisRun++;
    else {
      if (thisRun > longestRun) {
        longestRun = thisRun;
        val = lastV;
      }
      thisRun = 1;
      lastV = *it;
    }
  }

  return longestRun;
}

void dataStatisticsAction(InstanceStream &sourceInstanceStream, FilterSet &filters, const DataStatisticsActionArgs &args) {
  InstanceStream* instanceStream = filters.apply(&sourceInstanceStream);

  printf("Collecting statistics from %s\n", instanceStream->getName());

  instance inst(*instanceStream);
  InstanceCount noInstances = 0;
  unsigned int noAttValues = 0;
  std::vector<std::vector<NumValue> > vals(instanceStream->getNoNumAtts()); // record all of the values of each numeric attribute
  std::vector<InstanceCount> numMissing(instanceStream->getNoNumAtts(), 0);  // no of missing nmeric valules
  InstanceCount totalNumMissing = 0;
  std::vector<InstanceCount> catMissing(instanceStream->getNoCatAtts(), 0);  // no of missing categorical values
  InstanceCount totalCatMissing = 0;
  std::vector<InstanceCount> classCount(instanceStream->getNoClasses(), 0);
  for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++) {
    noAttValues += instanceStream->getNoValues(a);
  }

  while (!instanceStream->isAtEnd()) {
    instanceStream->advance(inst);

    classCount[inst.getClass()]++;
    
    for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
      if (inst.isMissing(a)) {
        numMissing[a]++;
        totalNumMissing++;
      }
      else {
        if (args.numericRuns_) {
          vals[a].push_back(inst.getNumVal(a));
        }
      }
    }

    for (CategoricalAttribute a = 0; a < instanceStream->getNoCatAtts(); a++) {
      if (instanceStream->hasCatMissing(a) && inst.getCatVal(a) == instanceStream->getNoValues(a)-1) {
        catMissing[a]++;
      }
    }

    noInstances++;
  }

  printf("\n\n--- Statistics for %s ---\n\n", instanceStream->getName());
  printf("# instances: %d\n", noInstances);
  printf("# attributes: %d\n", instanceStream->getNoNumAtts()+instanceStream->getNoCatAtts());
  printf("# categorical atts: %d\n", instanceStream->getNoCatAtts());
  printf("average # attribute values: %.1f\n", noAttValues/static_cast<double>(instanceStream->getNoCatAtts()));
  printf("average %% missing categorical values: %.1f\n", totalCatMissing?totalCatMissing/static_cast<double>(instanceStream->getNoCatAtts()*noInstances)*100:0.0);
  printf("# numeric atts: %d\n", instanceStream->getNoNumAtts());
  printf("average %% missing numeric values: %.1f\n", totalNumMissing?totalNumMissing/static_cast<double>(instanceStream->getNoNumAtts()*noInstances)*100:0.0);
  if (args.numericRuns_) {
    for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {
      NumValue v;
      const InstanceCount maxRun = getMaxRun(vals[a], v);
      printf("longest run for %s: value=%f, count=%" ICFMT " (%.3f%%)\n", instanceStream->getNumAttName(a), v, maxRun, maxRun/static_cast<double>(noInstances));
    }
  }
  printf("# classes: %d\n", instanceStream->getNoClasses());
  printf("class distribution:\n");
  const char *sep = "";
  for (std::vector<unsigned int>::const_iterator it = classCount.begin(); it != classCount.end(); it++) {
    printf("%s%0.4f", sep, static_cast<double>(*it)/noInstances);
    sep = ", ";
  }
  putchar('\n');
}
