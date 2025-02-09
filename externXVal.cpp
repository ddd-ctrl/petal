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
#include "externXVal.h"
#include "instanceStream.h"
#include "xValInstanceStream.h"
#include "utils.h"
#include "mtrand.h"
#include "globals.h"
#include "crosstab.h"

#include <assert.h>
#include <vector>
#include <string>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "externLearner.h"
#include "extLearnPetal.h"
#include "extLearnLibSVM.h"
#include "extLearnSVMSGD.h"
#include "extLearnOMCLPBoost.h"
#include "extLearnWeka.h"
#include "extLearnVW.h"


void externXVal(InstanceStream *instanceStream, FilterSet &filters, char *args, char*const argv[], int argc) {
  unsigned int noFolds = 10;
  unsigned int noExperiments = 1;
  std::vector<unsigned int*> vals;
  char *workingDir;
  char *binDir;

  if (argv[0][0] == '+' && argv[0][1] == 'd') {
    workingDir = argv[0]+2;
    argv++;
    argc--;
  }
  else workingDir = ".";

  if (argv[0][0] == '+' && argv[0][1] == 'b') {
    binDir = argv[0]+2;
    argv++;
    argc--;
  }
  else binDir = ".";

  //elPetal extL; // create the external learner
  externLearner *extL;

  vals.push_back(&noFolds);
  vals.push_back(&noExperiments);

  getUIntListFromStr(args, vals, "cross validation settings");

  if (argc == 0) {
    error("Must specify the external learner");
  }

  if (streq(argv[0], "petal")) {
    extL = new extLearnPetal(workingDir, binDir);
  }
  else if (streq(argv[0], "libsvm")) {
    extL = new extLearnLibSVM(workingDir, binDir, argv+1, argc-1);
  }
  else if (streq(argv[0], "SVMSGD")) {
    extL = new extLearnSVMSGD(workingDir, binDir, argv+1, argc-1);
  }
  else if (streq(argv[0], "OMCLPBoost")) {
    extL = new extLearnOMCLPBoost(workingDir, binDir, argv+1, argc-1);
  }
  else if (streq(argv[0], "weka")) {
    extL = new extLearnWeka(workingDir, binDir, argv+1, argc-1);
  }
  else if (streq(argv[0], "vw")) {
    extL = new extLearnVW(workingDir, binDir, argv+1, argc-1, instanceStream);
    noExperiments = 10;
  }
  else error("External learner %s not available", argv[0]);

  for (unsigned int exp = 0; exp < noExperiments; exp++) {
    
    XValInstanceStream xValStream(instanceStream, noFolds, exp);
    
    if (verbosity >= 1){
        printf("----------------------------------------------------------------\n");
    	printf("Cross validation experiment %d for %s\n", exp+1, instanceStream->getName());
    }
    
    for (unsigned int fold = 0; fold < noFolds; fold++) {
      if (verbosity >= 2) printf("Fold %d\n", fold);

      xValStream.startSubstream(fold, true);

      InstanceStream* filteredStream = filters.apply(&xValStream);

      // create the meta file
      extL->printMeta(filteredStream);

      extL->printTrainingData(filteredStream);

      if (verbosity >= 3) printf("Fold %d testing\n", fold);

      xValStream.startSubstream(fold, false);
      
      filteredStream->rewind();  // rewind the filtered stream to the start

      extL->classify(filteredStream, fold);
    }
    extL->printResults(filters.apply(&xValStream),exp);
  }
}
