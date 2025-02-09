/*
 * extLearnVW.h
 *
 *      Author: ana
 */


#pragma once
#include "externLearner.h"
#include "instance.h"

#include "stdlib.h"
#include "utils.h"
#ifdef __linux__
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

/*
 * Call to extLearnVW (example, the name of the output vowpal wabbit file always at the end):
 * petal data.pmeta data.pdata -e vw --passes 3 --sgd --oaa k [--bfgs] vwFile.output
 */

class extLearnVW : public externLearner {
private:
  CatValue noClasses;
  FILE *testf;
  FILE *trainf;
  char*const* args;  // arguments to pass to executable
  int argcnt; // count of args
  std::string tmpTrain;
  std::string tmpTest;
  std::string tmpOutput;
  std::string dirTemp;
  
  std::vector<double> foldZOLoss;   ///< 0-1 loss from each fold
  std::vector<double> foldrmse;     ///< RMSE (petal) from each fold
  crosstab<InstanceCount> xtab;
  
  std::vector<double> expZOLoss;   ///< 0-1 loss from all experiments
  std::vector<double> exprmse;     ///< RMSE (petal) from all experiments
  std::vector<double> expMCC;     ///< RMSE (petal) from all experiments
  
  template <typename T> std::string toString(T tmp)
  {
    std::ostringstream out;
    out << tmp;
    return out.str();
  }

public:  
  extLearnVW(char *wd, char *bd, char*const argv[], int argc, InstanceStream *instanceStream) : externLearner(wd, bd), testf(NULL), trainf(NULL) {
          args = argv;
          argcnt = argc;
          tmpTrain = workingDir;
          tmpTest = workingDir;
          #ifdef __linux__
          unsigned int dirNumber = 1;
          dirTemp = tmpTrain+"/temp"+toString(dirNumber);
          int status = mkdir(dirTemp.c_str(),0777);
          while(status==-1){ //Different directory for different experiments
            dirNumber++;
            dirTemp =  tmpTrain+"/temp"+toString(dirNumber);
            status = mkdir(dirTemp.c_str(),0777);
          }
          tmpTrain += "/temp"+toString(dirNumber)+"/tmp.train.vw";
          tmpTest += "/temp"+toString(dirNumber)+"/tmp.test.vw";
          #endif

          noClasses = instanceStream->getNoClasses();
          xtab.init(noClasses);
          
//          std::string vwCommand = "";
//          vwCommand = "perl -e 'srand 0;' ";
//          printf("Executing command - %s\n", vwCommand.c_str());
//          system(vwCommand.c_str()); 
        }
	virtual ~extLearnVW() { fclose(testf); fclose(trainf); }

	void printTrainingData(InstanceStream *instanceStream);
	void printTest(InstanceStream *instanceStream);
	void printInst(FILE * f, instance &inst, InstanceStream *instanceStream);
	void printMeta(InstanceStream *instanceStream) const;
	void classify(InstanceStream *instanceStream, int fold);
        void printResults(InstanceStream *instanceStream, unsigned int exp);
        void calculateLossFunctions(unsigned int fold);
};

