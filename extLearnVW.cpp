/*
 * extLearnVW.cpp
 *
 *  Created on: 17/09/2012
 *      Author: nayyar and ana
 */

#include "extLearnVW.h"
#include "correlationMeasures.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>

void extLearnVW::printTrainingData(InstanceStream *instanceStream) {
	if (trainf == NULL) {
		printf("Creating %s\n",tmpTrain.c_str());
		trainf = fopen(tmpTrain.c_str(), "w");
		if (trainf == NULL) error("Cannot open output file %s", tmpTrain.c_str());
	}
        
        instance inst(*instanceStream);
        while (instanceStream->advance(inst)) {
          printInst(trainf, inst, instanceStream);
        }
}

void extLearnVW::printTest(InstanceStream *instanceStream) {
	if (testf == NULL) {
		printf("Creating %s\n", tmpTest.c_str());
		testf = fopen(tmpTest.c_str(), "w");
		if (testf == NULL) error("Cannot open output file %s", tmpTest.c_str());
	}
        instance inst(*instanceStream);
        while (instanceStream->advance(inst)) {
                printInst(testf, inst, instanceStream);
        }
}

void extLearnVW::printInst(FILE *f, instance &inst, InstanceStream *instanceStream) {
  
        //All classes must be in the range [1,k]
        fprintf(f, "%d |n", inst.getClass()+1);
        unsigned int numCatAtts = instanceStream->getNoCatAtts();

        unsigned int attIndex = 0;
	for (CategoricalAttribute a = 0; a < numCatAtts; a++){
          unsigned int noValues = instanceStream->getNoValues(a);
          if(noValues>2){
             attIndex+=inst.getCatVal(a);
             fprintf(f, " %d:%d", attIndex, 1); 
             attIndex+=noValues-inst.getCatVal(a);
          }else if(inst.getCatVal(a)!=0){
            fprintf(f, " %d:%d", attIndex, inst.getCatVal(a)); //Missing values have already been added as a new value
            attIndex++;
          }
        } 
        for (NumericAttribute a = 0; a < instanceStream->getNoNumAtts(); a++) {  
            if (!inst.isMissing(a) && inst.getNumVal(a)!=0) { //Assume 0 if a numeric value is missing.
                    fprintf(f, " %d:%0.*f", attIndex, instanceStream->getPrecision(a), inst.getNumVal(a));
            }
            attIndex++;
	}
        fputc('\n',f);
}

void extLearnVW::printMeta(InstanceStream *instanceStream) const{
      
}

void extLearnVW::classify(InstanceStream *instanceStream, int fold) {
        printTest(instanceStream);
	fclose(testf);
	fclose(trainf);
	testf = NULL;
	trainf = NULL;

        ///////////////////////////////////////////////////////////////////////
        //                         TRAINING
        ///////////////////////////////////////////////////////////////////////
        //int j = 0;
	//std::string vwCommand = "vw ";

//	if(argcnt!=0) {
//		wekaCommand += ' ';
//		wekaCommand += args[j];
//                j++;
//	}
        std::string vwCommand = "";
        vwCommand += "cat "; vwCommand += tmpTrain.c_str();
        vwCommand += "  | perl -MList::Util=shuffle -e 'srand "+toString(fold)+"; print shuffle(<STDIN>);' "; 
        vwCommand += "| vw";
	//vwCommand += "-d "; vwCommand += tmpTrain.c_str();
        vwCommand += " -f "; vwCommand += dirTemp+"/model.vw";
        vwCommand += " --cache_file "; vwCommand += dirTemp+"/cache_train"+toString(fold)+".vw";
        vwCommand += " --oaa "; vwCommand += toString(noClasses);
         
        int i;
        for (i = 0; i < argcnt-1; i++) {
                std::string argum = args[i];
                if(streq(argum.c_str(),"--bfgs")){
                  break;
                }
		vwCommand += ' ';
		vwCommand += args[i];
	}
        
        
        //vwCommand += "\n";
        //system("vw --version");
        printf("\n\n------ TRAINING FOLD %d ------\n",fold);
        printf("Executing command - %s\n", vwCommand.c_str());
        printf("1) Calling Vowpal Wabbit\n"); system(vwCommand.c_str()); 
        //printf("Finished training Vowpal Wabbit\n");

        std::string bfgs = args[i]; 
         std::string two = "";
        
        if(streq(bfgs.c_str(),"--bfgs")){
          // --bfgs
          //vwCommand += "cat "; vwCommand += tmpTrain.c_str();
          //vwCommand += "  | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' "; 
          vwCommand = "vw";
          //vwCommand += "-d "; vwCommand += tmpTrain.c_str();
          vwCommand += " -i "; vwCommand += dirTemp+"/model.vw";
          vwCommand += " -f "; vwCommand += dirTemp+"/model2.vw"; two = "2";
          vwCommand += " --cache_file "; vwCommand += dirTemp+"/cache_train"+toString(fold)+".vw";
          vwCommand += " --oaa "; vwCommand += toString(noClasses);
          vwCommand += " --bfgs ";
          i++;
          while (i < argcnt-1) {
		vwCommand += ' ';
		vwCommand += args[i];
                i++;
	  }

          printf("Executing command - %s\n", vwCommand.c_str());
          printf("2) Calling Vowpal Wabbit\n"); system(vwCommand.c_str()); 
        }
        printf("Finished training Vowpal Wabbit\n");
        
        ///////////////////////////////////////////////////////////////////////
        //                         TEST
        ///////////////////////////////////////////////////////////////////////
        
        tmpOutput = args[i];
                
        printf("\n------ TESTING FOLD %d ------\n",fold);
        vwCommand = "vw -i "; vwCommand += dirTemp+"/model"+two+".vw";
        vwCommand += " -t -d "; vwCommand += tmpTest.c_str();
        vwCommand += " --cache_file "; vwCommand += dirTemp+"/cache_test"+toString(fold)+".vw";
        vwCommand += " --oaa "; vwCommand += toString(noClasses);
        vwCommand += " -r "; vwCommand += dirTemp+"/raw.txt";
        //vwCommand += " -p "; vwCommand += tmpOutput+"_pred";
        

        printf("Executing command - %s\n", vwCommand.c_str());
        printf("Calling Vowpal Wabbit\n"); system(vwCommand.c_str()); 
        printf("Finished testing Vowpal Wabbit\n");

	//remove("tmp.meta");
	printf("Deleting files \n");
	remove(tmpTrain.c_str());
        std::string aux = dirTemp+"/model.vw";
        remove(aux.c_str());
	//remove(tmpTest.c_str()); //we'll need it to calculate the loss functions in printResults
	printf("Finished deleting files\n");
        
        calculateLossFunctions(fold);
        
}


inline void sigmoid(std::vector<double> &v){
  for (int i=0; i<v.size(); i++) {
         v[i] = 1/(1+exp(-v[i]));
  }
}

void extLearnVW::calculateLossFunctions(unsigned int fold){

   //Parse output file and combine loss functions
   
    std::ifstream vwOutput; 
    
    double squaredError = 0.0;
    double zeroOneLoss = 0;
    InstanceCount foldcount = 0;
    //tmpOutput += "raw";
    std::string raw = dirTemp+"/raw.txt";
    crosstab<InstanceCount> foldxtab(noClasses);

   std::ifstream tmpTestStream;
   vwOutput.open(raw.c_str());
    if(vwOutput){
      tmpTestStream.open(tmpTest.c_str());
      std::string linePred = "";
      std::string lineTest = "";
      getline(vwOutput,linePred);
      while(!vwOutput.eof()){       
        foldcount++;        
        std::vector<double> classDist; 
        size_t pos1, pos2 = 0;
        for(unsigned int c=0; c<noClasses; c++){
          pos1 = linePred.find(":",pos2+1);
          pos2 = linePred.find(" ",pos1+1);
          classDist.push_back(atof(linePred.substr(pos1+1,pos2).c_str()));
        }        
        sigmoid(classDist);
        //normalise(classDist);
        
        getline(tmpTestStream,lineTest);
        size_t classPos = lineTest.find(" "); // position of the end of the class
        unsigned long  trueClass = atoi(lineTest.substr(0,classPos).c_str())-1; 
        
        //Compute LFs
        const CatValue prediction = indexOfMaxVal(classDist);
        if (prediction != trueClass) {
            zeroOneLoss++;
        }
        const double error = 1.0-classDist[trueClass];
        squaredError += error * error;
        xtab[trueClass][prediction]++;
        foldxtab[trueClass][prediction]++;
        getline(vwOutput,linePred);
      }     
    }
    
   if(foldcount!=0){
     foldZOLoss.push_back(zeroOneLoss/static_cast<double>(foldcount));
     foldrmse.push_back(sqrt(squaredError/foldcount));
     expZOLoss.push_back(zeroOneLoss/static_cast<double>(foldcount));
     exprmse.push_back(sqrt(squaredError/foldcount));
   }
   
     double foldMCC = 0.0;
     foldMCC = calcMCC(foldxtab);
     printf("\n0-1 loss (fold %d): %0.4f\n", fold, zeroOneLoss/static_cast<double>(foldcount));
     printf("RMSE (fold %d): %0.4f\n", fold, sqrt(squaredError/foldcount));
     printf("MCC (fold %d): %0.4f\n", fold, foldMCC);
     printf("--------------------------------------------\n");
    //foldrmsea.push_back(sqrt(squaredErrorAll/(foldcount* noClasses)));
    
    vwOutput.close();
    tmpTestStream.close();
    remove(tmpTest.c_str());
    remove(raw.c_str());
    
    std::string cacheTrain =  dirTemp+"/cache_train"+toString(fold)+".vw";
    remove(cacheTrain.c_str());
    std::string cacheTest =  dirTemp+"/cache_test"+toString(fold)+".vw";
    remove(cacheTest.c_str());
   

}

void extLearnVW::printResults(InstanceStream *instanceStream, unsigned int exp){
    
    
  
    printf("Experiment %d:",exp+1);
    printf("\n0-1 loss:\n");
    printf("%0.4f", mean(foldZOLoss));
    printf("\n+/-:");
    printf("%0.4f", stddev(foldZOLoss));
    printf("\nRMSE:\n");
    printf("%0.4f", mean(foldrmse));
    printf("\n+/-:");
    printf("%0.4f\n", stddev(foldrmse));
    printf("----------------------------------------------------------------");


//          for (CatValue predicted = 0; predicted < instanceStream->getNoClasses(); predicted++) {
//            for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
//              xtab[y][predicted]/=noFolds;
//            }
//          }

    // Compute MCC for multi-class problems
    double MCC = 0.0;
    MCC = calcMCC(xtab);
    
    expMCC.push_back(MCC);

    printf("\nMCC:\n");
    printf("%0.4f\n", MCC);
    
    // Print the confusion matrix
    // find the maximum value to determine how wide the output fields need to be
    InstanceCount maxval = 0;
    for (CatValue predicted = 0; predicted < instanceStream->getNoClasses(); predicted++) {
      for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
        if (xtab[y][predicted] > maxval) maxval = xtab[y][predicted];
      }
    }

    // calculate how wide the output fields should be
    const int printwidth = max(4, printWidth(maxval));

    // print the heading line of class names
    printf("\n");
    for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
      printf(" %*.*s", printwidth, printwidth, instanceStream->getClassName(y));
    }
    printf(" <- Actual class\n");

    // print the counts of instances classified as each class
    for (CatValue predicted = 0; predicted < instanceStream->getNoClasses(); predicted++) {
      for (CatValue y = 0; y < instanceStream->getNoClasses(); y++) {
        printf(" %*" ICFMT, printwidth, xtab[y][predicted]);
      }
      printf(" <- %s predicted\n", instanceStream->getClassName(predicted));
    } 

    //Reset LFs for the fold.
    foldZOLoss.clear();
    foldrmse.clear();
    xtab.init(noClasses);
    
    if(exp == 9){ //Last one, print final average loss functions
        printf("\nMean 0-1 loss:\n");
        printf("%0.4f", mean(expZOLoss));
        printf("\n+/-:");
        printf("%0.4f", stddev(expZOLoss));
        printf("\nMean RMSE:\n");
        printf("%0.4f", mean(exprmse));
        printf("\n+/-:");
        printf("%0.4f", stddev(exprmse));
        printf("\nMean MCC:\n");
        printf("%0.4f\n\n", mean(expMCC));
        
  
#ifdef __linux__
 rmdir(dirTemp.c_str());
#endif
    }

}

