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

#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>#endif
#endif
#endif

#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>
#include <queue>

#include "kdbSelective.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "ALGLIB_specialfunctions.h"
#include "crosstab.h"

kdbSelective::kdbSelective(char*const*& argv, char*const* end) {
  name_ = "SELECTIVE-KDB";

  // defaults
  k_ = 1;
  
  selectiveMCC_ = false;
  selectiveTest_ = false;
  selectiveWeighted_ = false;
  selectiveK_ = false;
  selectiveLinks_ = false;
  minCount_ = 0;
  trainSize_ = 0;
  
  // get arguments
  while (argv != end) {
    if (*argv[0] != '+') {
      break;
    }
    else if (argv[0][1] == 'k') {
      getUIntFromStr(argv[0]+2, k_, "k");
    }
    else if (argv[0][1] == 'm') {
     getUIntFromStr(argv[0]+2, minCount_, "m");
    }
    else if (streq(argv[0]+1, "selectiveMCC")) {
      selectiveMCC_ = true;
    }
    else if (streq(argv[0]+1, "selectiveTest")) {
      selectiveTest_ = true;
    }
    else if (streq(argv[0]+1, "selectiveWeighted")) {
      selectiveWeighted_ = true;
    }
    else if (streq(argv[0]+1, "selectiveK")) {
      selectiveK_ = true;
    }
    else if (streq(argv[0]+1, "selectiveLinks")) {
      selectiveLinks_ = true;
      selectiveK_ = true;
    }
    else {
      break;
    }

    name_ += argv[0];

    ++argv;
  }
}

kdbSelective::~kdbSelective(void)
{
}

void  kdbSelective::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
}

void kdbSelective::reset(InstanceStream &is) {
  kdb::reset(is);
  
  order_.clear();
  bestKatt_.assign(noCatAtts_,0);
  active_.assign(noCatAtts_, true);
  foldLossFunct_.assign(noCatAtts_+1,0.0); //1 more for the prior
  binomialTestCounts_.assign(noCatAtts_+1,0); //1 more for the prior
  sampleSizeBinomTest_.assign(noCatAtts_+1,0);//1 more for the prior
  if (selectiveMCC_) {
      if(selectiveK_){
        xtab_.resize(k_+1);
        for(int i=0; i<= k_; i++){
          xtab_[i].resize( noCatAtts_+1, crosstab<InstanceCount>(noClasses_));
          for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            xtab_[i][a] = crosstab<InstanceCount>(noClasses_);
          }
        }
      }
      else{
          xtab_.resize(1);
          xtab_[0].resize( noCatAtts_+1, crosstab<InstanceCount>(noClasses_));
          for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            xtab_[0][a] = crosstab<InstanceCount>(noClasses_);
          }          
      }
  }
  if(selectiveK_){
    foldLossFunctallK_.resize(k_+1); //+1 for NB (k=0)
    for(int i=0; i<= k_; i++){
      foldLossFunctallK_[i].assign(noCatAtts_+1,0);
    }
  }
  inactiveCnt_ = 0;
  trainSize_ = 0;  
}

void kdbSelective::train(const instance &inst) {
  if (pass_ == 1) {
    // in the first pass collect the xxy distribution
    dist_.update(inst);
    trainSize_++; // to calculate the RMSE for each LOOCV
  }
  else if(pass_ == 2){
    // on the second pass collect the distributions to the k-dependence classifier
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      dTree_[a].update(inst, a, parents_[a]);
    }
    classDist_.update(inst);
  }else{
      assert(pass_ == 3); //only for selective KDB
      if(selectiveK_){
          std::vector<std::vector<double> > posteriorDist(k_+1);//+1 for NB (k=0)
          for(int k=0; k<= k_; k++){
              posteriorDist[k].assign(noClasses_,0.0);
          }
          //Only the class is considered
          for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist[0][y] = classDist_.ploocv(y, inst.getClass());//Discounting inst from counts
          }
          normalise(posteriorDist[0]);

          const CatValue trueClass = inst.getClass();
          
          if (selectiveMCC_) {
            const CatValue prediction = indexOfMaxVal(posteriorDist[0]);
            xtab_[0][noCatAtts_][trueClass][prediction]++;
            for(int k=1; k<=k_; k++){
              for (CatValue y = 0; y < noClasses_; y++){
                posteriorDist[k][y] = posteriorDist[0][y];
              }
              //normalise(posteriorDist[k]);
              xtab_[k][noCatAtts_][trueClass][prediction]++;
            }
          }else{
            const double error = 1.0-posteriorDist[0][trueClass];
            foldLossFunctallK_[0][noCatAtts_] += error*error;
            for(int k=1; k<= k_; k++){
              for (CatValue y = 0; y < noClasses_; y++){ 
                posteriorDist[k][y] = posteriorDist[0][y];
              }
              //normalise(posteriorDist[k]);
              foldLossFunctallK_[k][noCatAtts_] += error*error;
            }
          }
          
                    
          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                 it != order_.end(); it++){
              dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst, k_);//Discounting inst from counts
              for(int k=0; k<= k_; k++)
                normalise(posteriorDist[k]);
              
              if (selectiveMCC_) {
                for(int k=0; k<= k_; k++){
                  const CatValue prediction = indexOfMaxVal(posteriorDist[k]);
                  xtab_[k][*it][trueClass][prediction]++;
                }
              }else{
                for(int k=0; k<= k_; k++){
                  const double error = 1.0-posteriorDist[k][trueClass];
                  foldLossFunctallK_[k][*it] += error*error;
                }
              }
              
          }
      }else{
         //Proper kdb selective
         std::vector<double> posteriorDist(noClasses_);
         std::vector<double> errorsAtts; // Store the att errors for this instance (needed for selectiveTest)
         errorsAtts.assign(noCatAtts_+1,0.0);
         //Only the class is considered
         for (CatValue y = 0; y < noClasses_; y++) {
           posteriorDist[y] = classDist_.ploocv(y,inst.getClass());//Discounting inst from counts
         }
         normalise(posteriorDist);
         const CatValue trueClass = inst.getClass();
         const double error = 1.0-posteriorDist[trueClass];
         if (selectiveWeighted_) {
           foldLossFunct_[noCatAtts_] += (1.0-prior_[trueClass])*error*error;
         }
         else {
           foldLossFunct_[noCatAtts_] += error*error;
         }
         errorsAtts[noCatAtts_] = error;
         if (selectiveMCC_) {
           const CatValue prediction = indexOfMaxVal(posteriorDist);
           xtab_[0][noCatAtts_][trueClass][prediction]++;
         }

         for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                it != order_.end(); it++){
           dTree_[*it].updateClassDistributionloocv(posteriorDist, *it, inst);//Discounting inst from counts
           normalise(posteriorDist);
           const double error = 1.0-posteriorDist[trueClass];

           if (selectiveWeighted_) {
             foldLossFunct_[*it] += (1.0-prior_[trueClass]) * error*error;
           }
           else {
             foldLossFunct_[*it] += error*error;
           }
           errorsAtts[*it] = error;
           if (selectiveMCC_) {
             const CatValue prediction = indexOfMaxVal(posteriorDist);
             xtab_[0][*it][trueClass][prediction]++;
           }
         }

         if(selectiveTest_){
           double allerror = errorsAtts[order_[noCatAtts_-1]];
           //Only considering the class
           if ( (errorsAtts[noCatAtts_] - allerror) < 0.00001){//Draws are not counted
           }
           else if ( errorsAtts[noCatAtts_] < allerror ){
             binomialTestCounts_[noCatAtts_]++;
             sampleSizeBinomTest_[noCatAtts_]++;
           }
           else if ( errorsAtts[noCatAtts_] > allerror )//Draws are not counted
             sampleSizeBinomTest_[noCatAtts_]++;
             //For the rest of the attributes
             for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                                    it != order_.end()-1; it++){
               if( fabs(errorsAtts[*it]- allerror) < 0.00001){//Draws are not counted
               }
               else if( errorsAtts[*it] < allerror ){
                 binomialTestCounts_[*it]++;
                 sampleSizeBinomTest_[*it]++;
               }
               else if( errorsAtts[*it] > allerror ){
                 sampleSizeBinomTest_[*it]++;
               }
             }
         }
      }
  }
}

/// true iff no more passes are required. updated by finalisePass()
bool kdbSelective::trainingIsFinished() {
    return pass_ > 3;
}

void kdbSelective::classify(const instance &inst, std::vector<double> &posteriorDist) {
  const unsigned int noClasses = noClasses_;

  for (CatValue y = 0; y < noClasses; y++) {
    posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
  }

  for (CategoricalAttribute x = 0; x < noCatAtts_; x++) {
    if (active_[x]) {
      if (minCount_) {
        if(selectiveK_) {
          dTree_[x].updateClassDistributionForK(posteriorDist, x, inst, minCount_, bestKatt_[x]);
        }else {
          dTree_[x].updateClassDistribution(posteriorDist, x, inst, minCount_);
        }
      }
      else if(selectiveK_){
             dTree_[x].updateClassDistributionForK(posteriorDist, x, inst, bestKatt_[x]);
      }else{
             dTree_[x].updateClassDistribution(posteriorDist, x, inst);
      }
    }
  }

  normalise(posteriorDist);
}

// creates a comparator for two attributes based on their relative mutual information with the class
class miCmpClass {
public:
  miCmpClass(std::vector<float> *m) {
    mi = m;
  }

  bool operator() (CategoricalAttribute a, CategoricalAttribute b) {
    return (*mi)[a] > (*mi)[b];
  }

private:
  std::vector<float> *mi;
};


void kdbSelective::finalisePass() {
  if (pass_ == 1) {
    
    std::vector<float> mi;  
    crosstab<float> cmi = crosstab<float>(noCatAtts_);  //CMI(X;Y|C) = H(X|C) - H(X|Y,C) -> cmi[X][Y]
              
    getMutualInformation(dist_.xyCounts, mi);
    getCondMutualInf(dist_,cmi);

    if (selectiveWeighted_) {
      for (CatValue y = 0; y < dist_.getNoClasses(); ++y) {
        prior_.push_back(dist_.xyCounts.p(y));
      }
    }
    
    dist_.clear();
    
    
    if (verbosity >= 4) {
      printf("\nMutual information table\n");
      print(mi);
      putchar('\n');
    }
    
    if (verbosity >= 4) {
      printf("\nConditional mutual information table\n");
      cmi.print();
    }
    
    // sort the attributes on MI with the class

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
      order_.push_back(a);
    }

    if (!order_.empty()) {
      miCmpClass cmp(&mi);

      std::sort(order_.begin(), order_.end(), cmp);
      if(verbosity>=4){
        printf("Att. Order (mi): ");
        for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); 
                                                               it != order_.end(); it++){
          printf("%s,",instanceStream_->getCatAttName(*it));
        }     
        putchar('\n');
      }
          
       
       // proper KDB assignment of parents
       if (verbosity >= 4) {
         printf("\n%s parents:\n", instanceStream_->getCatAttName(order_[0]));
       }
       
       for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin()+1;  
                                                              it != order_.end(); it++){
         parents_[*it].push_back(order_[0]);
         for (std::vector<CategoricalAttribute>::const_iterator 
                                           it2 = order_.begin()+1; it2 != it; it2++) {
           // make parents into the top k attributes on mi that precede *it in order
           if (parents_[*it].size() < k_) {
             // create space for another parent
             // set it initially to the new parent.
             // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
             parents_[*it].push_back(*it2);
           }
           for (unsigned int i = 0; i < parents_[*it].size(); i++) {
             if (cmi[*it2][*it] > cmi[parents_[*it][i]][*it]) {
               // move lower value parents down in order
               for (unsigned int j = parents_[*it].size()-1; j > i; j--) {
                 parents_[*it][j] = parents_[*it][j-1];
               }
               // insert the new att
               parents_[*it][i] = *it2;
               break;
             }
           }
         }
         if (verbosity >= 4) {
           printf("%s parents: ", instanceStream_->getCatAttName(*it));
           for (unsigned int i = 0; i < parents_[*it].size(); i++) {
             printf("%s ", instanceStream_->getCatAttName(parents_[*it][i]));
           }
           putchar('\n');
         }
       }
    }
  }
  else if(pass_ == 3) {//only for selective KDB
    int bestatt;

    if(selectiveK_){
      if (selectiveMCC_){
        for (unsigned int k=0; k<=k_;k++) {
          foldLossFunctallK_[k][noCatAtts_] = -calcMCC(xtab_[k][noCatAtts_]);
          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
            const unsigned int i = *it;
            foldLossFunctallK_[k][i] = -calcMCC(xtab_[k][i]);
          }
        }
      }else{//Proper kdb selective (RMSE)      
        for (unsigned int k=0; k<=k_;k++) {
          for (unsigned int att=0; att<noCatAtts_+1;att++) {
            foldLossFunctallK_[k][att] = sqrt(foldLossFunctallK_[k][att]/trainSize_);
          }
        }
      }
      if(verbosity>=3){
        if(selectiveMCC_){
          printf("MCC: \n");
          for (unsigned int k=0; k<=k_;k++) {
            printf("k = %d : ",k);
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
              printf("%.3f,", -foldLossFunctallK_[k][*it]);
            }
            printf("%.3f(class)\n", -foldLossFunctallK_[k][noCatAtts_]);
          }
        }
        else{
          printf("RMSE: \n");
          for (unsigned int k=0; k<=k_;k++) {
            printf("k = %d : ",k);
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
              printf("%.3f,", foldLossFunctallK_[k][*it]);
            }
            printf("%.3f(class)\n", foldLossFunctallK_[k][noCatAtts_]);
          }
        }
      }

      
      bestatt = noCatAtts_;
      unsigned int bestK = 0; //naive Bayes
      if(!selectiveLinks_){
        double min = foldLossFunctallK_[0][noCatAtts_];
        for (unsigned int k=0; k<=k_;k++) {
          for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
            if(foldLossFunctallK_[k][*it] < min){
              min = foldLossFunctallK_[k][*it];
              bestatt = *it;
              bestK = k;
            }
          }
        }
        bestKatt_.assign(noCatAtts_, bestK);
      }else{      
          //Remove parents
          double globalmin = foldLossFunctallK_[0][noCatAtts_];
            for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
              double min = foldLossFunctallK_[0][*it];
              for (unsigned int k=1; k<=k_;k++) {
                  if(foldLossFunctallK_[k][*it] < globalmin){
                    globalmin = foldLossFunctallK_[k][*it];
                    bestatt = *it;
                  }
                  if(foldLossFunctallK_[k][*it] < min){
                    min = foldLossFunctallK_[k][*it];   
                    bestKatt_[*it] = k;
                  }
              }
            }
      }
      
    }
    else{//proper selective

      if (selectiveMCC_){
        foldLossFunct_[noCatAtts_] = -calcMCC(xtab_[0][noCatAtts_]);
        for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
          const unsigned int i = *it;
          foldLossFunct_[i] = -calcMCC(xtab_[0][i]);
        }
      }
      else{
        for (unsigned int att=0; att<foldLossFunct_.size();att++) {
          foldLossFunct_[att] = sqrt(foldLossFunct_[att]/trainSize_);
        }
      }
      if(verbosity>=3){
        if (selectiveMCC_)
          printf("MCC: ");
        else
          printf("RMSE: ");
        for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
          printf("%.3f,", foldLossFunct_[*it]);
        }
        printf("%.3f(class)", foldLossFunct_[noCatAtts_]);
      }

      if(verbosity>=3)
        putchar('\n');      

      //Find the best attribute in order (to resolve ties in the best way possible)
      //It is the class only by default
      double min = foldLossFunct_[noCatAtts_];
      bestatt = noCatAtts_;
      for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
        if(foldLossFunct_[*it] < min){
          min = foldLossFunct_[*it];
          bestatt = *it;
        }
      }
    }

    if(selectiveTest_){
      //H_0 -> There is no difference between selecting until bestatt or taking all the attributes
      if (alglib::binomialcdistribution(binomialTestCounts_[bestatt], 
        sampleSizeBinomTest_[bestatt],0.5) > 0.05)
        //Complemented binomial distribution: calculates p(x>binomialTestCounts[bestatt])
        //H_0 is accepted
        bestatt = order_[noCatAtts_-1];
    }

    bool erase;
    if(bestatt==noCatAtts_)
      erase = true;
    else 
      erase = false;     
    for (std::vector<CategoricalAttribute>::const_iterator it = order_.begin(); it != order_.end(); it++){
        if(erase)
        {
          active_[*it] = false;
          inactiveCnt_++;
        }
        else if(*it==bestatt)
          erase=true;                  
    }
    if(verbosity>=2){
      printf("Number of features selected is: %d out of %d\n",noCatAtts_-inactiveCnt_, noCatAtts_);
      if(selectiveK_ && !selectiveLinks_)
        printf("best k is: %d\n",bestKatt_[0]);
      if(selectiveLinks_){
        printf("best k is: %.2f\n",sum(bestKatt_)/static_cast<double>(bestKatt_.size()));
        printf("Number of parents per attribute selected is: ");
        //print(bestKatt_);
        const char *sep = "";
          for (std::vector<unsigned int>::const_iterator it = order_.begin(); it != order_.end(); it++) {
            printf("%s%d", sep, bestKatt_[*it]);
            sep = ", ";
          }
        putchar('\n');
      }
    }
      
    
  }else{
    assert(pass_ == 2);
  }
  ++pass_;
}


void kdbSelective::printClassifier() {
}
