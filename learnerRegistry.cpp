/* Petal: An open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
**
** Module for registering each of the available learners.
** WOuld ideally use a map from char* to learner constructors, but c++ does not seem to support that
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
#include "learnerRegistry.h"
#include "utils.h"


/// 该访问方法应在初始化过程中用于访问学习者注册表，以确保对象在使用前已被初始化
LearnerRegistry& getLearnerRegistry() {
static LearnerRegistry theLearnerRegistry;
  return theLearnerRegistry;
}

learner* LearnerRegistry::createLearner(const char *name, char*const*& argv, char*const* end) {
  std::map<std::string, PtrToLearnerConstructor>::const_iterator it;

  it = constructors_.find(name);
  
  if (it != constructors_.end()) {
    return it->second(argv, end);
  }

  return NULL;
}

void LearnerRegistry::registerLearner(const char* name, PtrToLearnerConstructor constructor) {
  constructors_[name] = constructor;
}

void LearnerRegistry::getLearnerList(std::vector<std::string>& learners) const {
  for (std::map<std::string, PtrToLearnerConstructor>::const_iterator it = constructors_.begin(); it != constructors_.end(); it++) {
    learners.push_back(it->first);
  }
}

/// create an object of this type in order to register a new learner
LearnerRegistrar::LearnerRegistrar(const char* name, PtrToLearnerConstructor constructor) {
  getLearnerRegistry().registerLearner(name, constructor);
}

//// learners
//#include "AdaBoost.h"
//#include "baggedLearner.h"
//#include "DTree.h"
//#include "ensembleLearner.h"
//#include "featingLearner.h"
//#include "Feating2Learner.h"
//#include "Feating3Learner.h"
//#include "kdb.h"
//#include "kdbEager.h"
////#include "kdb2.h"
//#include "kdbExt.h"
//#include "kdbSelective.h"
//#include "kdbSelectiveClean.h"
//#include "kdbCDdisc.h"
//#include "kdbCDRAM.h"
//#include "nb.h"
//#include "tan.h"
//#include "random.h"
//#include "RFDTree.h"
//#include "aodeDist.h"
//#include "aode.h"
//#include "aodeBSE.h"
//#include "aodeEager.h"
//#include "a2de.h"
//#include "a2je.h"
//#include "a2de2.h"
//#include "a2de_ms.h"
//#include "a3de.h"
//#include "a3de_ms.h"
//#include "sampler.h"
//#include "dataStatistics.h"
//#include "kdbGaussian.h"
//#include "LR_SGD.h"
//#include "stackedLearner.h"
//#include "filteredLearner.h"
//
//// create a new learner, selected by name
//learner *createLearner(const char *learnername, char*const*& argv, char*const* end) {
//  if (streq(learnername, "adaboost")) {
//    return new AdaBoost(argv, end);
//  }
//  else if (streq(learnername, "aode")) {
//    return new aode(argv, end);
//  }
//  else if (streq(learnername, "aode-eager")) {
//    return new aodeEager(argv, end);
//  }
//  else if (streq(learnername, "aodeDist")) {
//	    return new aodeDist(argv, end);
//	  }
//  else if (streq(learnername, "a2de")) {
//    return new a2de(argv, end);
//  }
//  else if (streq(learnername, "a2de2")) {
//    return new a2de2(argv, end);
//  }
//  else if (streq(learnername, "a2de_ms")) {
//    return new a2de_ms(argv, end);
//  }
//  else if (streq(learnername, "a3de")) {
//    return new a3de(argv, end);
//  }
//  else if (streq(learnername, "a3de_ms")) {
//    return new a3de_ms(argv, end);
//  }
//  else if (streq(learnername, "a2je")) {
//    return new a2je(argv, end);
//  }
//  else if (streq(learnername, "bagging")) {
//    return new BaggedLearner(argv, end);
//  }
//  else if (streq(learnername, "bseaode")) {
//    return new aodeBSE(argv, end);
//  }
//  else if (streq(learnername, "dtree")) {
//    return new DTree(argv, end);
//  }
//  else if (streq(learnername, "ensembled")) {
//    return new EnsembleLearner(argv, end);
//  }
//  else if (streq(learnername, "feating")) {
//    return new FeatingLearner(argv, end);
//  }
//  else if (streq(learnername, "feating2")) {
//    return new Feating2Learner(argv, end);
//  }
//  else if (streq(learnername, "feating3")) {
//    return new Feating3Learner(argv, end);
//  }
//  else if (streq(learnername, "filtered")) {
//    return new FilteredLearner(argv, end);
//  }
//  else if (streq(learnername, "kdb")) {
//    return new kdb(argv, end);
//  }
//  else if (streq(learnername, "kdb-eager")) {
//    return new kdbEager(argv, end);
//  }
//  //else if (streq(learnername, "kdb2")) {
//  //  return new kdb2(argv, end);
//  //}
//  else if (streq(learnername, "kdb-ext")) {
//    return new kdbExt(argv, end);
//  }
//  else if (streq(learnername, "kdb-selective")) {
//    return new kdbSelective(argv, end);
//  }
//  else if (streq(learnername, "kdb-selectiveClean")) {
//    return new kdbSelectiveClean(argv, end);
//  }
//  else if (streq(learnername, "kdbCDdisc")) {
//    return new kdbCDdisc(argv, end);
//  }
//  else if (streq(learnername, "kdbCDRAM")) {
//    return new kdbCDRAM(argv, end);
//  }
//  else if (streq(learnername, "kdbGauss")) {
//    return new kdbGaussian(argv, end);
//  }
//  else if (streq(learnername, "lrsgd")) {
//    return new LR_SGD(argv, end);
//  }
//  else if (streq(learnername, "nb")) {
//    return new nb(argv, end);
//  }
//  else if (streq(learnername, "random")) {
//    return new randomClassifier(argv, end);
//  }
//  else if (streq(learnername, "rfdtree")) {
//    return new RFDTree(argv, end);
//  }
//  else if (streq(learnername, "sampler")) {
//    return new sampler(argv, end);
//  }
//  else if (streq(learnername, "stacked")) {
//    return new StackedLearner(argv, end);
//  }
//  else if (streq(learnername, "stats")) {
//    return new dataStatistics(argv, end);
//  }
//  else if (streq(learnername, "tan")) {
//    return new TAN(argv, end);
//  }
//  return NULL;
//}

