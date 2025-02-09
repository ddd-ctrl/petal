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
#pragma once

/**
<!-- globalinfo-start -->
 * Class for registering learners and processing learner command line arguments.<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 * @author Geoff Webb (geoff.webb@monash.edu) 
 */

#include <map>

#include "learner.h"

typedef learner* (*PtrToLearnerConstructor)(char*const*& argv, char*const* end);

/// This class contains the set of learners and handles the creation of a new learner
class LearnerRegistry {
public:
  learner* createLearner(const char *name, char*const*& argv, char*const* end);
  void registerLearner(const char* name, PtrToLearnerConstructor constructor);
  void getLearnerList(std::vector<std::string>& learners) const;

private:
  std::map<std::string, PtrToLearnerConstructor> constructors_;
};

/// this access method should be used to access the learner registry during initialisation so as to ensure that the object is initialised before used
LearnerRegistry& getLearnerRegistry();

/// create a LearnerRegistrar in order to register a learner
class LearnerRegistrar {
public:
  LearnerRegistrar(const char* name, PtrToLearnerConstructor constructor);
};

inline learner* createLearner(const char *name, char*const*& argv, char*const* end) {
  return getLearnerRegistry().createLearner(name, argv, end);
}

template <typename T> learner* constructor(char*const*& argv, char*const* end) {
  return new T(argv, end);
}
