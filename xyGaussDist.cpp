/* Open source system for classification learning from very large data
** Copyright (C) 2012 Geoffrey I Webb
** Class for handling a joint distribution between an Attribute and a class
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


#include <cmath>
#include <algorithm>

#include "ALGLIB_specialfunctions.h"
#include "instanceStream.h"
#include "xyGaussDist.h"

xyGaussDist::xyGaussDist() {
  precision_ = 0.01;
}

xyGaussDist::~xyGaussDist() {
}

void xyGaussDist::init(unsigned int noOfClasses){
  noOfClasses_ = noOfClasses;
  
  counts_.assign(noOfClasses_,0);
  sumOfVals_.assign(noOfClasses_, 0);
  sumOfValsSq_.assign(noOfClasses_, 0);
  mean_.assign(noOfClasses_, 0);
  stdDev_.assign(noOfClasses_, precision_ / (2 * 3));    
}

void xyGaussDist::reset(InstanceStream *is) {

  noOfClasses_ = is->getNoClasses();

  counts_.assign(noOfClasses_,0);
  sumOfVals_.assign(noOfClasses_, 0);
  sumOfValsSq_.assign(noOfClasses_, 0);
  mean_.assign(noOfClasses_, 0);
  stdDev_.assign(noOfClasses_, precision_ / (2 * 3));
  
}

void xyGaussDist::update(NumValue v, CatValue y) {

  counts_[y] ++;
  v = round(v);
  sumOfVals_[y] += v;
  sumOfValsSq_[y] += v * v;
  mean_[y] = sumOfVals_[y] / counts_[y];
  double stdDev = std::sqrt(std::abs(sumOfValsSq_[y]-mean_[y] * sumOfVals_[y]) / counts_[y]);
  // As in Weka: if the stdDev ~= 0, we really have no idea of scale yet, so stick with the default. Otherwise...
  if (stdDev > 1e-10)
    stdDev_[y] = std::max(precision_/(2*3),stdDev);
 
}

double xyGaussDist::p(NumValue v, CatValue y) {
    v = round(v);
    
    double zLower = (v - mean_[y] - (precision_ / 2)) / stdDev_[y];
    double zUpper = (v - mean_[y] + (precision_ / 2)) / stdDev_[y];
    
    double pLower = alglib::normaldistribution(zLower);
    double pUpper = alglib::normaldistribution(zUpper);
    return pUpper - pLower; 
}

double xyGaussDist::p(CatValue y) {
    
    double zLower = (-precision_ / 2) / stdDev_[y];
    double zUpper = ( precision_ / 2) / stdDev_[y];
    
    double pLower = alglib::normaldistribution(zLower);
    double pUpper = alglib::normaldistribution(zUpper);
    return pUpper - pLower; 
}

void xyGaussDist::clear(){
  counts_.assign(noOfClasses_, 0);
  sumOfVals_.assign(noOfClasses_, 0);
  sumOfValsSq_.assign(noOfClasses_, 0);
  mean_.assign(noOfClasses_, 0);
  stdDev_.assign(noOfClasses_, precision_ / (2 * 3));
}
