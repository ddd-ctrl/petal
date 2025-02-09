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

#pragma once

/**
<!-- globalinfo-start -->
 * Class for a Gaussian univariate (incremental) estimator over an attribute given the class .
 * It is different form xyDist in the sense that only one attribute is considered here, 
 * so it does not even have to be specified. <br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */
class xyGaussDist {
public:
    xyGaussDist();
    ~xyGaussDist();            ///< constructor without initialisation of InstanceStream specific data
    
    void init(unsigned int noOfClasses);
    
    void reset(InstanceStream *is); ///< initialise with InstanceStream specific information but do not read the distribution

    void update(NumValue v, CatValue y); ///< update the distribution according to the given instance
    
    double p(NumValue v, CatValue y); // p(a=v|y)
    double p(CatValue y); //p(a=mean|y)
    
    void clear();
    
    inline double round(double v) {
      return floor(v/precision_+0.5)*precision_;
    }
    
    inline NumValue getMean(CatValue y){
        return mean_[y];
    }
  
private:
    InstanceStream::MetaData* metaData_;
    unsigned int noOfClasses_;    ///< store the number of classes for use in indexing the inner vector
    
    std::vector<InstanceCount> counts_;      ///< current number of values seen,  per each attribute per class
    std::vector<double> sumOfVals_;   ///< current sum of values, per each attribute per class
    std::vector<double> sumOfValsSq_; ///< current sum of values squared, per each attribute per class 
    std::vector<NumValue> mean_;           ///< current mean, per attribute per class
    std::vector<double> stdDev_;   ///< current standard deviation, per attribute per class
    double precision_;                   ///< precission per each (numeric) attribute (i.e., minimum standar deviation allowed)
};


