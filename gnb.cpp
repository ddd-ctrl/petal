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

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include "correlationMeasures.h"
#include "utils.h"
#include "gnb.h"
#include "globals.h"
#include "instanceStreamDiscretiser.h"

gnb::gnb(char*const*&, char*const*) : xyDist_(), trainingIsFinished_(false)
 {
	name_ = "Gaussian Naive Bayes";
}


gnb::~gnb(void)
{
}

void gnb::getCapabilities(capabilities &c){
  c.setCatAtts(true);  // only categorical attributes are supported at the moment
  c.setNumAtts(true); 
}

void gnb::reset(InstanceStream &is) {
  trainingIsFinished_ = false;
  noCatAtts_ = is.getNoCatAtts();
  xyDist_.reset(&is, noCatAtts_); //In is the attributes are considered discretized  
  noNumAtts_ = is.getNoNumAtts();
  xyGaussDist_.resize(noNumAtts_);
  noClasses_=is.getNoClasses();
  for (NumericAttribute a = 0; a < noNumAtts_; a++) {
    xyGaussDist_[a].init(noClasses_);
  }
  classDist_.reset(is);
  instanceStream_ = &is;
}


void gnb::train(const instance &inst) {
  xyDist_.update(inst);
  for (NumericAttribute a = 0; a < noNumAtts_; a++) {
      if(!inst.isMissing(a)){
        xyGaussDist_[a].update(inst.getNumVal(a), inst.getClass());
      }
  }
  classDist_.update(inst);
}


void gnb::initialisePass() {
}


void gnb::finalisePass() {    
    trainingIsFinished_ = true;
}


bool gnb::trainingIsFinished() {
  return trainingIsFinished_;
}

void gnb::classify(const instance &inst, std::vector<double> &classDist) {
	const unsigned int noClasses = xyDist_.getNoClasses();
        
        for (CatValue y = 0; y < noClasses; y++) {
            classDist[y] = classDist_.p(y);
        }

        for (CategoricalAttribute a = 0; a < xyDist_.getNoAtts(); a++) {
            double temp, max = 0;
            for (CatValue y = 0; y < noClasses; y++) {
                temp = std::max(1e-75,xyDist_.p(a, inst.getCatVal(a), y));
                classDist[y] *= temp;
                if (classDist[y] > max) {
                  max = classDist[y];
                }
            }    
            if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
                for (int j = 0; j < noClasses; j++) {
                   classDist[j] *= 1e75;
                }
            }
        }
        
        for (NumericAttribute a = 0; a < noNumAtts_; a++) {
            if(!inst.isMissing(a)){
                double temp, max = 0;
                for (CatValue y = 0; y < noClasses; y++) {
                    temp = std::max(1e-75,xyGaussDist_[a].p(inst.getNumVal(a),y));
                    classDist[y] *= temp;
                    if (classDist[y] > max) {
                      max = classDist[y];
                    }
                }    
                if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
                    for (int j = 0; j < noClasses; j++) {
                       classDist[j] *= 1e75;
                    }
                }
            }
        }
  

	normalise(classDist);
}


