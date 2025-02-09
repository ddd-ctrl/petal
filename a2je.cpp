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
 ** Please report any bugs to Shenglei Chen <tristan_chen@126.com>
 */

#include "a2je.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include "crosstab.h"
#include "utils.h"

a2je::a2je(char* const *& argv, char* const * end): pass_(1), weight_(1){
	name_ = "a2je";
	noCatAtts_ = 0;
	noClasses_ = 0;
	instanceStream_=NULL;
	weighted_ = false;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		} else if (streq(argv[0] + 1, "w")) {
			weighted_ = true;
		} else {
			error("a2je does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}

}

a2je::~a2je(void) {
}

void a2je::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a2je::reset(InstanceStream &is) {
	xxyDist_.reset(is);
	instanceStream_=&is;
	pass_=1;
	count_=0;
	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();


	weight_ = crosstab<float>(noCatAtts_);
	//initialise the weight for non-weighting

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			weight_[x1][x2] = 1;
			weight_[x2][x1] = 1;
		}
	}

}

void a2je::initialisePass() {

}

void a2je::train(const instance &inst) {
		xxyDist_.update(inst);
}

/// true iff no more passes are required. updated by finalisePass()
bool a2je::trainingIsFinished() {
		return pass_ > 1;
}


void a2je::finalisePass() {
	if(pass_==1)
	{
		if(weighted_)
		{
			//compute the mutual information between pair attributes and class as weight for a2de
			getPairMutualInf(xxyDist_,weight_);
			normalise(weight_);

			if(verbosity>=3)
			{
				printf("pair mutual information as weight for a2je:\n");
				for(unsigned int i=0;i<noCatAtts_;i++)
				{
					print(weight_[i]);
					printf("\n");
				}
			}

		}

	}
	++pass_;
}


void a2je::classify(const instance &inst, std::vector<double> &classDist) {
	const InstanceCount totalCount = xxyDist_.xyCounts.count;
	const std::vector<InstanceCount> &classCount=xxyDist_.xyCounts.classCounts;

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	//adding log of probabilities of all pairs
	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		const CatValue x1Val = inst.getCatVal(x1);
		const unsigned int noX1Vals = xxyDist_.getNoValues(x1);

		constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
				noClasses_);
		std::vector<float> weightX1=weight_[x1];

		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
			CatValue x2Val = inst.getCatVal(x2);

			const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

			//compute the number of pair appearing
			//if it has never appeared, drop this pair
			InstanceCount x1x2Count = xySubDist.getCount(
					x2, x2Val, 0);
			for (CatValue y = 1; y < noClasses_; y++) {
				x1x2Count += xySubDist.getCount(x2, x2Val, y);
			}
			if(x1x2Count==0){
				continue;
				if(verbosity>=3)
					printf("pair %u,%u is dropped.\n",x1,x2);
			}

			for (CatValue y = 0; y < noClasses_; y++) {
				const InstanceCount x1x2yCount = xySubDist.getCount(x2, x2Val, y);
				classDist[y]+=weightX1[x2]*log(mEstimate(x1x2yCount,classCount[y], noX1Vals*noX2Vals));
			}
		}
	}

	//compute the geometric average of p(x|y)
	//for even or odd number of attributes, we have (noCatAtts-1) p(x|y)
	double min=0;
	for (CatValue y = 0; y < noClasses_; y++){
		classDist[y] /=(noCatAtts_ - 1);
		classDist[y]+=log(mEstimate(classCount[y], totalCount, noClasses_));
		if(classDist[y]<min)
			min=classDist[y];
		if (verbosity >=4) {
			//printf("%d,%d,%d,%f",x1,x2,y,mEstimate(x1x2yCount,classCount[y], noX1Vals*noX2Vals));
			printf("count:y:%u, %f\n", y, classDist[y]);
			//printf("%u,%u,%f\n",x1x2yCount,classCount[y],log(mEstimate(x1x2yCount,classCount[y], noX1Vals*noX2Vals)));
		}
	}


	//as the log of distribution of some class is less than zero
	//so subtracting the minimal value will not overflows
	for (CatValue y = 0; y < noClasses_; y++){
		classDist[y]=exp(classDist[y]-min);
	}

	normalise(classDist);
}

