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

#include "aodeBSE.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include "crosstab.h"
#include "ALGLIB_specialfunctions.h"
#include "learnerRegistry.h"


static LearnerRegistrar registrar("aodebse", constructor<aodeBSE>);

aodeBSE::aodeBSE(char* const *& argv, char* const * end) {
	name_ = "AODEBSE";

	empiricalMEst_ = false;
	empiricalMEst2_ = false;

	ce_=false;
	pe_=false;
	poce_=false;
	pace_=false;
	loo_=false;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		} else if (streq(argv[0] + 1, "loo")) {
			loo_ = true;
		} else if (streq(argv[0] + 1, "ce")) {
			ce_ = true;
		} else if (streq(argv[0] + 1, "pe")) {
			pe_ = true;
		} else if (streq(argv[0] + 1, "pace")) {
			pace_ = true;
		} else if (streq(argv[0] + 1, "poce")) {
			poce_ = true;
		} else if (streq(argv[0] + 1, "empirical")) {
			empiricalMEst_ = true;
		} else if (streq(argv[0] + 1, "empirical2")) {
			empiricalMEst2_ = true;
		} else {
			error("Aode does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}
}

aodeBSE::aodeBSE(const aodeBSE& l) {
  name_ = l.name_;
  empiricalMEst_ = l.empiricalMEst_;
  empiricalMEst2_ = l.empiricalMEst2_;
  loo_=l.loo_;
  ce_=l.ce_;
  pe_=l.pe_;
  poce_=l.poce_;
  pace_=l.pace_;
}

learner* aodeBSE::clone() const {
  return new aodeBSE(*this);
}

aodeBSE::~aodeBSE(void) {
}

void aodeBSE::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aodeBSE::reset(InstanceStream &is) {
	xxyDist_.reset(is);

	count_=0;
	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	noChild_.resize(noCatAtts_,noCatAtts_);

	trainingIsFinished_=false;

	instanceStream_ = &is;
	pass_ = 1;

	activeChild_.assign(noCatAtts_, true);
	activeParent_.assign(noCatAtts_, true);
	activeParentChild_.assign(noCatAtts_, true);

	noActiveChild_=noCatAtts_;
	noActiveParent_=noCatAtts_;

	squaredErrorAode_=0;

	squaredErrorChild_.assign(noCatAtts_,1.0);
	squaredErrorParent_.assign(noCatAtts_,1.0);
	squaredErrorParentChild_.assign(noCatAtts_,1.0);


	minError_.assign(noCatAtts_,0.0);
}

void aodeBSE::initialisePass() {
	count_=0;
	if(pass_==2)
	{
		predictedBest_.resize(xxyDist_.xyCounts.count,false);
	}
	else if(pass_>2)
	{
		predicted_.resize(noCatAtts_);
		for (CategoricalAttribute att = 0; att < noCatAtts_;att++) {
			predicted_[att].resize(xxyDist_.xyCounts.count,false);
		}
	}
}

void aodeBSE::train(const instance &inst) {

	if (pass_ == 1)
		xxyDist_.update(inst);
	else {
		LOOCV(inst);
	}
}

/// true iff no more passes are required. updated by finalisePass()
bool aodeBSE::trainingIsFinished() {
	if (loo_==true)
		return trainingIsFinished_;
	else
		return pass_ > 1;
}

void aodeBSE::finalisePass() {

	if(verbosity>=2)
		printf("pass %u.\n",pass_);

	if(pass_==2)
	{
		squaredErrorAode_= sqrt(squaredErrorAode_/ xxyDist_.xyCounts.count);
		minError_[pass_-2]=squaredErrorAode_;
		if(verbosity>=2)
		{
			printf("error 0: %f.\n",squaredErrorAode_);
		}
	}else if(pass_>2)
	{

		if(pass_==10)
			pass_=10;


		if(verbosity>=3)
		{
			printf("child:\n");
			print(squaredErrorChild_);
			putchar('\n');
		}


		for (CategoricalAttribute att = 0; att < noCatAtts_;att++) {

			if(ce_==true||poce_==true)
			{
				if(activeChild_[att]==true)
					squaredErrorChild_[att] = sqrt(squaredErrorChild_[att]/ xxyDist_.xyCounts.count);
				else
					squaredErrorChild_[att] = 1;
			}
			if(pe_==true||poce_==true)
			{
				if(activeParent_[att]==true)
					squaredErrorParent_[att] = sqrt(squaredErrorParent_[att]/ xxyDist_.xyCounts.count);
				else
					squaredErrorParent_[att] = 1;

			}
			if(pace_==true||poce_==true)
			{
				if(activeParentChild_[att]==true)
					squaredErrorParentChild_[att] = sqrt(squaredErrorParentChild_[att]/ xxyDist_.xyCounts.count);
				else
					squaredErrorParentChild_[att] = 1;
			}


		}

		unsigned int minimalChild;
		unsigned int minimalParent;
		unsigned int minimalParentChild;

		//use SFN
//		if(verbosity>=3)
//		{
//			print(squaredError_);
//			printf("\nMinimal is att %u with value %f of %u.\n",indexOfMinVal(squaredError_),squaredError_[minimal],noCatAtts_);
//		}


		if(poce_==true)
		{

			minimalChild=indexOfMinVal(squaredErrorChild_);
			minimalParent=indexOfMinVal(squaredErrorParent_);
			minimalParentChild=indexOfMinVal(squaredErrorParentChild_);

			double minSquaredErrorChild=squaredErrorChild_[minimalChild];
			double minSquaredErrorParent=squaredErrorParent_[minimalParent];
			double minSquaredErrorParentChild=squaredErrorParentChild_[minimalParentChild];


			//if the error of child elimination is the lowest

			if(minSquaredErrorChild<=minSquaredErrorParent&&
					minSquaredErrorChild<=minSquaredErrorParentChild)
			{

				if(minSquaredErrorChild>=minError_[pass_-3]
				                                   ||noActiveChild_==1)
				{
					trainingIsFinished_=true;
				}
				else
				{
					activeChild_[minimalChild]=false;
					minError_[pass_-2]=minSquaredErrorChild;

					noActiveChild_--;
					if(verbosity>=2)
					{
							print(squaredErrorChild_);
							printf("\n%u is removed as a child.\n",minimalChild);
							printf("Min val is %f.\n",minSquaredErrorChild);
					}
					squaredErrorChild_.assign(noCatAtts_,0.0);
					squaredErrorParent_.assign(noCatAtts_,0.0);
					squaredErrorParentChild_.assign(noCatAtts_,0.0);
				}
			}else if(minSquaredErrorParent<=minSquaredErrorChild
				&&minSquaredErrorParent<=minSquaredErrorParentChild)
			{
				if(minSquaredErrorParent>=minError_[pass_-3]
					||noActiveParent_==1)
				{
					trainingIsFinished_=true;
				}
				else
				{
					activeParent_[minimalParent]=false;
					minError_[pass_-2]=minSquaredErrorParent;
					noActiveParent_--;

					if(verbosity>=2)
					{
							print(squaredErrorParent_);
							printf("\n%u is removed as a parent.\n",minimalParent);
							printf("Min val is %f.\n",minSquaredErrorParent);
					}
					squaredErrorChild_.assign(noCatAtts_,0.0);
					squaredErrorParent_.assign(noCatAtts_,0.0);
					squaredErrorParentChild_.assign(noCatAtts_,0.0);

				}

			}else if(minSquaredErrorParentChild<=minSquaredErrorChild
					&&minSquaredErrorParentChild<=minSquaredErrorParent)
			{

				if(minSquaredErrorParentChild>=minError_[pass_-3]
					||noActiveParent_==1||noActiveChild_==1)
				{
					trainingIsFinished_=true;
				}
				else
				{
					activeParentChild_[minimalParentChild]=false;

					activeParent_[minimalParentChild]=false;
					activeChild_[minimalParentChild]=false;
					minError_[pass_-2]=minSquaredErrorParentChild;

					noActiveParent_--;
					noActiveChild_--;

					if(verbosity>=2)
					{
						print(squaredErrorParentChild_);
						printf("\n%u is removed as parent and child.\n",minimalParentChild);
						printf("Min val is %f.\n",minSquaredErrorParentChild);
					}
					squaredErrorChild_.assign(noCatAtts_,0.0);
					squaredErrorParent_.assign(noCatAtts_,0.0);
					squaredErrorParentChild_.assign(noCatAtts_,0.0);

				}
			}
		}else if(ce_==true){

			minimalChild=indexOfMinVal(squaredErrorChild_);

			//stop when the error is larger than the previous or
			//there is only one active attribute
			if(squaredErrorChild_[minimalChild]>=minError_[pass_-3]
			      ||noActiveChild_==1)
			{
				trainingIsFinished_=true;
			}
			else
			{
				activeChild_[minimalChild]=false;
				minError_[pass_-2]=squaredErrorChild_[minimalChild];
				noActiveChild_--;

				if(verbosity>=2)
				{
						printf("%u is removed as a child, rmse is %f.\n",
								minimalChild,squaredErrorChild_[minimalChild]);
				}
				squaredErrorChild_.assign(noCatAtts_,0.0);
			}


		}else if(pe_==true)	{
			minimalParent=indexOfMinVal(squaredErrorParent_);

			//stop when the error is larger than the previous or
			//there is only one active attribute
			if (squaredErrorParent_[minimalParent] >= minError_[pass_ - 3]
					|| noActiveParent_ == 1) {
				trainingIsFinished_ = true;
			}
			else
			{
				activeParent_[minimalParent]=false;
				minError_[pass_-2]=squaredErrorParent_[minimalParent];
				noActiveParent_--;
				if(verbosity>=2)
				{
						printf("%u is removed as a parent, rmse is %f.\n",
								minimalParent,squaredErrorParent_[minimalParent]);
				}

				squaredErrorParent_.assign(noCatAtts_,0.0);
			}

		}else if(pace_==true)	{

			minimalParentChild=indexOfMinVal(squaredErrorParentChild_);

			if(squaredErrorParentChild_[minimalParentChild]>=minError_[pass_-3]||noActivePC_==1)
			{
				trainingIsFinished_=true;
			}
			else
			{
				activeParentChild_[minimalParentChild]=false;
				minError_[pass_-2]=squaredErrorParentChild_[minimalParentChild];
				noActivePC_--;
				if(verbosity>=2)
				{
						printf("%u is removed as a parent and child, rmse is %f.\n",
								minimalParentChild,squaredErrorParentChild_[minimalParentChild]);
				}


				squaredErrorParentChild_.assign(noCatAtts_,0.0);
			}

		}








/*
		//use sign test to stop
		win_=0;
	 	loss_=0;
		for (unsigned int i = 0; i < count_; i++) {
			if(predictedBest_[i]==false&&predicted_[minimal][i]==true)
				win_++;
			if(predictedBest_[i]==true&&predicted_[minimal][i]==false)
				loss_++;
			predictedBest_[i] = predicted_[minimal][i];
		}

		double significance=alglib::binomialcdistribution(win_-1, win_+loss_,0.5);

		if(significance>=0.05)
			trainingIsFinished_=true;
*/
	}

	++pass_;

}

void aodeBSE::LOOCV(const instance &inst)
{

	const InstanceCount totalCount = xxyDist_.xyCounts.count-1;

	const CatValue trueClass = inst.getClass();

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;
//	scaleFactor =1;

	CatValue delta = 0;

	//try to increase the efficiency

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);
	std::vector<std::vector<std::vector<double> > > model;


	// initial spode assignment of joint probability of parent and class
	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		const CatValue parentVal = inst.getCatVal(parent);
		CatValue parentCount=0;
		for (CatValue y = 0; y < noClasses_; y++) {
			if(y==trueClass)
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
					parentVal, y)-1;
			else
				xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
									parentVal, y);
			parentCount+=xyCount[parent][y];
		}

		if (parentCount > 0) {
			delta++;
			active[parent] = true;
			for (CatValue y = 0; y < noClasses_; y++) {
				spodeProbs[parent][y] = mEstimate(xyCount[parent][y], totalCount,
								noClasses_* xxyDist_.getNoValues(parent))
						* scaleFactor;
			}
		}
	}


	if (delta == 0) {
		printf("there are no eligible parents.\n");
		return;
	}


	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {

		if(activeParentChild_[x1]==false)
			continue;

		const CatValue x1Val = inst.getCatVal(x1);
		const unsigned int noX1Vals = xxyDist_.getNoValues(x1);
		const bool x1Active = active[x1];

		constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
				noClasses_);

		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			if(activeParentChild_[x2]==false)
				continue;

			const bool x2Active = active[x2];

			if ((x1Active&&activeChild_[x2]==true&&activeParent_[x1]==true) || (x2Active&&activeChild_[x1]==true&&activeParent_[x2]==true) ) {
				CatValue x2Val = inst.getCatVal(x2);
				const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

				for (CatValue y = 0; y < noClasses_; y++) {
					InstanceCount x1x2yCount;

					if(y==trueClass)
						x1x2yCount = xySubDist.getCount(x2, x2Val, y)-1;
					else
						x1x2yCount = xySubDist.getCount(x2, x2Val, y);

					if (x1Active&&activeChild_[x2]==true&&activeParent_[x1]==true) {

							spodeProbs[x1][y] *= mEstimate(x1x2yCount,
									xyCount[x1][y], noX2Vals);
							if(verbosity>=2&&pass_==7&&count_==1&&x1==8&&y==0)
							{
								printf("4->parent %u has child:%u\n",x1,x2);

							}
					}
					if (x2Active&&activeChild_[x1]==true&&activeParent_[x2]==true) {
							spodeProbs[x2][y] *= mEstimate(x1x2yCount,
									xyCount[x2][y], noX1Vals);
							if(verbosity>=2&&pass_==7&&count_==1&&x2==8&&y==0)
							{
								printf("4->parent %u has child:%u\n",x2,x1);

							}

					}
				}
			}
		}
	}


	std::vector < double > classDist(noClasses_,0);

	if(pass_==2)
	{

		for (CatValue parent = 0; parent < noCatAtts_; parent++) {
			if (active[parent]) {
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent][y];
				}
			}
		}
		normalise(classDist);
		const double error = 1.0 - classDist[trueClass];
		squaredErrorAode_ += error * error;

		if(trueClass==indexOfMaxVal(classDist))
		{
			predictedBest_[count_]=true;
		}
	}
	else
	{

		model.resize(noCatAtts_);

		//child elimination
		if(ce_==true||poce_==true)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {

				if(activeParentChild_[parent]==false)
					continue;
				if(activeParent_[parent]==false)
					continue;

				if (active[parent]) {

					const CatValue parentVal = inst.getCatVal(parent);

					model[parent].resize(noCatAtts_);
					for (CatValue child = 0; child < noCatAtts_; child++) {

						if(activeParentChild_[child]==false)
							continue;
						if(activeChild_[child]==false)
							continue;
						const unsigned int noChildVals = xxyDist_.getNoValues(
								child);

						CatValue childVal = inst.getCatVal(child);

						model[parent][child].resize(noClasses_);

						for (CatValue y = 0; y < noClasses_; y++) {
							if(child==parent)
								model[parent][child][y]=spodeProbs[parent][y];
							else
							{
								InstanceCount x1x2yCount= xxyDist_.getCount(parent,parentVal,child, childVal,
										y);
								if (y == trueClass)
									x1x2yCount -=1;
								model[parent][child][y]=spodeProbs[parent][y]/mEstimate(x1x2yCount,xyCount[parent][y], noChildVals);

								if(verbosity>=2&&pass_==7&&count_==1&&child==2)
								{
									printf("3->parent:%u,y:%u,\n%f,\n%f,\n%f\n",parent,y,spodeProbs[parent][y],mEstimate(x1x2yCount,xyCount[parent][y], noChildVals),model[parent][child][y]);

								}
							}

						}
					}
				}
			}

			for(CatValue att=0;att<noCatAtts_;att++)
			{

				if(activeParentChild_[att]==false)
					continue;

				if(activeChild_[att]==false)
					continue;


				for (CatValue y = 0; y < noClasses_; y++)
					classDist[y] = 0;

				for (CatValue parent = 0; parent < noCatAtts_; parent++) {

					if(activeParentChild_[parent]==false)
						continue;
					if(activeParent_[parent]==false)
						continue;

					if (active[parent]) {
						for (CatValue y = 0; y < noClasses_; y++) {
							classDist[y] += model[parent][att][y];

							if(verbosity>=2&&pass_==7&&count_==1&&att==2)
							{
								printf("2->att:%u,parent: %u,y:%u,%f\n",att,parent,y,model[parent][att][y]);
							}
						}
					}
				}

				normalise(classDist);
				const double error = 1.0 - classDist[trueClass];


				if( verbosity>=2&&pass_==7&& count_<20)
					printf("1->pass: %u, count: %u, att: %u, %f\n",pass_,count_,att, error);
				squaredErrorChild_[att] += error * error;

				if(trueClass==indexOfMaxVal(classDist))
				{
					predicted_[att][count_]=true;
				}
			}

		}

		//parent elimination
		if(pe_==true||poce_==true)
		{


			for (CatValue parent = 0; parent < noCatAtts_; parent++) {

				if(activeParentChild_[parent]==false)
					continue;

				if(activeParent_[parent]==false)
					continue;
				if (active[parent]) {
					for (CatValue y = 0; y < noClasses_; y++) {
						classDist[y] += spodeProbs[parent][y];
					}
				}
			}
			std::vector < double > classDistForParent(noClasses_,0);
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {

				if(activeParentChild_[parent]==false)
					continue;

				if(activeParent_[parent]==false)
					continue;
				if (active[parent]) {
					for (CatValue y = 0; y < noClasses_; y++) {
						classDistForParent[y]=classDist[y]-spodeProbs[parent][y];
					}

					normalise(classDistForParent);
					const double error = 1.0 - classDistForParent[trueClass];
					squaredErrorParent_[parent] += error * error;

					if(trueClass==indexOfMaxVal(classDistForParent))
					{
						predicted_[parent][count_]=true;
					}

				}
			}
		}

		//parent and child elimination
		if(pace_==true||poce_==true)
		{
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {

				if(activeParentChild_[parent]==false)
					continue;
				if(activeParent_[parent]==false)
					continue;

				if (active[parent]) {

					const CatValue parentVal = inst.getCatVal(parent);

					model[parent].resize(noCatAtts_);
					for (CatValue child = 0; child < noCatAtts_; child++) {

						if(activeParentChild_[child]==false)
							continue;
						if(activeChild_[child]==false)
							continue;

						const unsigned int noChildVals = xxyDist_.getNoValues(
								child);

						CatValue childVal = inst.getCatVal(child);

						model[parent][child].resize(noClasses_);

						for (CatValue y = 0; y < noClasses_; y++) {
							if(child==parent)
								model[parent][child][y]=spodeProbs[parent][y];
							else
							{
								InstanceCount x1x2yCount= xxyDist_.getCount(parent,parentVal,child, childVal,
										y);
								if (y == trueClass)
									x1x2yCount -=1;
								model[parent][child][y]=spodeProbs[parent][y]/mEstimate(x1x2yCount,
																	xyCount[parent][y], noChildVals);
							}

						}
					}
				}
			}

			fdarray<double> modelChild(noCatAtts_, noClasses_);

			modelChild.clear();


			std::vector < double > classDistForParent(noClasses_,0);
			for (CatValue parent = 0; parent < noCatAtts_; parent++) {
				if (active[parent]) {

					if(activeParentChild_[parent]==false)
						continue;
					if(activeParent_[parent]==false)
						continue;
					for (CatValue child = 0; child < noCatAtts_; child++) {

						if(activeParentChild_[child]==false)
							continue;
						if(activeChild_[child]==false)
							continue;
						for (CatValue y = 0; y < noClasses_; y++) {
							modelChild[child][y]+=model[parent][child][y];
						}
					}
				}
			}

			for(CatValue att=0;att<noCatAtts_;att++)
			{
				if(activeParentChild_[att]==false)
					continue;
				if(activeParent_[att]==false)
					continue;
				if(activeChild_[att]==false)
					continue;
				if (active[att]) {
					for (CatValue y = 0; y < noClasses_; y++) {
						classDist[y] = modelChild[att][y]-model[att][att][y];
					}

					normalise(classDist);
					const double error = 1.0 - classDist[trueClass];
					squaredErrorParentChild_[att] += error * error;

					if(trueClass==indexOfMaxVal(classDist))
					{
						predicted_[att][count_]=true;
					}
				}

			}
		}
	}

	count_++;
}


void aodeBSE::classify(const instance &inst, std::vector<double> &classDist) {


	if(verbosity>=4)
		count_++;

	const InstanceCount totalCount = xxyDist_.xyCounts.count;

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;
	CatValue delta = 0;

	//try to increase the efficiency

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);


	for (CatValue parent = 0; parent < noCatAtts_; parent++) {



		//discard the attribute that is not active or in generalization set
		const CatValue parentVal = inst.getCatVal(parent);

		for (CatValue y = 0; y < noClasses_; y++) {
			xyCount[parent][y] = xxyDist_.xyCounts.getCount(parent,
					parentVal, y);
		}

		if (xxyDist_.xyCounts.getCount(parent, parentVal) > 0) {
			delta++;
			active[parent] = true;

			if (empiricalMEst_) {
				for (CatValue y = 0; y < noClasses_; y++) {
					spodeProbs[parent][y] = empiricalMEstimate(xyCount[parent][y],
									totalCount,
									xxyDist_.xyCounts.p(y)
											* xxyDist_.xyCounts.p(
													parent, parentVal))
							* scaleFactor;
				}
			} else {
				for (CatValue y = 0; y < noClasses_; y++) {
					spodeProbs[parent][y] = mEstimate(xyCount[parent][y], totalCount,
									noClasses_
											* xxyDist_.getNoValues(
													parent))
							* scaleFactor;
				if (verbosity >= 5&&parent==1) {

					printf("%u,%u,%f\n", parent, y,
							spodeProbs[parent][y]);
				}
				}

			}
		} else if (verbosity >= 5)
			printf("%d\n", parent);
	}

	if (delta == 0) {
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {

		if(activeParentChild_[x1]==false)
			continue;

		//discard the attribute that is in generalization set

			const CatValue x1Val = inst.getCatVal(x1);
			const unsigned int noX1Vals = xxyDist_.getNoValues(x1);
			const bool x1Active = active[x1];

			constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
					noClasses_);

			//calculate only for empricial2
			const InstanceCount x1Count = xxyDist_.xyCounts.getCount(x1,
						x1Val);

			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				if(activeParentChild_[x2]==false)
					continue;

					const bool x2Active = active[x2];

					if ((x1Active&&activeChild_[x2]==true&&activeParent_[x1]==true) || (x2Active&&activeChild_[x1]==true&&activeParent_[x2]==true) ) {
						CatValue x2Val = inst.getCatVal(x2);
						const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

							//calculate only for empricial2
							InstanceCount x1x2Count = xySubDist.getCount(
									x2, x2Val, 0);
							for (CatValue y = 1; y < noClasses_; y++) {
								x1x2Count += xySubDist.getCount(x2, x2Val, y);
							}
							const InstanceCount x2Count =
									xxyDist_.xyCounts.getCount(x2, x2Val);

							const double pX2gX1=empiricalMEstimate(x1x2Count,x1Count, xxyDist_.xyCounts.p(x2, x2Val));
							const double pX1gX2=empiricalMEstimate(x1x2Count,x2Count, xxyDist_.xyCounts.p(x1, x1Val));

						for (CatValue y = 0; y < noClasses_; y++) {
							const InstanceCount x1x2yCount = xySubDist.getCount(
									x2, x2Val, y);

							if (x1Active&& activeChild_[x2]==true&&activeParent_[x1]==true) {
								if (empiricalMEst_) {

									spodeProbs[x1][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x1][y],
											xxyDist_.xyCounts.p(x2, x2Val));
								} else if (empiricalMEst2_) {
									//double probX2OnX1=mEstimate();
									spodeProbs[x1][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x1][y],
											pX2gX1);
								} else {
									spodeProbs[x1][y] *= mEstimate(x1x2yCount,
											xyCount[x1][y], noX2Vals);
								}
							}
							if (x2Active&& activeChild_[x1]==true&&activeParent_[x2]==true ) {
								if (empiricalMEst_) {
									spodeProbs[x2][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x2][y],
											xxyDist_.xyCounts.p(x1, x1Val));
								} else if (empiricalMEst2_) {
									//double probX2OnX1=mEstimate();
									spodeProbs[x2][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x2][y],
											pX1gX2);
								} else {
									spodeProbs[x2][y] *= mEstimate(x1x2yCount,
											xyCount[x2][y], noX1Vals);
								}

					}
				}
			}
		}
	}

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		if(activeParent_[parent]==false)
			continue;

		if(activeParentChild_[parent]==false)
			continue;

		if (active[parent]) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[parent][y];
				if(count_==1)
				printf("%f,",spodeProbs[parent][y]);
			}
			if(count_==1)
				printf("\n");
		}
	}
	if(count_==1)
		print(classDist);
	normalise(classDist);

}


void aodeBSE::nbClassify(const instance &inst, std::vector<double> &classDist,
		xyDist &xyDist_) {

	for (CatValue y = 0; y < noClasses_; y++) {
		double p = xyDist_.p(y) * (std::numeric_limits<double>::max() / 2.0);
		// scale up by maximum possible factor to reduce risk of numeric underflow

		for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
			p *= xyDist_.p(a, inst.getCatVal(a), y);
		}

		assert(p >= 0.0);
		classDist[y] = p;
	}
	normalise(classDist);
}

