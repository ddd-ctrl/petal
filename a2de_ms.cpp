/*
 * a2de3.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#include "a2de_ms.h"
#include "assert.h"
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStream.h"
#include "learnerRegistry.h"
//this is a two passes learning of a2de.
//

static LearnerRegistrar registrar("a2de_ms", constructor<a2de_ms>);

a2de_ms::a2de_ms(char* const *& argv, char* const * end) :
		pass_(1), weight(1) {
	name_ = "a2de3";

	// TODO Auto-generated constructor stub
	weighted = false;
	minCount = 100;
	subsumptionResolution = false;


	avg_=false;
	sum_=false;

	factored_=false;
	mi_ = false;
	su_ = false;
	acmi_ = false;
	directRank_=false;
	random_=false;


	empiricalMEst_ = false;
	empiricalMEst2_ = false;

	factor_ = 1;  //default value
//
// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		} else if (streq(argv[0] + 1, "empirical")) {
			empiricalMEst_ = true;
		} else if (streq(argv[0] + 1, "empirical2")) {
			empiricalMEst2_ = true;
		} else if (streq(argv[0] + 1, "w")) {
			weighted = true;
		} else if (streq(argv[0] + 1, "sub")) {
			subsumptionResolution = true;
		} else if (argv[0][1] == 'c') {
			getUIntFromStr(argv[0] + 2, minCount, "c");

		} else if (streq(argv[0] + 1, "mi")) {
			mi_ = true;
		} else if (streq(argv[0] + 1, "su")) {
			su_ = true;
		} else if (streq(argv[0] + 1, "acmi")) {
			acmi_ = true;
		} else if (streq(argv[0] + 1, "sum")) {
			sum_ = true;
		} else if (streq(argv[0] + 1, "avg")) {
			avg_ = true;
		} else if (streq(argv[0] + 1, "rank")) {
			directRank_ = true;
		} else if (streq(argv[0] + 1, "random")) {
			random_ = true;
		} else if (argv[0][1] == 'f') {
			unsigned int factor;
			factored_=true;
			getUIntFromStr(argv[0] + 2, factor, "f");
			factor_ = factor / 10.0;
			while (factor_ >= 1)
				factor_ /= 10;
		} else {
			error("a2de3 does not support argument %s\n", argv[0]);
			break;
		}
		name_ += *argv;
		++argv;
	}

	if(sum_==false&& avg_==false)
		sum_=true;

	trainingIsFinished_ = false;
}

a2de_ms::~a2de_ms() {
	// TODO Auto-generated destructor stub
}

void a2de_ms::reset(InstanceStream &is) {
	xxyDist_.reset(is);

	trainingIsFinished_ = false;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();

	order_.clear();

	weight = crosstab<float>(noCatAtts_);

	weightaode.assign(noCatAtts_, 1);

	//initialise the weight for non-weighting

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			weight[x1][x2] = 1;
			weight[x2][x1] = 1;
		}
	}

	active_.assign(noCatAtts_, true);
	instanceStream_ = &is;
	pass_ = 1;
	count = 0;

}

void a2de_ms::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a2de_ms::initialisePass() {

}

/// true iff no more passes are required. updated by finalisePass()
bool a2de_ms::trainingIsFinished() {
	return pass_ > 2;
}

void a2de_ms::train(const instance &inst) {

	if (pass_ == 1)
		xxyDist_.update(inst);
	else {
		assert(pass_ == 2);
		xxxyDist_.update(inst);
	}
}

// creates a comparator for two attributes based on their
//relative value with the class,such as mutual information, symmetrical uncertainty

class valCmpClass {
public:
	valCmpClass(std::vector<float> *s) {
		val = s;
	}

	bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
		return (*val)[a] > (*val)[b];
	}

private:
	std::vector<float> *val;
};

void a2de_ms::finalisePass() {
	if (pass_ == 1) {

		if (weighted) {

			//compute the mutual information as weight for aode
			getMutualInformation(xxyDist_.xyCounts, weightaode);
			//normalise to prevent overflow
			normalise(weightaode);

			if (verbosity >= 3) {
				printf("mutual information as weight for aode:\n");
				print(weightaode);
				printf("\n");
			}

			//compute the mutual information between pair attributes and class as weight for a2de
			getPairMutualInf(xxyDist_,weight);
			normalise(weight);

			if(verbosity>=3)
			{
				printf("pair mutual information as weight for a2de:\n");
				for(unsigned int i=0;i<noCatAtts_;i++)
				{
					print(weight[i]);
					printf("\n");
				}
			}
		}

		if (mi_ == true || su_ == true || acmi_ == true ||random_ == true
				|| directRank_ == true) {

			if (directRank_ == true) {
				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> mi;
				crosstab<float> cmiac(noCatAtts_);
				getAttClassCondMutualInf(xxyDist_, cmiac);
				getMutualInformation(xxyDist_.xyCounts, mi);

				if(verbosity>=3)
				{
					printf("The mutual information:\n");
					print(mi);
					printf("\n");
				}

				if(verbosity>=3)
				{
					printf("Conditional mutual information:\n");
					for(unsigned int i=0;i<noCatAtts_;i++)
					{
						print(cmiac[i]);
						printf("\n");
					}
				}
				std::vector<bool> seletedAttributes(noCatAtts_,false);
				std::vector<float> minimalCMI(noCatAtts_,0);
				float maxCMI;
				unsigned int maxIndex;
				bool maxFirst=true;

				unsigned int first=indexOfMaxVal(mi);
				order_.push_back(first);
				seletedAttributes[first]=true;

				while(order_.size()<noCatAtts_)
				{
					maxFirst=true;
					for(CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						if(seletedAttributes[a]==false)
						{
							minimalCMI[a]=cmiac[a][order_[0]];
							for (CategoricalAttribute i = 1; i < order_.size(); i++) {
								if( cmiac[a][order_[i]]< minimalCMI[a] )
								{
									minimalCMI[a] =cmiac[a][order_[i]];
								}
							}
							if(maxFirst==true)
							{
								maxFirst=false;
								maxCMI=minimalCMI[a];
								maxIndex=a;
							}else if(minimalCMI[a]>maxCMI)
							{
								maxCMI=minimalCMI[a];
								maxIndex=a;
							}

						}
					}
					order_.push_back(maxIndex);
					seletedAttributes[maxIndex]=true;
				}

				if (verbosity >= 2) {
					const char * sep = "";
					printf("The order of attributes:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						printf("%s%d",sep, order_[a] );
						sep = ", ";
					}
					printf("\n");
				}

			}else {
				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> measure;
				crosstab<float> accmi(noCatAtts_);
				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					order_.push_back(a);
				}
				if (mi_ == true && acmi_ == true) {
					if (sum_ == true) {
						getAttClassCondMutualInf(xxyDist_, accmi);
						getMutualInformation(xxyDist_.xyCounts, measure);

						for (CategoricalAttribute x1 = 1; x1 < noCatAtts_;
								x1++) {
							for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

								measure[x1] += accmi[x1][x2];
								measure[x2] += accmi[x2][x1];
								if (verbosity >= 5) {
									if (x1 == 2)
										printf("%u,", x2);
									if (x2 == 2)
										printf("%u,", x1);
								}
							}
						}

					} else if (avg_ == true) {
						getAttClassCondMutualInf(xxyDist_, accmi);
						measure.assign(noCatAtts_, 0.0);
						for (CategoricalAttribute x1 = 1; x1 < noCatAtts_;
								x1++) {
							for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

								measure[x1] += accmi[x1][x2];
								measure[x2] += accmi[x2][x1];
								if (verbosity >= 5) {
									if (x1 == 2)
										printf("%u,", x2);
									if (x2 == 2)
										printf("%u,", x1);
								}
							}
						}
						std::vector<float> mi;
						getMutualInformation(xxyDist_.xyCounts, mi);

						for (CategoricalAttribute x1 = 0; x1 < noCatAtts_;
								x1++) {
							measure[x1] = measure[x1] / (noCatAtts_ - 1)
									+ mi[x1];
						}
					}
				} else if (mi_ == true)
					getMutualInformation(xxyDist_.xyCounts, measure);
				else if (su_ == true)
					getSymmetricalUncert(xxyDist_.xyCounts, measure);
				else if (acmi_ == true) {
					getAttClassCondMutualInf(xxyDist_, accmi);
					measure.assign(noCatAtts_, 0.0);
					for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
						for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

							measure[x1] += accmi[x1][x2];
							measure[x2] += accmi[x2][x1];
							if (verbosity >= 5) {
								if (x1 == 2)
									printf("%u,", x2);
								if (x2 == 2)
									printf("%u,", x1);
							}
						}
					}
				}

				if (verbosity >= 3) {
					if (mi_ == true && acmi_ == true) {
						if (sum_ == true)
							printf(
									"Selecting according to mutual information and sum acmi:\n");
						else if (avg_ == true)
							printf(
									"Selecting according to mutual information and avg acmi:\n");
					} else if (mi_ == true)
						printf("Selecting according to mutual information:\n");
					else if (su_ == true)
						printf(
								"Selecting according to symmetrical uncertainty:\n");
					else if (acmi_ == true)
						printf(
								"Selecting according to attribute and class conditional mutual information:\n");

					print(measure);
					printf("\n");
				}

				//rank all the attributes according to the above measure
				if (!order_.empty()) {

					if(random_==true)
					{
						std::random_shuffle(order_.begin(), order_.end());

						if (verbosity >= 2) {
							printf("The random order of attributes:\n");
							const char *s="";
							for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
								printf("%s%d", s,order_[a]);
								s=",";
							}
							printf("\n");
						}
					}else{
						valCmpClass cmp(&measure);
						std::sort(order_.begin(), order_.end(), cmp);

						if (verbosity >= 3) {
							printf("The order of attributes by the measure:\n");
							for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
								printf("%d:\t%f\n", order_[a], measure[order_[a]]);
							}
							printf("\n");
						}
					}
				}
			}

			//compute the number of selected attributes

			if (factored_ == true)
			     noSelectedCatAtts_ = static_cast<unsigned int>(noCatAtts_ * factor_); ///< the number of selected attributes
			else{
				unsigned int v;
				unsigned int sumSelected;
				unsigned int i;

				v = instanceStream_->getNoValues(0);
				for (i = 1; i < noCatAtts_; i++)
					v += instanceStream_->getNoValues(i);
				v = v / noCatAtts_;

				/// memory of selective A2DE equal to memory of AODE
				/// k *[c(s,3)+c(s,2)*(a-s)]*v'^3=k*c(a,2)*v^2
				///  where k: number of classes
				///        a: number of attributes
				///        v: average number of values for each attribute
				///		   v':average number of values for selected attributes
				///        s: number of selected attributes

				/// simplification:  s*(s-1)(3a-2s-2)*v'^3=3a(a-1)*v^2
				sumSelected = instanceStream_->getNoValues(order_[0]);
				for (i = 1; i < noCatAtts_;	i++) {

					if(i*(i-1)*(3*noCatAtts_-2*i-2)*pow(static_cast<double>(sumSelected/i),3)>=3*noCatAtts_*(noCatAtts_-1)*pow(static_cast<double>(v),2))
						break;
					sumSelected+=instanceStream_->getNoValues(order_[i]);
				}
				noSelectedCatAtts_=i;

				if(verbosity>=3){
					printf("The number of attributes and selected attributes: %u,%u\n",
									noCatAtts_, noSelectedCatAtts_);
					printf("The average number of values for all attributes: %u\n",v);
				}
			}

			//rank the selected attributes based on attributes number
			if (!order_.empty()) {
				//order by attribute number for selected attributes
				std::sort(order_.begin(), order_.begin() + noSelectedCatAtts_);

				//set the attribute selected or unselected for aode
				for (CategoricalAttribute a = noSelectedCatAtts_;
						a < noCatAtts_; a++) {
					active_[order_[a]] = false;
				}

				if (verbosity >= 2) {
					const char * sep = "";
					if (factored_ == true)
						printf("The attributes specified by the user:\n");
					else
						printf("The attributes being selected according to the memory:\n");

					for (CategoricalAttribute a = 0; a < noSelectedCatAtts_;
							a++) {
						printf("%s%d", sep, order_[a]);
						sep = ", ";
					}
					printf("\n");
				}
			}
		}

		noUnSelectedCatAtts_ = noCatAtts_ - noSelectedCatAtts_;
		xxxyDist_.setNoSelectedCatAtts(noSelectedCatAtts_);
		xxxyDist_.setOrder(order_);
		xxxyDist_.reset(*instanceStream_);
	}
	pass_++;
}

void a2de_ms::classify(const instance &inst, std::vector<double> &classDist) {
	if (verbosity >= 3)
		count++;

	if (verbosity == 4 ) {
		printf("current instance:\n");
		for (CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++) {
			printf("%u:%u\n", x1, inst.getCatVal(x1));
		}
	}

	generalizationSet.assign(noCatAtts_, false);

	xxyDist * xxydist = &xxxyDist_.xxyCounts;
	xyDist * xydist = &xxxyDist_.xxyCounts.xyCounts;

	const InstanceCount totalCount = xydist->count;

	//compute the generalisation set and substitution set for
	//lazy subsumption resolution
	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					InstanceCount countOfxixj = xxydist->getCount(i,
							inst.getCatVal(i), j, inst.getCatVal(j));
					InstanceCount countOfxj = xydist->getCount(j,
							inst.getCatVal(j));
					InstanceCount countOfxi = xydist->getCount(i,
							inst.getCatVal(i));

					if (countOfxj == countOfxixj && countOfxj >= minCount) {
						//xi is a generalisation or substitution of xj
						//once one xj has been found for xi, stop for rest j
						generalizationSet[i] = true;
						break;
					} else if (countOfxi == countOfxixj
							&& countOfxi >= minCount) {
						//xj is a generalisation of xi
						generalizationSet[j] = true;
					}
				}
			}
		}

		if (verbosity >= 4) {

			for (CategoricalAttribute i = 0; i < noCatAtts_; i++)
				if (!generalizationSet[i])
					printf("%d\t", i);
			printf("\n");
		}
	}


	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	CatValue delta = 0;

////	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noSelectedCatAtts_) * (noSelectedCatAtts_ - 1) / 2.0);


//	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<double> > > spodeProbs;
	spodeProbs.resize(noSelectedCatAtts_);

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
		spodeProbs[father].resize(father);
		for (CatValue mother = 0; mother < father; mother++)
			spodeProbs[father][mother].assign(noClasses_, 0);
	}

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {

		//selecct attribute for subsumption resolution
		if (generalizationSet[order_[father]])
			continue;
		const CatValue fatherVal = inst.getCatVal(order_[father]);

		for (CatValue mother = 0; mother < father; mother++) {

			//selecct attribute for subsumption resolution
			if (generalizationSet[order_[mother]])
				continue;
			const CatValue motherVal = inst.getCatVal(order_[mother]);

			CatValue parent = 0;
			for (CatValue y = 0; y < noClasses_; y++) {
				parent += xxydist->getCount(order_[father], fatherVal,
						order_[mother], motherVal, y);
			}

			if (parent > 0) {
				delta++;

				for (CatValue y = 0; y < noClasses_; y++) {
					spodeProbs[father][mother][y] = weight[order_[father]][order_[mother]]
							* scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

					InstanceCount parentYCount = xxydist->getCount(
							order_[father], fatherVal, order_[mother],
							motherVal, y);
					if (empiricalMEst_) {

						spodeProbs[father][mother][y] *= empiricalMEstimate(
								parentYCount, totalCount,
								xydist->p(y)
										* xydist->p(order_[father], fatherVal)
										* xydist->p(order_[mother], motherVal));

					} else {
						double temp = mEstimate(parentYCount, totalCount,
								noClasses_ * xxxyDist_.getNoValues(father)
										* xxxyDist_.getNoValues(mother));
						spodeProbs[father][mother][y] *= temp;
					}

					if (verbosity == 4)
						printf("%f,", spodeProbs[father][mother][y]);
				}

			}
		}

	}

	if (verbosity == 3) {
		for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
			for (CatValue mother = 0; mother < father; mother++) {
				printf("initial spode probs for %u,%u:\n", order_[father],
						order_[mother]);
				print(spodeProbs[father][mother]);
				printf("\n");
			}
		}
		printf("initial spode probs ending.<<<<<\n");
	}

	if (delta == 0) {
		if (verbosity == 3)
			printf("aode is called for %u instance\n", count);

		aodeClassify(inst, classDist, *xxydist);

		return;
	}

	if (verbosity == 3 ) {
		printf("print every prob for each parents:\n");
	}

	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {

		//select the attribute according to centain measure
		//selecct attribute for subsumption resolution
		if (generalizationSet[order_[father]])
			continue;
		const CatValue fatherVal = inst.getCatVal(order_[father]);
//
		XXYSubDist xxySubDist(xxxyDist_.getXXYSubDist(father, fatherVal),
				noClasses_);

		XXYSubDist xxySubDistRest(
				xxxyDist_.getXXYSubDistRest(father, fatherVal), noClasses_);

		XYSubDist xySubDistFather(
				xxydist->getXYSubDist(order_[father], fatherVal), noClasses_);

		for (CatValue mother = 0; mother < father; mother++) {
			//select the attribute according to centain measure

			//selecct attribute for subsumption resolution
			if (generalizationSet[order_[mother]])
				continue;
			const CatValue motherVal = inst.getCatVal(order_[mother]);

			XYSubDist xySubDist(xxySubDist.getXYSubDist(mother, motherVal),
					noClasses_);

			XYSubDist xySubDistRest(
					xxySubDistRest.getXYSubDist(mother, motherVal,
							noUnSelectedCatAtts_), noClasses_);

			XYSubDist xySubDistMother(
					xxydist->getXYSubDist(order_[mother], motherVal),
					noClasses_);

			// as we store the instance count completely for the third attributes
			// we here can set child to all possible values
			for (CatValue child = 0; child < mother; child++) {

				if (!generalizationSet[order_[child]]) {

					const CatValue childVal = inst.getCatVal(order_[child]);

					if (child != father && child != mother) {

						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDist.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistFather.getCount(order_[mother],
											motherVal, y);
							InstanceCount fatherYChildCount =
									xySubDistFather.getCount(order_[child],
											childVal, y);
							InstanceCount motherYChildCount =
									xySubDistMother.getCount(order_[child],
											childVal, y);

							double temp;
							if (empiricalMEst_) {
								if (xxydist->getCount(order_[father], fatherVal,
										order_[mother], motherVal) > 0) {

									temp = empiricalMEstimate(parentChildYCount,
											parentYCount,
											xydist->p(order_[child], childVal));
									spodeProbs[father][mother][y] *= temp;
								}
								if (xxydist->getCount(order_[father], fatherVal,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											fatherYChildCount,
											xydist->p(mother, motherVal));
									spodeProbs[father][child][y] *= temp;
								}

								if (xxydist->getCount(order_[mother], motherVal,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											motherYChildCount,
											xydist->p(father, fatherVal));
									spodeProbs[mother][child][y] *= temp;

								}

							} else {

								if (xxydist->getCount(order_[father], fatherVal,
										order_[mother], motherVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parentYCount,
											xxxyDist_.getNoValues(child));
									spodeProbs[father][mother][y] *= temp;

									if (verbosity == 3 && y == 0) {
										printf("%u,%u,%u,%u,%f\n",
												order_[father], order_[mother],
												order_[child], y, temp);
									}

								}
								if (xxydist->getCount(order_[father], fatherVal,
										order_[child], childVal) > 0) {

									temp = mEstimate(parentChildYCount,
											fatherYChildCount,
											xxxyDist_.getNoValues(mother));
									spodeProbs[father][child][y] *= temp;

									if (verbosity == 3 && y == 0) {
										printf("%u,%u,%u,%u,%f\n",
												order_[father], order_[child],
												order_[child], y, temp);
									}
								}
								if (xxydist->getCount(order_[mother], motherVal,
										order_[child], childVal) > 0) {

									temp = mEstimate(parentChildYCount,
											motherYChildCount,
											xxxyDist_.getNoValues(father));
									spodeProbs[mother][child][y] *= temp;

									if (verbosity == 3 && y == 0) {
										printf("%u,%u,%u,%u,%f---\n",
												order_[mother], order_[child],
												order_[child], y, temp);
									}
								}
								if (verbosity == 3 && y == 0) {
									if (order_[father] == 3
											&& order_[mother] == 1
											&& order_[child] == 5) {
										printf("%u,%u,%u\n", parentChildYCount,
												parentYCount,
												xxxyDist_.getNoValues(child));
									}

								}

								if (verbosity == 3 && y == 0) {
									if (order_[father] == 3
											&& order_[child] == 1
											&& order_[mother] == 5) {
										printf("%u,%u,%u\n", parentChildYCount,
												fatherYChildCount,
												xxxyDist_.getNoValues(mother));
									}

								}

								if (verbosity == 3 && y == 0) {
									if (order_[mother] == 3
											&& order_[child] == 1
											&& order_[father] == 5) {
										printf("%u,%u,%u---\n",
												parentChildYCount,
												motherYChildCount,
												xxxyDist_.getNoValues(father));
									}

								}

							}

						}
					}
				}
			}

			//check if the parent is qualified
			if (xxydist->getCount(order_[father], fatherVal, order_[mother],
					motherVal) == 0)
				continue;

			for (CatValue child = 0; child < noUnSelectedCatAtts_; child++) {

				if (!generalizationSet[order_[child + noSelectedCatAtts_]]) {

					const CatValue childVal = inst.getCatVal(
							order_[child + noSelectedCatAtts_]);

					//if (child != father && child != mother) {
					//if (order_[child] != order_[father] && order_[child] != order_[mother]) {

					for (CatValue y = 0; y < noClasses_; y++) {

						InstanceCount parentChildYCount =
								xySubDistRest.getCount(child, childVal, y);

						InstanceCount parentYCount = xySubDistFather.getCount(
								order_[mother], motherVal, y);

						double temp;
						if (empiricalMEst_) {

							temp = empiricalMEstimate(parentChildYCount,
									parentYCount,
									xydist->p(order_[child], childVal));
							spodeProbs[father][mother][y] *= temp;

						} else {

							temp = mEstimate(parentChildYCount, parentYCount,
									xxxyDist_.getNoValues(
											child + noSelectedCatAtts_));
							spodeProbs[father][mother][y] *= temp;

							if (verbosity == 3 && y == 0) {
								if (order_[father] == 3 && order_[mother] == 1
										&& order_[child + noSelectedCatAtts_]
												== 5) {
									printf("%u,%u,%u\n", parentChildYCount,
											parentYCount,
											xxxyDist_.getNoValues(
													child
															+ noSelectedCatAtts_));
								}

							}
							if (verbosity == 3&& y == 0) {
								printf("%u,%u,%u,%u,%f\n", order_[father],
										order_[mother],
										order_[child + noSelectedCatAtts_], y,
										temp);
							}
						}
					}
				}
			}
		}
	}
	for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
		for (CatValue mother = 0; mother < father; mother++)
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[father][mother][y];
			}
	}

	if (verbosity == 3) {
		printf("distribute of instance %u for each parent.\n", count);
		for (CatValue father = 1; father < noSelectedCatAtts_; father++) {
			for (CatValue mother = 0; mother < father; mother++) {
				printf("%u,%u:", order_[father], order_[mother]);
				print(spodeProbs[father][mother]);
				printf("\n");
			}
		}
		printf("spode probs ending.<<<<<\n");
	}
	normalise(classDist);

}

void a2de_ms::aodeClassify(const instance &inst, std::vector<double> &classDist,
		xxyDist & xxyDist_) {


// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	CatValue delta = 0;

	const InstanceCount totalCount = xxyDist_.xyCounts.count;

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {

		//discard the attribute that is not active or in generalization set
		if (!generalizationSet[parent]) {
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
							spodeProbs[parent][y] = weightaode[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parent][y] = weightaode[parent]
									* mEstimate(xyCount[parent][y], totalCount,
											noClasses_
													* xxyDist_.getNoValues(
															parent))
									* scaleFactor;
						}
					}
				} else if (verbosity >= 5)
					printf("%d\n", parent);

		}
	}

	if (delta == 0) {
		//count++;
		if (verbosity == 3)
			printf("nb is called for %u instance\n", count);
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {

		//discard the attribute that is in generalization set
		if (!generalizationSet[x1]) {
			const CatValue x1Val = inst.getCatVal(x1);
			const unsigned int noX1Vals = xxyDist_.getNoValues(x1);
			const bool x1Active = active[x1];

			constXYSubDist xySubDist(xxyDist_.getXYSubDist(x1, x1Val),
					noClasses_);

			//calculate only for empricial2
			const InstanceCount x1Count = xxyDist_.xyCounts.getCount(x1, x1Val);

			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				//	printf("c:%d\n", x2);
				if (!generalizationSet[x2]) {
					const bool x2Active = active[x2];

					if (x1Active || x2Active) {
						CatValue x2Val = inst.getCatVal(x2);
						const unsigned int noX2Vals = xxyDist_.getNoValues(x2);

						//calculate only for empricial2
						InstanceCount x1x2Count = xySubDist.getCount(x2, x2Val,
								0);
						for (CatValue y = 1; y < noClasses_; y++) {
							x1x2Count += xySubDist.getCount(x2, x2Val, y);
						}
						const InstanceCount x2Count =
								xxyDist_.xyCounts.getCount(x2, x2Val);

						const double pX2gX1 = empiricalMEstimate(x1x2Count,
								x1Count, xxyDist_.xyCounts.p(x2, x2Val));
						const double pX1gX2 = empiricalMEstimate(x1x2Count,
								x2Count, xxyDist_.xyCounts.p(x1, x1Val));

						for (CatValue y = 0; y < noClasses_; y++) {
							const InstanceCount x1x2yCount = xySubDist.getCount(
									x2, x2Val, y);

							if (x1Active) {
								if (empiricalMEst_) {

									spodeProbs[x1][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x1][y],
											xxyDist_.xyCounts.p(x2, x2Val));
								} else if (empiricalMEst2_) {
									//double probX2OnX1=mEstimate();
									spodeProbs[x1][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x1][y], pX2gX1);
								} else {
									spodeProbs[x1][y] *= mEstimate(x1x2yCount,
											xyCount[x1][y], noX2Vals);
								}
							}
							if (x2Active) {
								if (empiricalMEst_) {
									spodeProbs[x2][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x2][y],
											xxyDist_.xyCounts.p(x1, x1Val));
								} else if (empiricalMEst2_) {
									//double probX2OnX1=mEstimate();
									spodeProbs[x1][y] *= empiricalMEstimate(
											x1x2yCount, xyCount[x1][y], pX1gX2);
								} else {
									spodeProbs[x2][y] *= mEstimate(x1x2yCount,
											xyCount[x2][y], noX1Vals);
								}
							}
						}
					}
				}
			}
		}
	}

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {
		if (active[parent]) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[parent][y];
			}
		}
	}

	normalise(classDist);

	//count++;
	if (verbosity == 3 && count == 1) {
		printf("the class distribution is :\n");
		print(classDist);
		printf("\n");
	}
}
void a2de_ms::nbClassify(const instance &inst, std::vector<double> &classDist,
		xyDist & xyDist_) {

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

