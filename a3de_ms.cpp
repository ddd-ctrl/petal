/*
 * a3de3.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#include "a3de_ms.h"
#include "assert.h"
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"

#include "globals.h"
#include "instanceStream.h"
#include "learnerRegistry.h"

static LearnerRegistrar registrar("a3de_ms",constructor<a3de_ms>);

a3de_ms::a3de_ms(char* const *& argv, char* const * end) :
		weight_a2de(1) {
	name_ = "A3DE3";

	// TODO Auto-generated constructor stub
	count_=0;

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

	oneSelective_ = false;
	twoSelective_ = false;

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
		} else if (streq(argv[0] + 1, "acmi")) {
			acmi_ = true;
		} else if (streq(argv[0] + 1, "mi")) {
			mi_ = true;
		} else if (streq(argv[0] + 1, "su")) {
			su_ = true;
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
		} else if (streq(argv[0] + 1, "one")) {
			oneSelective_ = true;
		} else if (streq(argv[0] + 1, "two")) {
			twoSelective_ = true;
		} else {
			error("a3de3 does not support argument %s\n", argv[0]);
			break;
		}
		name_ += *argv;
		++argv;
	}

	if (sum_ == false && avg_ == false)
		sum_ = true;
}

a3de_ms::~a3de_ms() {
	// TODO Auto-generated destructor stub
}

void a3de_ms::reset(InstanceStream &is) {

	xxxyDist_.reset(is);
	instanceStream_ = &is;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();


	order_.clear();

	weight_aode.assign(noCatAtts_, 1);
	weight_a2de = crosstab<float>(noCatAtts_);
	weight = crosstab3D<float>(noCatAtts_);

	//initialise the weight for non-weighting
	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

			weight_a2de[x1][x2] = 1;
			weight_a2de[x2][x1] = 1;
		}
	}

	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
		for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				weight[x1][x2][x3] = 1;
				weight[x2][x1][x3] = 1;
				weight[x1][x3][x2] = 1;
				weight[x2][x3][x1] = 1;
				weight[x3][x1][x2] = 1;
				weight[x3][x2][x1] = 1;
			}
		}
	}

	active_.assign(noCatAtts_, true);
	pass_ = 1;
	instanceStream_ = &is;
}

void a3de_ms::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a3de_ms::initialisePass() {

}

/// true iff no more passes are required. updated by finalisePass()
bool a3de_ms::trainingIsFinished() {
	return pass_ > 2;
}

void a3de_ms::train(const instance &inst) {

	if (pass_ == 1)
		xxxyDist_.update(inst);
	else {
		assert(pass_ == 2);
		xxxxyDist_.update(inst);
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

void a3de_ms::finalisePass() {

	if (pass_ == 1) {

		xxyDist &xxyDist_=xxxyDist_.xxyCounts;

		if (weighted) {
			printf("A3DE will be weighted by the mutual information between triple attributes and class\n");
			//compute the mutual information as weight for aode
			getMutualInformation(xxyDist_.xyCounts, weight_aode);
			//normalise to prevent overflow
			normalise(weight_aode);

			if (verbosity >= 3) {
				printf("Mutual information as weight for aode:\n");
				print(weight_aode);
				printf("\n");
			}

			//compute the mutual information between pair attributes and class as weight for a2de
			getPairMutualInf(xxyDist_,weight_a2de);
			normalise(weight_a2de);

			if(verbosity>=3)
			{
				printf("Pair mutual information as weight for a2de:\n");
				for(unsigned int i=0;i<noCatAtts_;i++)
				{
					print(weight_a2de[i]);
					printf("\n");
				}
			}

			//compute the mutual information between triple attributes and class as weight for a3de
			getTripleMutualInf(xxxyDist_,weight);
			normalise(weight);

			if(verbosity>=3)
			{
				printf("Triple mutual information as weight for a3de:\n");
				for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
					for (CategoricalAttribute x2 = 0; x2 < noCatAtts_; x2++) {
						printf("%u,%u\t",x1,x2);
						print(weight[x1][x2]);
						printf("\n");
					}
				}
			}

		}



		if (mi_ == true || su_ == true || acmi_ == true||random_ == true || directRank_ == true) {

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

				if (verbosity >= 3) {
					const char * sep = "";
					printf("The order of attributes:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						printf("%s%d",sep, order_[a] );
						sep = ", ";
					}
					printf("\n");
				}
			}else
			{
				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> measure;
				crosstab<float> acmi(noCatAtts_);

				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					order_.push_back(a);
				}
				if (mi_ == true && acmi_ == true) {
					if (sum_ == true) {
						getAttClassCondMutualInf(xxyDist_, acmi);
						getMutualInformation(xxyDist_.xyCounts, measure);

						for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
							for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

								measure[x1] += acmi[x1][x2];
								measure[x2] += acmi[x2][x1];
								if (verbosity >= 5) {
									if (x1 == 2)
										printf("%u,", x2);
									if (x2 == 2)
										printf("%u,", x1);
								}
							}
						}

					} else if (avg_ == true) {
						getAttClassCondMutualInf(xxyDist_, acmi);
						measure.assign(noCatAtts_, 0.0);
						for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
							for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

								measure[x1] += acmi[x1][x2];
								measure[x2] += acmi[x2][x1];
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

						for (CategoricalAttribute x1 = 0; x1 < noCatAtts_; x1++) {
							measure[x1] = measure[x1] / (noCatAtts_ - 1) + mi[x1];
						}
					}
				} else if (mi_ == true)
					getMutualInformation(xxyDist_.xyCounts, measure);
				else if (su_ == true)
					getSymmetricalUncert(xxyDist_.xyCounts, measure);
				else if (acmi_ == true) {
					getAttClassCondMutualInf(xxyDist_, acmi);
					measure.assign(noCatAtts_, 0.0);

					for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
						for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

							measure[x1] += acmi[x1][x2];
							measure[x2] += acmi[x2][x1];
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
						printf("Selecting according to symmetrical uncertainty:\n");
					else if (acmi_ == true)
						printf(
								"Selecting according to attribute and class conditional mutual information:\n");

					print(measure);
					printf("\n");
				}

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
							const char * sep = "";
							printf("The order of attributes ordered by the measure:\n");
							for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
								printf("%s%d", sep, order_[a]);
								sep = ", ";
							}
							printf("\n");
						}
					}
				}
			}


			//compute the number of selected attributes
			if (factored_ == true)
			     noSelectedCatAtts_ = static_cast<unsigned int>(noCatAtts_ * factor_); ///< the number of selected attributes
			else if (oneSelective_ == true) {

				unsigned int v;
				unsigned int sumSelected;
				unsigned int i;

				v = instanceStream_->getNoValues(0);
				for (i = 1; i < noCatAtts_; i++)
					v += instanceStream_->getNoValues(i);
				v = v / noCatAtts_;


				/// memory of selective A3DE equal to memory of AODE
				/// k *[c(s,4)+c(s,3)*(a-s)]*v'^4=k*c(a,2)*v^2
				///  where k: number of classes
				///        a: number of attributes
				///        v: average number of values for each attribute
				///		   v':average number of values for selected attributes
				///        s: number of selected attributes

				/// simplification:  s*(s-1)*(s-2)*(4a-3s-3)*v'^4=12*a*(a-1)*v^2
				sumSelected = instanceStream_->getNoValues(order_[0]);
				for (i = 1; i < noCatAtts_;	i++) {
					if(i*(i-1)*(i-2)*(4*noCatAtts_-3*i-3)*pow(static_cast<double>(sumSelected/i),4)>=12*noCatAtts_*(noCatAtts_-1)*pow(static_cast<double>(v),2))
						break;
					sumSelected+=instanceStream_->getNoValues(order_[i]);
				}
				noSelectedCatAtts_=i;
			} else  {
				assert(twoSelective_== true);

				unsigned int v;
				unsigned int sumSelected;
				unsigned int i;

				v = instanceStream_->getNoValues(0);
				for (i = 1; i < noCatAtts_; i++)
					v += instanceStream_->getNoValues(i);
				v = v / noCatAtts_;

				/// memory of selective A3DE equal to memory of A2DE
				/// k *[c(s,4)+c(s,3)*(a-s)]*v'^4=k*c(a,3)*v^3
				///  where k: number of classes
				///        a: number of attributes
				///        v: average number of values for each attribute
				///		   v':average number of values for selected attributes
				///        s: number of selected attributes

				/// simplification:  s*(s-1)*(s-2)*(4a-3s-3)*v'^4=4*a*(a-1)*(a-2)*v^3
				sumSelected = instanceStream_->getNoValues(order_[0]);
				for (i = 1; i < noCatAtts_;	i++) {

					if(i*(i-1)*(i-2)*(4*noCatAtts_-3*i-3)*pow(static_cast<double>(sumSelected/i),4)>=4*noCatAtts_*(noCatAtts_-1)*(noCatAtts_-2)*pow(static_cast<double>(v),3))
						break;
					sumSelected+=instanceStream_->getNoValues(order_[i]);
				}
				noSelectedCatAtts_=i;

		//		printf("Selecting the attributes according to the memory a2de required.\n");
		//		printf("The number of attributes and selected attributes: %u,%u\n",
		//				noCatAtts_, noSelectedCatAtts_);
		//		printf("The average number of values for all attributes: %u\n", v);

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

			if (verbosity >= 3) {
				const char * sep = "";
				if (oneSelective_ == true)
					printf(
							"The attributes being selected according to the memory of AODE:\n");
				else if (twoSelective_ == true)
										printf(
												"The attributes being selected according to the memory of A2DE:\n");
				else
					printf("The attributes specified by the user:\n");
				for (CategoricalAttribute a = 0; a < noSelectedCatAtts_;
						a++) {
					printf("%s%d", sep, order_[a]);
					sep = ", ";
				}
				printf("\n");
			}
		}

		noUnSelectedCatAtts_ = noCatAtts_ - noSelectedCatAtts_;
		xxxxyDist_.setNoSelectedCatAtts(noSelectedCatAtts_);
		xxxxyDist_.setOrder(order_);
		xxxxyDist_.reset(*instanceStream_);
	}
	pass_++;
}

void a3de_ms::classify(const instance &inst, std::vector<double> &classDist) {
	count_++;
	generalizationSet.assign(noCatAtts_, false);

	xxxyDist * xxxydist = &xxxxyDist_.xxxyCounts;
	xxyDist * xxydist = &xxxydist->xxyCounts;
	xyDist * xydist = &xxydist->xyCounts;

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

		if (verbosity >= 4&&count_==11) {
			for (CategoricalAttribute i = 0; i < noCatAtts_; i++)
				if (!generalizationSet[i])
					printf("%d\t", i);
			printf("\n");
		}
	}

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	CatValue delta = 0;

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noSelectedCatAtts_ - 1) * (noSelectedCatAtts_ - 2)
					* (noSelectedCatAtts_) / 6.0);

	//	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<std::vector<double> > > > spodeProbs;
	spodeProbs.resize(noSelectedCatAtts_);

	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {
		spodeProbs[parent1].resize(parent1);
		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			spodeProbs[parent1][parent2].resize(parent2);
			for (CatValue parent3 = 0; parent3 < parent2; parent3++)
				spodeProbs[parent1][parent2][parent3].assign(noClasses_, 0);
		}
	}

	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {

		//select attribute for subsumption resolution
		if (generalizationSet[order_[parent1]])
			continue;
		const CatValue parent1Val = inst.getCatVal(order_[parent1]);

		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {

			//select attribute for subsumption resolution
			if (generalizationSet[order_[parent2]])
				continue;
			const CatValue parent2Val = inst.getCatVal(order_[parent2]);

			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {

				//select attribute for subsumption resolution
				if (generalizationSet[order_[parent3]])
					continue;
				const CatValue parent3Val = inst.getCatVal(order_[parent3]);

				CatValue parent = 0;
				for (CatValue y = 0; y < noClasses_; y++) {
					parent += xxxydist->getCount(order_[parent1], parent1Val,
							order_[parent2], parent2Val, order_[parent3],
							parent3Val, y);
				}

				if (parent > 0) {

					delta++;
					for (CatValue y = 0; y < noClasses_; y++) {
						spodeProbs[parent1][parent2][parent3][y] =
								weight[order_[parent1]][order_[parent2]][order_[parent3]] * scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

						InstanceCount parentYCount = xxxydist->getCount(
								order_[parent1], parent1Val, order_[parent2],
								parent2Val, order_[parent3], parent3Val, y);
						if (empiricalMEst_) {

							spodeProbs[parent1][parent2][parent3][y] *=
									empiricalMEstimate(parentYCount, totalCount,
											xydist->p(y)
													* xydist->p(order_[parent1],
															parent1Val)
													* xydist->p(order_[parent2],
															parent2Val)
													* xydist->p(order_[parent3],
															parent3Val));

						} else {
							double temp = mEstimate(parentYCount, totalCount,
									noClasses_ * xxxxyDist_.getNoValues(parent1)
											* xxxxyDist_.getNoValues(parent2)
											* xxxxyDist_.getNoValues(parent3));
							spodeProbs[parent1][parent2][parent3][y] *= temp;
						}
					}

				}
			}

		}
	}

	if (delta == 0) {
		a2deClassify(inst, classDist, *xxxydist);
		return;
	}

	//deal with the selected attributes as parents and child
	for (CatValue parent1 = 3; parent1 < noSelectedCatAtts_; parent1++) {
		if (generalizationSet[order_[parent1]])
			continue;
		const CatValue parent1Val = inst.getCatVal(order_[parent1]);

		for (CatValue parent2 = 2; parent2 < parent1; parent2++) {
			if (generalizationSet[order_[parent2]])
				continue;
			const CatValue parent2Val = inst.getCatVal(order_[parent2]);

			XXYSubDist xxySubDist(
					xxxxyDist_.getXXYSubDist(parent1, parent1Val, parent2,
							parent2Val), noClasses_);

			XYSubDist xySubDistParent12(
					xxxydist->getXYSubDist(order_[parent1], parent1Val,
							order_[parent2], parent2Val), noClasses_);

			for (CatValue parent3 = 1; parent3 < parent2; parent3++) {
				if (generalizationSet[order_[parent3]])
					continue;
				const CatValue parent3Val = inst.getCatVal(order_[parent3]);

				XYSubDist xySubDist(
						xxySubDist.getXYSubDist(parent3, parent3Val),
						noClasses_);

				XYSubDist xySubDistParent23(
						xxxydist->getXYSubDist(order_[parent2], parent2Val,
								order_[parent3], parent3Val), noClasses_);
				XYSubDist xySubDistParent13(
						xxxydist->getXYSubDist(order_[parent1], parent1Val,
								order_[parent3], parent3Val), noClasses_);

				for (CatValue child = 0; child < parent3; child++) {

					if(verbosity>=4&&count_==1)
					{
						printf("%u,%u,%u>>%u\n",parent1,parent2,parent3,child);
						printf("%u,%u,%u>>%u\n",parent2,parent3,child,parent1);
						printf("%u,%u,%u>>%u\n",parent1,parent3,child,parent2);
						printf("%u,%u,%u>>%u\n",parent1,parent2,child,parent3);
					}

					if (!generalizationSet[order_[child]]) {

						const CatValue childVal = inst.getCatVal(order_[child]);

						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDist.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistParent12.getCount(order_[parent3],
											parent3Val, y);

							InstanceCount parent23YChildCount =
									xySubDistParent23.getCount(order_[child],
											childVal, y);
							InstanceCount parent13YChildCount =
									xySubDistParent13.getCount(order_[child],
											childVal, y);
							InstanceCount parent12YChildCount =
									xySubDistParent12.getCount(order_[child],
											childVal, y);
							double temp;
							if (empiricalMEst_) {

								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[parent3], parent3Val) > 0) {

									temp = empiricalMEstimate(parentChildYCount,
											parentYCount,
											xydist->p(order_[child], childVal));
									spodeProbs[parent1][parent2][parent3][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent2],
										parent2Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent23YChildCount,
											xydist->p(order_[parent1],
													parent1Val));
									spodeProbs[parent2][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent13YChildCount,
											xydist->p(order_[parent2],
													parent2Val));
									spodeProbs[parent1][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent12YChildCount,
											xydist->p(order_[parent3],
													parent3Val));
									spodeProbs[parent1][parent2][child][y] *=
											temp;
								}

							} else {

								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[parent3], parent3Val) > 0) {
									temp = mEstimate(parentChildYCount,
											parentYCount,
											xxxxyDist_.getNoValues(child));

									spodeProbs[parent1][parent2][parent3][y] *=
											temp;

								}
								if (xxxydist->getCount(order_[parent2],
										parent2Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent23YChildCount,
											xxxxyDist_.getNoValues(parent1));
									spodeProbs[parent2][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent3], parent3Val,
										order_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent13YChildCount,
											xxxxyDist_.getNoValues(parent2));
									spodeProbs[parent1][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(order_[parent1],
										parent1Val, order_[parent2], parent2Val,
										order_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent12YChildCount,
											xxxxyDist_.getNoValues(parent3));
									spodeProbs[parent1][parent2][child][y] *=
											temp;
								}

							}

						}

					}
				}
			}
		}
	}

	//deal with the unselected attributes as the child
	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {
		if (generalizationSet[order_[parent1]])
			continue;
		const CatValue parent1Val = inst.getCatVal(order_[parent1]);

		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			if (generalizationSet[order_[parent2]])
				continue;
			const CatValue parent2Val = inst.getCatVal(order_[parent2]);

			XXYSubDist xxySubDistRest(
					xxxxyDist_.getXXYSubDistRest(parent1, parent1Val, parent2,
							parent2Val), noClasses_);

			XYSubDist xySubDistParent12(
					xxxydist->getXYSubDist(order_[parent1], parent1Val,
							order_[parent2], parent2Val), noClasses_);

			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {
				if (generalizationSet[order_[parent3]])
					continue;
				const CatValue parent3Val = inst.getCatVal(order_[parent3]);

				XYSubDist xySubDistRest(
						xxySubDistRest.getXYSubDist(parent3, parent3Val,
								noUnSelectedCatAtts_), noClasses_);

				//check if the parent is qualified
				if (xxxydist->getCount(order_[parent1],
						parent1Val, order_[parent2], parent2Val,
						order_[parent3], parent3Val) == 0)
					continue;

				for (CatValue child = 0; child < noUnSelectedCatAtts_; child++) {

					if(verbosity>=4&&count_==1)
					{
						printf("%u,%u,%u>>%u\n",parent1,parent2,parent3,child + noSelectedCatAtts_);
					}

					if (!generalizationSet[order_[child + noSelectedCatAtts_]]) {

						const CatValue childVal = inst.getCatVal(
								order_[child + noSelectedCatAtts_]);

						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDistRest.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistParent12.getCount(order_[parent3],
											parent3Val, y);
							double temp;
							if (empiricalMEst_) {

								temp = empiricalMEstimate(parentChildYCount,
										parentYCount,
										xydist->p(order_[child], childVal));
								spodeProbs[parent1][parent2][parent3][y] *=
										temp;

							} else {

								temp = mEstimate(parentChildYCount, parentYCount,
										xxxxyDist_.getNoValues(
												child + noSelectedCatAtts_));
								spodeProbs[parent1][parent2][parent3][y] *=
										temp;
							}
						}
					}
				}
			}
		}
	}

	for (CatValue parent1 = 2; parent1 < noSelectedCatAtts_; parent1++) {
		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent1][parent2][parent3][y];
				}
			}
		}
	}
	normalise(classDist);
}

void a3de_ms::a2deClassify(const instance &inst, std::vector<double> &classDist,
		xxxyDist & xxxyDist_) {

	xxyDist * xxydist = &xxxyDist_.xxyCounts;
	xyDist * xydist = &xxxyDist_.xxyCounts.xyCounts;

	const InstanceCount totalCount = xydist->count;

	for (CatValue y = 0; y < noClasses_; y++)
		classDist[y] = 0;

	CatValue delta = 0;

//	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_) / 2.0);

//	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<double> > > spodeProbs;
	spodeProbs.resize(noCatAtts_);

	for (CatValue father = 1; father < noCatAtts_; father++) {
		spodeProbs[father].resize(father);
		for (CatValue mother = 0; mother < father; mother++)
			spodeProbs[father][mother].assign(noClasses_, 0);
	}

	// now the activeParent tab is no use for two passes a2de
	crosstab<bool> activeParent(noCatAtts_);

	for (CatValue father = 1; father < noCatAtts_; father++) {

		//select attribute for subsumption resolution
		if (generalizationSet[father])
			continue;
		const CatValue fatherVal = inst.getCatVal(father);

		for (CatValue mother = 0; mother < father; mother++) {
			//select attribute for subsumption resolution
			if (generalizationSet[mother])
				continue;
			const CatValue motherVal = inst.getCatVal(mother);

			CatValue parent = 0;
			for (CatValue y = 0; y < noClasses_; y++) {
				parent += xxydist->getCount(father, fatherVal, mother,
						motherVal, y);
			}

			if (parent > 0) {
				activeParent[father][mother] = true;

				delta++;

				for (CatValue y = 0; y < noClasses_; y++) {
					spodeProbs[father][mother][y] = weight_a2de[father][mother]*scaleFactor;
					// scale up by maximum possible factor to reduce risk of numeric underflow

					InstanceCount parentYCount = xxydist->getCount(father,
							fatherVal, mother, motherVal, y);
					if (empiricalMEst_) {
						spodeProbs[father][mother][y] *= empiricalMEstimate(
								parentYCount, totalCount,
								xydist->p(y) * xydist->p(father, fatherVal)
										* xydist->p(mother, motherVal));

					} else {
						double prob=mEstimate(parentYCount,
							totalCount,
							noClasses_ * xxxyDist_.getNoValues(father)
									* xxxyDist_.getNoValues(mother));
						spodeProbs[father][mother][y] *=prob ;

					}
					if (verbosity == 4)
						printf("%f,", spodeProbs[father][mother][y]);
				}

			}
		}

	}

	if (verbosity == 3 ) {
		for (CatValue father = 1; father < noCatAtts_; father++) {
			for (CatValue mother = 0; mother < father; mother++) {
				printf("initial spode probs for %u,%u:\n", father, mother);
				print(spodeProbs[father][mother]);
				printf("\n");
			}
		}
		printf("initial spode probs ending.<<<<<\n");
	}

	if (delta == 0) {
		aodeClassify(inst, classDist, *xxydist);
		return;
	}

	if (verbosity == 3 ) {
		printf("print every prob for each parents:\n");
	}
	for (CatValue father = 2; father < noCatAtts_; father++) {

		//for a2de2, selecting must be in the most inner loop
		// because we want to select only the parents, which means we should let
		//child be any value different from parent.
		//if we select here, child can only have values in this set

		//select attribute for subsumption resolution
		if (generalizationSet[father])
			continue;
		const CatValue fatherVal = inst.getCatVal(father);

		XXYSubDist xxySubDist(xxxyDist_.getXXYSubDist(father, fatherVal),
				noClasses_);
		XYSubDist xySubDistFather(xxydist->getXYSubDist(father, fatherVal),
				noClasses_);
		//	xxySubDist.getXYSubDist(mother,motherVal)

		for (CatValue mother = 1; mother < father; mother++) {
			//select the attribute according to centain measure

			//select attribute for subsumption resolution
			if (generalizationSet[mother])
				continue;
			const CatValue motherVal = inst.getCatVal(mother);

			XYSubDist xySubDist(xxySubDist.getXYSubDist(mother, motherVal),
					noClasses_);

			XYSubDist xySubDistMother(xxydist->getXYSubDist(mother, motherVal),
					noClasses_);

			for (CatValue child = 0; child < mother; child++) {

				if (!generalizationSet[child]) {
					const CatValue childVal = inst.getCatVal(child);
					if (child != father && child != mother) {

						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDist.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistFather.getCount(mother, motherVal,
											y);

							InstanceCount fatherYChildCount =
									xySubDistFather.getCount(child, childVal,
											y);

							InstanceCount motherYChildCount =
									xySubDistMother.getCount(child, childVal,
											y);

							double temp1, temp2, temp3;
							if (empiricalMEst_) {

								//perform selecting here
								// it seems complicated, but is the only choice for a2de2
								// of course, this will sacrifice a little efficiency

								if (xxydist->getCount(father, fatherVal, mother,
										motherVal) > 0) {
										temp1 = empiricalMEstimate(
												parentChildYCount, parentYCount,
												xydist->p(child, childVal));
										spodeProbs[father][mother][y] *= temp1;


								}
								if (xxydist->getCount(father, fatherVal, child,
										childVal) > 0) {
										temp2 = empiricalMEstimate(
												parentChildYCount,
												fatherYChildCount,
												xydist->p(mother, motherVal));
										spodeProbs[father][child][y] *= temp2;
								}
								if (xxydist->getCount(mother, motherVal, child,
										childVal) > 0) {
										temp3 = empiricalMEstimate(
												parentChildYCount,
												motherYChildCount,
												xydist->p(father, fatherVal));
										spodeProbs[mother][child][y] *= temp3;
								}

							} else {
								if (xxydist->getCount(father, fatherVal, mother,
										motherVal) > 0) {
										temp1 = mEstimate(parentChildYCount,
												parentYCount,
												xxxyDist_.getNoValues(child));
										spodeProbs[father][mother][y] *= temp1;

								}
								if (xxydist->getCount(father, fatherVal, child,
										childVal) > 0) {
										temp2 = mEstimate(parentChildYCount,
												fatherYChildCount,
												xxxyDist_.getNoValues(mother));
										spodeProbs[father][child][y] *= temp2;
								}
								if (xxydist->getCount(mother, motherVal, child,
										childVal) > 0) {
										temp3 = mEstimate(parentChildYCount,
												motherYChildCount,
												xxxyDist_.getNoValues(father));
										spodeProbs[mother][child][y] *= temp3;
								}
							}
						}
					}
				}
			}
		}
	}

	for (CatValue father = 1; father < noCatAtts_; father++) {
		for (CatValue mother = 0; mother < father; mother++) {
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[father][mother][y];
			}
		}
	}
	normalise(classDist);
}

void a3de_ms::aodeClassify(const instance &inst, std::vector<double> &classDist,
		xxyDist & xxyDist_) {

	//scale up by maximum possible factor to reduce risk of numeric underflow
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
							spodeProbs[parent][y] = weight_aode[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parent][y] = weight_aode[parent]
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
}
void a3de_ms::nbClassify(const instance &inst, std::vector<double> &classDist,
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

