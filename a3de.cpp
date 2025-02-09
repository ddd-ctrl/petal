/*
 * a3de.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#include <algorithm>

#include "a3de.h"
#include "assert.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStream.h"
#include "learnerRegistry.h"

static LearnerRegistrar registrar("a3de", constructor<a3de>);

a3de::a3de(char* const *& argv, char* const * end) :
		weight_a2de(1) {
	name_ = "a3de";

	trainSize_ = 0;
	testSetSize_ = std::numeric_limits<InstanceCount>::max();   // by default include all the training data in the test set for pas 3

	// TODO Auto-generated constructor stub
	weighted = false;
	minCount = 100;
	subsumptionResolution = false;
	selected = false;

	selected = false;
	oneSelective_ = false;
	twoSelective_ = false;

	chisq_ = false;
	acmi_ = false;
	mi_ = false;
	su_ = false;
	sum_ = false;
	avg_ = false;
	factor_ = 1;  //default value

	memorySelective_ = false;

	loo_=false;
	directRank_=false;

	empiricalMEst_ = false;
	// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		} else if (streq(argv[0] + 1, "acmi")) {
			selected = true;
			acmi_ = true;
		} else if (streq(argv[0] + 1, "avg")) {
			avg_ = true;
		} else if (argv[0][1] == 'c') {
			getUIntFromStr(argv[0] + 2, minCount, "c");
		} else if (streq(argv[0] + 1, "chisq")) {
			selected = true;
			chisq_ = true;
		} else if (streq(argv[0] + 1, "empirical")) {
			empiricalMEst_ = true;
		} else if (argv[0][1] == 'f') {
			unsigned int factor;
			getUIntFromStr(argv[0] + 2, factor, "f");
			factor_ = factor / 10.0;
			while (factor_ >= 1)
				factor_ /= 10;
		} else if (streq(argv[0] + 1, "loo")) {
			loo_ = true;
		} else if (argv[0][1] == 't') {
	      getUIntFromStr(argv[0]+2, testSetSize_, "t");

		} else if (streq(argv[0] + 1, "mi")) {
			selected = true;
			mi_ = true;
		} else if (streq(argv[0] + 1, "rank")) {
			directRank_ = true;
		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;
		} else if (streq(argv[0] + 1, "su")) {
			selected = true;
			su_ = true;
		} else if (streq(argv[0] + 1, "sub")) {
			subsumptionResolution = true;
		} else if (streq(argv[0] + 1, "sum")) {
			sum_ = true;
		} else if (streq(argv[0] + 1, "w")) {
			weighted = true;
		} else if (streq(argv[0] + 1, "one")) {
			oneSelective_ = true;
		} else if (streq(argv[0] + 1, "two")) {
			twoSelective_ = true;
		} else {
			error("a3de does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}
	
	
	if(loo_==true && directRank_==false)
	{
		mi_=true;
	}
	if (selected == true) {
		if (mi_ == false && su_ == false && chisq_ == false)
			chisq_ = true;
	}

}

a3de::~a3de() {
	// TODO Auto-generated destructor stub
}

void a3de::reset(InstanceStream &is) {
	xxxxyDist_.reset(is);

	pass_ = 1;


	trainSize_ = 0;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	inactiveCnt_ = 0;

	weight_aode.assign(noCatAtts_, 1);
	weight_a2de = crosstab<double>(noCatAtts_);
	weight = crosstab3D<double>(noCatAtts_);


	//initialise the weight for non-weighting

	weight_aode.assign(noCatAtts_, 1);
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

	generalizationSet.assign(noCatAtts_, false);

	active_.assign(noCatAtts_, true);
	instanceStream_ = &is;

	optParentIndex_=noCatAtts_*(noCatAtts_-1)*(noCatAtts_-2)/6;


	//delete and initialise the rmse vector
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].clear();
	}
	squaredError_.clear();
	squaredError_.resize(noCatAtts_*(noCatAtts_-1)*(noCatAtts_-2)/6);

	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].assign(noCatAtts_,0.0);
	}

  // initialise orderedAtts_ here so that incremental learning works
  orderedAtts_.clear();
  for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
		orderedAtts_.push_back(a);
	}
}

void a3de::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a3de::initialisePass() {
	  if (pass_ == 2) {
	    testSetSoFar_ = 0;        // used in pass 2 to count how many test cases have been used so far
	    seenSoFar_  = 0;          // used in pass 2 to count how many cases from the input stream have been seen so far
	  }
}

/// true iff no more passes are required. updated by finalisePass()
bool a3de::trainingIsFinished() {
	if (loo_==true)
		return pass_ > 2;
	else
		return pass_ > 1;
}

void a3de::train(const instance &inst) {

	if (pass_ == 1){
		xxxxyDist_.update(inst);
		trainSize_++;
	}
	else {
		assert(pass_ == 2);
		if ((testSetSize_-testSetSoFar_) / static_cast<double>(trainSize_-seenSoFar_++) < rand_())
			return;  // ignore all but testSetSize_ randomly selected cases
		testSetSoFar_++;

		LOOCV(inst);
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

void a3de::finalisePass() {

	if(pass_==1)
	{
		if (loo_==true) {
			if(mi_==true)
			{
				//get the mutual information to rank the attrbiutes
				std::vector<float> measure;
				getMutualInformation(xxxxyDist_.xxxyCounts.xxyCounts.xyCounts, measure);

				// sort the attributes on mutual information with the class

				if (!orderedAtts_.empty()) {
					valCmpClass cmp(&measure);
					std::sort(orderedAtts_.begin(), orderedAtts_.end(), cmp);
				}

			}
			else{
				assert(directRank_==true);

				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> mi;
				crosstab<float> cmiac(noCatAtts_);
				getAttClassCondMutualInf(xxxxyDist_.xxxyCounts.xxyCounts, cmiac);
				getMutualInformation(xxxxyDist_.xxxyCounts.xxyCounts.xyCounts, mi);
				
				orderedAtts_.clear();

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
				orderedAtts_.push_back(first);
				seletedAttributes[first]=true;

				while(orderedAtts_.size()<noCatAtts_)
				{
					maxFirst=true;
					for(CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						if(seletedAttributes[a]==false)
						{
							minimalCMI[a]=cmiac[a][orderedAtts_[0]];
							for (CategoricalAttribute i = 1; i < orderedAtts_.size(); i++) {
								if( cmiac[a][orderedAtts_[i]]< minimalCMI[a] )
								{
									minimalCMI[a] =cmiac[a][orderedAtts_[i]];
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
					orderedAtts_.push_back(maxIndex);
					seletedAttributes[maxIndex]=true;
				}
			}
		}

		if (weighted) {

			// for aode
			weight_aode.assign(noCatAtts_, 0);
			for (CategoricalAttribute i = 0; i < noCatAtts_; i++) {

				for (CatValue x = 0; x < xxxxyDist_.getNoValues(i);
						x++) {
					for (CatValue y = 0; y < noClasses_; y++) {
						double pXy =
								xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.jointP(i,
										x, y);
						if (pXy == 0)
							continue;

						double pY = xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(y);

						double pX = xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(i,
								x);

						double weightXy = pXy * log2(pXy / (pX * pY));
						weight_aode[i] += weightXy;
					}
				}
			}

			//for a2de
			//use mutualExt to weight
			xxyDist dist = xxxxyDist_.xxxyCounts.xxyCounts;
			for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
				for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

					double m = 0.0;
					double n = 0.0;

					for (CatValue v1 = 0; v1 < instanceStream_->getNoValues(x1);
							v1++) {
						for (CatValue v2 = 0; v2 < instanceStream_->getNoValues(x2);
								v2++) {
							for (CatValue y = 0; y < noClasses_; y++) {

								const double px1x2y =
										xxxxyDist_.xxxyCounts.xxyCounts.jointP(x1,
												v1, x2, v2, y);

								if (verbosity >= 4) {
									printf("%d\t%" ICFMT "\n\t%" ICFMT "\n\t%f\n",
											y,
											xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getClassCount(
													y), dist.xyCounts.count,
											xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
													y));
									printf("%d,%d,%d,%f\n", v1, v2, y, px1x2y);
								}

								if (px1x2y) {
									n =
											px1x2y
													* log2(
															px1x2y
																	/ (xxxxyDist_.xxxyCounts.xxyCounts.jointP(
																			x1, v1,
																			x2, v2)
																			* xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
																					y)));
									m += n;
									if (verbosity >= 4)
										if (x1 == 2 && x2 == 0) {
											printf("%e\t%e\t%f\n", px1x2y,
													xxxxyDist_.xxxyCounts.xxyCounts.jointP(
															x1, v1, x2, v2),
													xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
															y));
											printf("%e\n", n);
										}
								}
							}
						}
					}
					assert(m >= -0.00000001);
					// CMI is always positive, but allow for some imprecision
					weight_a2de[x1][x2] = m;
					weight_a2de[x2][x1] = m;
					//printf("%d,%d\t%f\n", x1, x2, m);
				}
			}

			//for a3de
			for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
				for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
						double m = 0.0;
						double n = 0.0;

						for (CatValue v1 = 0; v1 < instanceStream_->getNoValues(x1);
								v1++) {
							for (CatValue v2 = 0;
									v2 < instanceStream_->getNoValues(x2); v2++) {
								for (CatValue v3 = 0;
										v3 < instanceStream_->getNoValues(x3);
										v3++) {

									for (CatValue y = 0; y < noClasses_; y++) {
										const double px1x2x3y =
												xxxxyDist_.xxxyCounts.jointP(x1, v1,
														x2, v2, x3, v3, y);
										if (px1x2x3y == 0)
											continue;
										n =
												px1x2x3y
														* log2(
																px1x2x3y
																		/ (xxxxyDist_.xxxyCounts.jointP(
																				x1,
																				v1,
																				x2,
																				v2,
																				x3,
																				v3)
																				* xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(
																						y)));
										m += n;
									}

								}
							}

						}

						assert(m >= -0.00000001);
						// CMI is always positive, but allow for some imprecision
						weight[x1][x2][x3] = m;
						weight[x2][x1][x3] = m;
						weight[x1][x3][x2] = m;
						weight[x2][x3][x1] = m;
						weight[x3][x1][x2] = m;
						weight[x3][x2][x1] = m;
					}
				}
			}
		}

		if (selected) {

			// sort the attributes on symmetrical uncertainty with the class

			std::vector<CategoricalAttribute> order_;

			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
				order_.push_back(a);
			}
			xxyDist &xxyDist_=xxxxyDist_.xxxyCounts.xxyCounts;

			if (mi_ == true || su_ == true || acmi_ == true) {
				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> measure;
				crosstab<float> acmi(noCatAtts_);

				if (mi_ == true && acmi_ == true) {
					if (sum_ == true) {
						getAttClassCondMutualInf(xxyDist_, acmi);
						getMutualInformation(xxyDist_.xyCounts, measure);

						for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
							for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

								measure[x1] += acmi[x2][x1];
								measure[x2] += acmi[x1][x2];
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

								measure[x1] += acmi[x2][x1];
								measure[x2] += acmi[x1][x2];
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

							measure[x1] += acmi[x2][x1];
							measure[x2] += acmi[x1][x2];
							if (verbosity >= 5) {
								if (x1 == 2)
									printf("%u,", x2);
								if (x2 == 2)
									printf("%u,", x1);
							}
						}
					}
				}

				if (verbosity >= 2) {
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

				unsigned int noSelectedCatAtts_;
				unsigned int v;
				unsigned int i;
				v = instanceStream_->getNoValues(0);
				for (i = 1; i < noCatAtts_; i++)
					v += instanceStream_->getNoValues(i);
				v = v / noCatAtts_;

				if (oneSelective_) {

					for (noSelectedCatAtts_ = noCatAtts_;
							noCatAtts_ * noCatAtts_
									< noSelectedCatAtts_ * noSelectedCatAtts_
											* noSelectedCatAtts_ * noSelectedCatAtts_ * v
											* v; noSelectedCatAtts_--)
						;
					printf("Select the attributes according to the memory aode required.\n");
					printf("The number of attributes and selected attributes: %u,%u\n",
							noCatAtts_, noSelectedCatAtts_);
					printf("The average number of values for all attributes: %u\n", v);

				} else if (twoSelective_) {
					for (noSelectedCatAtts_ = noCatAtts_;
							noCatAtts_ * noCatAtts_ * noCatAtts_
									< noSelectedCatAtts_ * noSelectedCatAtts_
											* noSelectedCatAtts_ * noSelectedCatAtts_ * v;
							noSelectedCatAtts_--)
						;
					printf("Selecting the attributes according to the memory a2de required.\n");
					printf("The number of attributes and selected attributes: %u,%u\n",
							noCatAtts_, noSelectedCatAtts_);
					printf("The average number of values for all attributes: %u\n", v);

				} else

					noSelectedCatAtts_ = static_cast<unsigned int>(noCatAtts_ * factor_); ///< the number of selected attributes



				if (!order_.empty()) {

					valCmpClass cmp(&measure);
					std::sort(order_.begin(), order_.end(), cmp);

					if (verbosity >= 2) {
						const char * sep = "";
						printf("The order of attributes ordered by the measure:\n");
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							printf("%d:\t%f\n", order_[a], measure[order_[a]]);
							sep = ", ";
						}
						printf("\n");
					}

					//order by attribute number for selected attributes
					std::sort(order_.begin(), order_.begin() + noSelectedCatAtts_);

					//set the attribute selected or unselected for aode
					for (CategoricalAttribute a = noSelectedCatAtts_;
							a < noCatAtts_; a++) {
						active_[order_[a]] = false;
					}

					if (verbosity >= 2) {
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

			}else if  (chisq_ == true) {

				bool flag = true;
				double lowest;
				CategoricalAttribute attLowest;

				for (std::vector<CategoricalAttribute>::const_iterator it =
						order_.begin(); it != order_.end(); it++) {

					CategoricalAttribute a = *it;
					const unsigned int rows = instanceStream_->getNoValues(a);

					if (rows < 2) {
						active_[a] = false;
						inactiveCnt_++;
					} else {
						const unsigned int cols = noClasses_;
						InstanceCount *tab;
						allocAndClear(tab, rows * cols);

						for (CatValue r = 0; r < rows; r++) {
							for (CatValue c = 0; c < cols; c++) {
								tab[r * cols + c] +=
										xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(a, r,
												c);
							}
						}

						double critVal = 0.05 / noCatAtts_;
						double chisqVal = chiSquare(tab, rows, cols);

						//select the attribute with lowest chisq value as parent if there is attribute satisfying the
						//significance level of 5%
						if (flag == true) {
							lowest = chisqVal;
							attLowest = a;

							flag = false;
						} else {
							if (lowest > chisqVal) {
								lowest = chisqVal;
								attLowest = a;
							}
						}

						if (chisqVal > critVal) {
							if (verbosity >= 2)
								printf(
										"%s suppressed by chisq test against class\n",
										instanceStream_->getCatAttName(a));
							active_[a] = false;
							inactiveCnt_++;
						}
						delete[] tab;
					}
				}
				if (inactiveCnt_ == noCatAtts_) {
					active_[attLowest] = true;
					if (verbosity >= 2)
						printf("Only the attribute %u is active.\n", attLowest);

				}
				if (verbosity >= 2)
					printf(
							"The number of active parent and total attributes are: %u,%u\n",
							noCatAtts_ - inactiveCnt_, noCatAtts_);
			}

		}

	}
	else if(pass_==2)
	{
		if(loo_==true)
		{

			const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)*(noCatAtts_-2)/6;
			for (CategoricalAttribute parent = 0; parent < parentNumber;
					parent++) {
				if(verbosity>=3)
					printf("parent:%d:",parent);
				for (CategoricalAttribute child = 0; child < noCatAtts_;
						child++) {
					squaredError_[parent][child] = sqrt(
							squaredError_[parent][child]
					/ xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.count);
					if(verbosity>=3)
						printf("%f,",squaredError_[parent][child]);
				}
				if(verbosity>=3)
					printf("\n");
			}

		
			optParentIndex_=0;
			optChildIndex_=indexOfMinVal(squaredError_[optParentIndex_]);
			double minVal=squaredError_[optParentIndex_][optChildIndex_];

			for (CatValue parent = 1; parent < parentNumber; parent++) {
				unsigned int optColumn=indexOfMinVal(squaredError_[parent]);
				double  minValRow=squaredError_[parent][optColumn];
				if(minVal>minValRow)
				{
					optParentIndex_=parent;
					optChildIndex_=optColumn;
					minVal=minValRow;
				}
			}
		
			optParentIndex_++;
			optChildIndex_++;
			if(verbosity>=2)
			{
				printf("Parent pairs, all: %d, selected: %d\n",parentNumber,optParentIndex_);
				printf("Child, all: %d, selected: %d\n",noCatAtts_,optChildIndex_);
			}
		}
	}
	pass_++;
}


void a3de::LOOCV(const instance &inst)
{


	generalizationSet.assign(noCatAtts_, false);

	xxxyDist * xxxydist = &xxxxyDist_.xxxyCounts;
	xxyDist * xxydist = &xxxydist->xxyCounts;
	xyDist * xydist = &xxydist->xyCounts;

	const InstanceCount totalCount = xydist->count-1;
	
	const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)*(noCatAtts_-2)/6;
	const CatValue trueClass = inst.getClass();
	std::vector<double> classDist;
	classDist.assign(noClasses_,0.0);


	//compute the generalisation set and substitution set for
	//lazy subsumption resolution
	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					InstanceCount countOfxixj =
							xxxxyDist_.xxxyCounts.xxyCounts.getCount(i,
									inst.getCatVal(i), j, inst.getCatVal(j))-1;
					InstanceCount countOfxj =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(j,
									inst.getCatVal(j))-1;
					InstanceCount countOfxi =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(i,
									inst.getCatVal(i))-1;

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

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_ - 2) * (noCatAtts_) / 6.0);
	

	//	// first assign the spodeProbs array
	std::vector<std::vector<std::vector<std::vector<double> > > > spodeProbs;
	spodeProbs.resize(noCatAtts_);


	fdarray<InstanceCount> xxxyCount(parentNumber, noClasses_);
	crosstab3D<bool> activeParent(noCatAtts_);


	for (CatValue parent1Index = 2; parent1Index < noCatAtts_; parent1Index++) {

		spodeProbs[parent1Index].resize(parent1Index);
		const CategoricalAttribute parent1 = orderedAtts_[parent1Index];
		const CatValue parent1Val = inst.getCatVal(parent1);

		for (CatValue parent2Index = 1; parent2Index < parent1Index; parent2Index++) {

			spodeProbs[parent1Index][parent2Index].resize(parent2Index);
			const CategoricalAttribute parent2= orderedAtts_[parent2Index];
			const CatValue parent2Val = inst.getCatVal(parent2);

			for (CatValue parent3Index = 0; parent3Index < parent2Index; parent3Index++) {

				spodeProbs[parent1Index][parent2Index][parent3Index].resize(noClasses_);
				const CategoricalAttribute parent3= orderedAtts_[parent3Index];
				const CatValue parent3Val = inst.getCatVal(parent3);
				
				const CategoricalAttribute parentIndex=parent1Index*(parent1Index-1)*(parent1Index-2)/6+parent2Index*(parent2Index-1)/2+parent3Index;
				CatValue parent = 0;
				for (CatValue y = 0; y < noClasses_; y++) {
					
					xxxyCount[parentIndex][y]= xxxydist->getCount(parent1, parent1Val,
							parent2, parent2Val, parent3, parent3Val, y);
					if(y==trueClass)
						xxxyCount[parentIndex][y]--;
					parent += xxxyCount[parentIndex][y];
				}

				if (parent > 0) {
					activeParent[parent1][parent2][parent3] = true;
					delta++;
					for (CatValue y = 0; y < noClasses_; y++) {

						spodeProbs[parent1Index][parent2Index][parent3Index][y] =
								weight[parent1][parent2][parent3]
								* scaleFactor
								* mEstimate(xxxyCount[parentIndex][y], totalCount,
									noClasses_ * xxxxyDist_.getNoValues(parent1)
											* xxxxyDist_.getNoValues(parent2)
											* xxxxyDist_.getNoValues(parent3)); // scale up by maximum possible factor to reduce risk of numeric underflow
					}
				}
			}
		}
	}

	if (delta == 0) {
		printf("there are no eligible parents.\n");
		return;
	}

	std::vector<std::vector<double> >spodeProbsSumOnRow;
	spodeProbsSumOnRow.resize(noCatAtts_);
	for (CatValue y = 0; y < noCatAtts_; y++) {
		spodeProbsSumOnRow[y].assign(noClasses_,0.0);
	}


	//deal with all the attributes as parents and child
	for (CatValue parent1Index = 2; parent1Index < noCatAtts_; parent1Index++) {

		const CategoricalAttribute parent1 = orderedAtts_[parent1Index];
		const CatValue parent1Val = inst.getCatVal(parent1);

		for (CatValue parent2Index = 1; parent2Index < parent1Index; parent2Index++) {

			const CategoricalAttribute parent2 = orderedAtts_[parent2Index];
			const CatValue parent2Val = inst.getCatVal(parent2);

			for (CatValue parent3Index = 0; parent3Index < parent2Index; parent3Index++) {

				const CategoricalAttribute parent3 = orderedAtts_[parent3Index];
				const CatValue parent3Val = inst.getCatVal(parent3);
				const CategoricalAttribute parentIndex=parent1Index*(parent1Index-1)*(parent1Index-2)/6+parent2Index*(parent2Index-1)/2+parent3Index;

				if(activeParent[parent1][parent2][parent3] != true)
					continue;

				for (CatValue childIndex = 0; childIndex < noCatAtts_; childIndex++) {

					const CategoricalAttribute child= orderedAtts_[childIndex];
					const CatValue childVal = inst.getCatVal(child);

					for (CatValue y = 0; y < noClasses_; y++) {

						if(!generalizationSet[parent1]&&!generalizationSet[parent2]&&!generalizationSet[parent3]&&!generalizationSet[child])
						{
							if (child != parent1 && child != parent2&& child != parent3) {
								InstanceCount parentChildYCount =
										xxxxyDist_.getCount(parent1,parent1Val, parent2,parent2Val,parent3,parent3Val,child,childVal, y);
								if(y==trueClass)
									parentChildYCount--;

								spodeProbs[parent1Index][parent2Index][parent3Index][y] *= mEstimate(parentChildYCount,
										xxxyCount[parentIndex][y],xxxxyDist_.getNoValues(child));

							}

						}

						if(!generalizationSet[parent1]&&!generalizationSet[parent2]&&!generalizationSet[parent3])
						{
							spodeProbsSumOnRow[childIndex][y] +=
									spodeProbs[parent1Index][parent2Index][parent3Index][y];
						}
						classDist[y] = spodeProbsSumOnRow[childIndex][y];
					}


					if(sum(classDist)!=0)
					{
						normalise(classDist);
						const double error = 1.0 - classDist[trueClass];
						squaredError_[parentIndex][childIndex] += error * error;
					   
					}

				}
			}
		}
	}

}

void a3de::classify(const instance &inst, std::vector<double> &classDist) {

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
					InstanceCount countOfxixj =
							xxxxyDist_.xxxyCounts.xxyCounts.getCount(i,
									inst.getCatVal(i), j, inst.getCatVal(j));
					InstanceCount countOfxj =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(j,
									inst.getCatVal(j));
					InstanceCount countOfxi =
							xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.getCount(i,
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

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_ - 2) * (noCatAtts_) / 6.0);
	std::vector<std::vector<std::vector<std::vector<double> > > > spodeProbs;
	spodeProbs.resize(noCatAtts_);
	

	if(loo_==true)
	{
		const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)*(noCatAtts_-2)/6;
		fdarray<InstanceCount> xxxyCount(parentNumber, noClasses_);
		crosstab3D<bool> activeParent(noCatAtts_);	

		for (CatValue parent1Index = 2; parent1Index < noCatAtts_; parent1Index++) {

			spodeProbs[parent1Index].resize(parent1Index);
			const CategoricalAttribute parent1 = orderedAtts_[parent1Index];
			const CatValue parent1Val = inst.getCatVal(parent1);

			for (CatValue parent2Index = 1; parent2Index < parent1Index; parent2Index++) {

				spodeProbs[parent1Index][parent2Index].resize(parent2Index);
				const CategoricalAttribute parent2= orderedAtts_[parent2Index];
				const CatValue parent2Val = inst.getCatVal(parent2);

				for (CatValue parent3Index = 0; parent3Index < parent2Index; parent3Index++) {

					spodeProbs[parent1Index][parent2Index][parent3Index].resize(noClasses_);
					const CategoricalAttribute parent3= orderedAtts_[parent3Index];
					const CatValue parent3Val = inst.getCatVal(parent3);
				
					const CategoricalAttribute parentIndex=parent1Index*(parent1Index-1)*(parent1Index-2)/6+parent2Index*(parent2Index-1)/2+parent3Index;
					CatValue parent = 0;
					for (CatValue y = 0; y < noClasses_; y++) {
						xxxyCount[parentIndex][y]= xxxydist->getCount(parent1, parent1Val,
								parent2, parent2Val, parent3, parent3Val, y);
						parent += xxxyCount[parentIndex][y];
					}

					if (parent > 0) {
						activeParent[parent1][parent2][parent3] = true;
						delta++;
						for (CatValue y = 0; y < noClasses_; y++) {

							spodeProbs[parent1Index][parent2Index][parent3Index][y] =
									weight[parent1][parent2][parent3]
									* scaleFactor
									* mEstimate(xxxyCount[parentIndex][y], totalCount,
										noClasses_ * xxxxyDist_.getNoValues(parent1)
												* xxxxyDist_.getNoValues(parent2)
												* xxxxyDist_.getNoValues(parent3)); // scale up by maximum possible factor to reduce risk of numeric underflow
						}
					}
				}
			}
		}

		if (delta == 0) {
			a2deClassify(inst, classDist, *xxxydist);
			return;
		}
		
		//deal with all the attributes as parents and child

		bool hasParent=false;
		for (CatValue parent1Index = 2; parent1Index < noCatAtts_; parent1Index++) {

			const CategoricalAttribute parent1 = orderedAtts_[parent1Index];
			const CatValue parent1Val = inst.getCatVal(parent1);
						
			if(generalizationSet[parent1]==true)
				continue;

			for (CatValue parent2Index = 1; parent2Index < parent1Index; parent2Index++) {

				const CategoricalAttribute parent2 = orderedAtts_[parent2Index];
				const CatValue parent2Val = inst.getCatVal(parent2);

				if(generalizationSet[parent2]==true)
					continue;

				for (CatValue parent3Index = 0; parent3Index < parent2Index; parent3Index++) {

					const CategoricalAttribute parent3 = orderedAtts_[parent3Index];
					const CatValue parent3Val = inst.getCatVal(parent3);

					if(generalizationSet[parent3]==true)
						continue;

					const CategoricalAttribute parentIndex=parent1Index*(parent1Index-1)*(parent1Index-2)/6+parent2Index*(parent2Index-1)/2+parent3Index;
					
					if(parentIndex>=optParentIndex_)
						goto over;

					if(activeParent[parent1][parent2][parent3] != true)
						continue;

					for (CatValue childIndex = 0; childIndex < optChildIndex_; childIndex++) {

						const CategoricalAttribute child= orderedAtts_[childIndex];
						const CatValue childVal = inst.getCatVal(child);

						if(generalizationSet[child]==true)
							continue;

						for (CatValue y = 0; y < noClasses_; y++) {
							if (child != parent1 && child != parent2&& child != parent3) {
								InstanceCount parentChildYCount =
										xxxxyDist_.getCount(parent1,parent1Val, parent2,parent2Val,parent3,parent3Val,child,childVal, y);

								spodeProbs[parent1Index][parent2Index][parent3Index][y] *= mEstimate(parentChildYCount,
										xxxyCount[parentIndex][y],xxxxyDist_.getNoValues(child));
							}
						}
					}
					
					hasParent=true;
					for (CatValue y = 0; y < noClasses_; y++) {
						classDist[y] += spodeProbs[parent1Index][parent2Index][parent3Index][y];
					}
				}
			}
		}
		
		//deal with the possible no parent case
over:   if(hasParent==false)
		{
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] =xxxxyDist_.xxxyCounts.xxyCounts.xyCounts.p(y) ;
			}
		}
		normalise(classDist);
		return;

	}



	//Normal A3DE


	//first assign the spodeProbs array
	for (CatValue parent1 = 2; parent1 < noCatAtts_; parent1++) {
		spodeProbs[parent1].resize(parent1);
		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {
			spodeProbs[parent1][parent2].resize(parent2);
			for (CatValue parent3 = 0; parent3 < parent2; parent3++)
				spodeProbs[parent1][parent2][parent3].assign(noClasses_, 0);
		}
	}

	for (CatValue parent1 = 2; parent1 < noCatAtts_; parent1++) {

		//select attribute for subsumption resolution
		if (generalizationSet[orderedAtts_[parent1]])
			continue;
		const CatValue parent1Val = inst.getCatVal(orderedAtts_[parent1]);

		for (CatValue parent2 = 1; parent2 < parent1; parent2++) {

			//select attribute for subsumption resolution
			if (generalizationSet[orderedAtts_[parent2]])
				continue;
			const CatValue parent2Val = inst.getCatVal(orderedAtts_[parent2]);

			for (CatValue parent3 = 0; parent3 < parent2; parent3++) {

				//select attribute for subsumption resolution
				if (generalizationSet[orderedAtts_[parent3]])
					continue;
				const CatValue parent3Val = inst.getCatVal(orderedAtts_[parent3]);

				CatValue parent = 0;
				for (CatValue y = 0; y < noClasses_; y++) {
					parent += xxxydist->getCount(orderedAtts_[parent1], parent1Val,
							orderedAtts_[parent2], parent2Val, orderedAtts_[parent3],
							parent3Val, y);
				}

				if (parent > 0) {

					delta++;
					for (CatValue y = 0; y < noClasses_; y++) {
						spodeProbs[parent1][parent2][parent3][y] =
								weight[orderedAtts_[parent1]][orderedAtts_[parent2]][orderedAtts_[parent3]] * scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

						InstanceCount parentYCount = xxxydist->getCount(
								orderedAtts_[parent1], parent1Val, orderedAtts_[parent2],
								parent2Val, orderedAtts_[parent3], parent3Val, y);
						if (empiricalMEst_) {

							spodeProbs[parent1][parent2][parent3][y] *=
									empiricalMEstimate(parentYCount, totalCount,
											xydist->p(y)
													* xydist->p(orderedAtts_[parent1],
															parent1Val)
													* xydist->p(orderedAtts_[parent2],
															parent2Val)
													* xydist->p(orderedAtts_[parent3],
															parent3Val));

						} else {
							double temp = mEstimate(parentYCount, totalCount,
									noClasses_ * xxxxyDist_.getNoValues(orderedAtts_[parent1])
											* xxxxyDist_.getNoValues(orderedAtts_[parent2])
											* xxxxyDist_.getNoValues(orderedAtts_[parent3]));
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

	//deal with all the attributes as parents and child
	for (CatValue parent1 = 3; parent1 < noCatAtts_; parent1++) {
		if (generalizationSet[orderedAtts_[parent1]])
			continue;
		const CatValue parent1Val = inst.getCatVal(orderedAtts_[parent1]);

		for (CatValue parent2 = 2; parent2 < parent1; parent2++) {
			if (generalizationSet[orderedAtts_[parent2]])
				continue;
			const CatValue parent2Val = inst.getCatVal(orderedAtts_[parent2]);

			XXYSubDist xxySubDist(
					xxxxyDist_.getXXYSubDist(orderedAtts_[parent1], parent1Val,
		 				orderedAtts_[parent2], parent2Val), noClasses_);

			XYSubDist xySubDistParent12(
					xxxydist->getXYSubDist(orderedAtts_[parent1], parent1Val,
							orderedAtts_[parent2], parent2Val), noClasses_);

			for (CatValue parent3 = 1; parent3 < parent2; parent3++) {
				if (generalizationSet[orderedAtts_[parent3]])
					continue;
				const CatValue parent3Val = inst.getCatVal(orderedAtts_[parent3]);

				XYSubDist xySubDist(
						xxySubDist.getXYSubDist(parent3, parent3Val),
						noClasses_);

				XYSubDist xySubDistParent23(
						xxxydist->getXYSubDist(orderedAtts_[parent2], parent2Val,
								orderedAtts_[parent3], parent3Val), noClasses_);
				XYSubDist xySubDistParent13(
						xxxydist->getXYSubDist(orderedAtts_[parent1], parent1Val,
								orderedAtts_[parent3], parent3Val), noClasses_);

				for (CatValue child = 0; child < parent3; child++) {

					if (!generalizationSet[orderedAtts_[child]]) {

						const CatValue childVal = inst.getCatVal(orderedAtts_[child]);

						for (CatValue y = 0; y < noClasses_; y++) {

							InstanceCount parentChildYCount =
									xySubDist.getCount(child, childVal, y);

							InstanceCount parentYCount =
									xySubDistParent12.getCount(orderedAtts_[parent3],
											parent3Val, y);

							InstanceCount parent23YChildCount =
									xySubDistParent23.getCount(orderedAtts_[child],
											childVal, y);
							InstanceCount parent13YChildCount =
									xySubDistParent13.getCount(orderedAtts_[child],
											childVal, y);
							InstanceCount parent12YChildCount =
									xySubDistParent12.getCount(orderedAtts_[child],
											childVal, y);
							double temp;
							if (empiricalMEst_) {

								if (xxxydist->getCount(orderedAtts_[parent1],
										parent1Val, orderedAtts_[parent2], parent2Val,
										orderedAtts_[parent3], parent3Val) > 0) {

									temp = empiricalMEstimate(parentChildYCount,
											parentYCount,
											xydist->p(orderedAtts_[child], childVal));
									spodeProbs[parent1][parent2][parent3][y] *=
											temp;
								}
								if (xxxydist->getCount(orderedAtts_[parent2],
										parent2Val, orderedAtts_[parent3], parent3Val,
										orderedAtts_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent23YChildCount,
											xydist->p(orderedAtts_[parent1],
													parent1Val));
									spodeProbs[parent2][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(orderedAtts_[parent1],
										parent1Val, orderedAtts_[parent3], parent3Val,
										orderedAtts_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent13YChildCount,
											xydist->p(orderedAtts_[parent2],
													parent2Val));
									spodeProbs[parent1][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(orderedAtts_[parent1],
										parent1Val, orderedAtts_[parent2], parent2Val,
										orderedAtts_[child], childVal) > 0) {
									temp = empiricalMEstimate(parentChildYCount,
											parent12YChildCount,
											xydist->p(orderedAtts_[parent3],
													parent3Val));
									spodeProbs[parent1][parent2][child][y] *=
											temp;
								}

							} else {

								if (xxxydist->getCount(orderedAtts_[parent1],
										parent1Val, orderedAtts_[parent2], parent2Val,
										orderedAtts_[parent3], parent3Val) > 0) {
									temp = mEstimate(parentChildYCount,
											parentYCount,
											xxxxyDist_.getNoValues(child));

									spodeProbs[parent1][parent2][parent3][y] *=
											temp;

								}
								if (xxxydist->getCount(orderedAtts_[parent2],
										parent2Val, orderedAtts_[parent3], parent3Val,
										orderedAtts_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent23YChildCount,
											xxxxyDist_.getNoValues(parent1));
									spodeProbs[parent2][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(orderedAtts_[parent1],
										parent1Val, orderedAtts_[parent3], parent3Val,
										orderedAtts_[child], childVal) > 0) {
									temp = mEstimate(parentChildYCount,
											parent13YChildCount,
											xxxxyDist_.getNoValues(parent2));
									spodeProbs[parent1][parent3][child][y] *=
											temp;
								}
								if (xxxydist->getCount(orderedAtts_[parent1],
										parent1Val, orderedAtts_[parent2], parent2Val,
										orderedAtts_[child], childVal) > 0) {
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

	for (CatValue parent1 = 2; parent1 < noCatAtts_; parent1++) {
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

void a3de::a2deClassify(const instance &inst, std::vector<double> &classDist,
		xxxyDist & xxxyDist_) {
	CatValue delta = 0;

//	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_ - 2) / 2.0);

	for (CatValue father = 0; father < noCatAtts_; father++) {

		//select the attribute according to mutual information as father
		if (active_[father] == false)
			continue;

		if (!generalizationSet[father]) {

			for (CatValue mother = 0; mother < father; mother++) {

				//select the attribute according to mutual information as mother
				if (active_[mother] == false)
					continue;

				if (!generalizationSet[mother]) {
					CatValue parent = 0;
					for (CatValue y = 0; y < noClasses_; y++) {
						parent += xxxyDist_.xxyCounts.getCount(father,
								inst.getCatVal(father), mother,
								inst.getCatVal(mother), y);
					}
					if (parent > 0) {

						delta++;

						for (CatValue y = 0; y < noClasses_; y++) {
							double p = weight_a2de[father][mother]
									* xxxyDist_.xxyCounts.jointP(father,
											inst.getCatVal(father), mother,
											inst.getCatVal(mother), y)
									* scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

							for (CatValue child = 0; child < noCatAtts_;
									child++) {

								if (!generalizationSet[child]) {
									if (child != father && child != mother)
										p *= xxxyDist_.unorderedP(child,
												inst.getCatVal(child), father,
												inst.getCatVal(father), mother,
												inst.getCatVal(mother), y);
								}
							}
							classDist[y] += p;
							if (verbosity >= 3) {
								printf("%f,", classDist[y]);
							}
						}
						if (verbosity >= 3) {
							printf("<<<<\n");
						}
					}
				}
			}
		}
		if (verbosity >= 3) {
			printf(">>>>>>>\n");
		}
	}
	if (delta == 0) {
		aodeClassify(inst, classDist, xxxyDist_.xxyCounts);
	} else

		normalise(classDist);
	if (verbosity >= 3) {
		print(classDist);
		printf("\n");
	}

}

void a3de::aodeClassify(const instance &inst, std::vector<double> &classDist,
		xxyDist & xxyDist_) {

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;

	CatValue delta = 0;

	for (CategoricalAttribute parent = 0; parent < noCatAtts_; parent++) {

		//select the attribute according to mutual information as parent
		if (active_[parent] == false)
			continue;

		if (!generalizationSet[parent]) {

			if (xxyDist_.xyCounts.getCount(parent, inst.getCatVal(parent))
					> 0) {

				delta++;
				for (CatValue y = 0; y < noClasses_; y++) {
					double p = weight_aode[parent]
							* xxyDist_.xyCounts.jointP(parent,
									inst.getCatVal(parent), y) * scaleFactor;

					for (CategoricalAttribute child = 0; child < noCatAtts_;
							child++) {

						//should select child using the conditional mutual information on parent
						//						if (active_[child] == false)
						//							continue;

						if (!generalizationSet[child]) {
							if (child != parent)
								p *= xxyDist_.p(child, inst.getCatVal(child),
										parent, inst.getCatVal(parent), y);
						}
					}
					classDist[y] += p;
				}
			}
		}
	}

	if (delta == 0) {
		nbClassify(inst, classDist, xxyDist_.xyCounts);
	} else
		normalise(classDist);

}
void a3de::nbClassify(const instance &inst, std::vector<double> &classDist,
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

