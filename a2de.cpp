/*
 * a2de.cpp
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */


#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
//#ifndef DBG_NEW
//#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
//#define new DBG_NEW
//#endif
#include <stdlib.h>
#include <crtdbg.h>
#endif
#endif

#include "a2de.h"
#include "assert.h"
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "instanceStream.h"
#include "learnerRegistry.h"
#include "ALGLIB_specialfunctions.h"



static LearnerRegistrar registrar("a2de", constructor<a2de>);

a2de::a2de(char* const *& argv, char* const * end) :
		weight(1) {
	name_ = "A2DE";


	trainSize_ = 0;
	testSetSize_ = std::numeric_limits<InstanceCount>::max();   // by default include all the training data in the test set for pas 3
	testSetSizeP_ = std::numeric_limits<InstanceCount>::max();   // by default include all the training data in the test set for pas 1
	sample4prmtr_=false;
    bayesRace_=false;


	// TODO Auto-generated constructor stub
	weighted = false;
	minCount = 100;
	subsumptionResolution = false;
	avg = false;
	selected = false;

	su_ = false;
	mi_ = false;
	mRMR_=false;
    ind_=false;
    allChild_=false;
    vote_=false;

	empiricalMEst_ = false;
	empiricalMEst2_ = false;
	factor_ = 2;
	count = 0;
	loo_=false;
	directRank_=false;


//
// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
        } else if (streq(argv[0] + 1, "bayesrace")) {
            bayesRace_ = true;
		} else if (streq(argv[0] + 1, "loo")) {
			loo_ = true;
		} else if (streq(argv[0] + 1, "ind")) {
			ind_ = true;
		} else if (streq(argv[0] + 1, "allchild")) {
			allChild_ = true;
		} else if (streq(argv[0] + 1, "vote")) {
			vote_ = true;
		} else if (streq(argv[0] + 1, "rank")) {
			directRank_ = true;
		} else if (argv[0][1] == 't') {
	      getUIntFromStr(argv[0]+2, testSetSize_, "t");

		} else if (argv[0][1] == 'p') {
	      getUIntFromStr(argv[0]+2, testSetSizeP_, "p");
	      sample4prmtr_ = true;
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
		} else if (streq(argv[0] + 1, "avg")) {
			avg = true;
		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;
		} else if (streq(argv[0] + 1, "mi")) {
			selected = true;
			mi_ = true;
		} else if (streq(argv[0] + 1, "mrmr")) {
			selected = true;
			mRMR_ = true;
		} else if (argv[0][1] == 'f') {
			getUIntFromStr(argv[0] + 2, factor_, "f");
		} else if (streq(argv[0] + 1, "su")) {
			selected = true;
			su_ = true;
		} else if (streq(argv[0] + 1, "chisq")) {
			selected = true;
			chisq_ = true;
		} else {
			error("A2de does not support argument %s\n", argv[0]);
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
		if (mi_ == false && su_ == false && chisq_ == false && mRMR_==false)
			chisq_ = true;
	}

}

a2de::~a2de() {
	// TODO Auto-generated destructor stub
}

a2de::a2de(const a2de& l) {
  name_ = "A2DE";
//  rand_=l.rand_;

 bayesRace_=l.bayesRace_;
  trainSize_ = l.trainSize_;
  testSetSize_ = l.testSetSize_;
  weighted = l.weighted;
  weight = l.weight;
  minCount = l.minCount;
  subsumptionResolution = l.subsumptionResolution;
  avg = l.avg;
  selected = l.selected;
  su_ = l.su_;
  mi_ = l.mi_;
  mRMR_=l.mRMR_;
  ind_=l.ind_;
  allChild_=l.allChild_;
  vote_=l.vote_;

  empiricalMEst_ = l.empiricalMEst_;
  empiricalMEst2_ = l.empiricalMEst2_;
  factor_ = l.factor_;
  count = l.count;
  loo_=l.loo_;
  directRank_=l.directRank_;
}

learner* a2de::clone() const {
  return new a2de(*this);
}

void a2de::reset(InstanceStream &is) {
	xxxyDist_.reset(is);
	pass_ = 1;
	trainSize_ = 0;

    delta_=0.01;

	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	inactiveCnt_ = 0;

	weight = crosstab<double>(noCatAtts_);

	weightaode.assign(noCatAtts_, 1);

	generalizationSet.assign(noCatAtts_, false);


	active_.assign(noCatAtts_, true);
	instanceStream_ = &is;
	count = 0;

	optParentIndex_=noCatAtts_*(noCatAtts_-1)/2;
	indOptChildIndex_.assign(noCatAtts_*(noCatAtts_-1)/2,noCatAtts_);
    optSquaredError_.clear();
    optSquaredError_.resize(noCatAtts_*(noCatAtts_-1)/2);


	//delete and initialise the rmse vector
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].clear();
	}
	squaredError_.clear();
	squaredError_.resize(noCatAtts_*(noCatAtts_-1)/2);
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].assign(noCatAtts_,0.0);
        }

    meanOfModels_.clear();
    varianceOfModels_.clear();
    modelThrownOut_.clear();

    meanOfModels_.assign(noCatAtts_*noCatAtts_*(noCatAtts_-1)/2, 0.0);
    varianceOfModels_.assign(noCatAtts_*noCatAtts_*(noCatAtts_-1)/2, 0.0);
    modelThrownOut_.assign(noCatAtts_*noCatAtts_*(noCatAtts_-1)/2,false);
    noOfModelsThrownOut_=0;
}

void a2de::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void a2de::initialisePass() {
	count = 0;

	if(pass_==1)
	{
		if(sample4prmtr_==true)
		{
			testSetSoFarP_ = 0;        // used in pass 1 to count how many test cases have been used so far
			seenSoFarP_  = 0;          // used in pass 1 to count how many cases from the input stream have been seen so far

			trainSize_=instanceStream_->size();
			instanceStream_->rewind();
		}
	}
	else if (pass_ == 2) {
	    testSetSoFar_ = 0;        // used in pass 2 to count how many test cases have been used so far
	    seenSoFar_  = 0;          // used in pass 2 to count how many cases from the input stream have been seen so far
	  }

    else if(pass_=3)
    {
        optSquaredError_.clear();
        optSquaredError_.resize(noCatAtts_*(noCatAtts_-1)/2);
    }
}

/// true iff no more passes are required. updated by finalisePass()
bool a2de::trainingIsFinished() {
	if (loo_==true)
    {
        if(ind_==true)
            return pass_>3;
        else
            return pass_ > 2;
    }
	else
		return pass_ > 1;
}

void a2de::train(const instance &inst) {

	if (pass_ == 1){
		if(sample4prmtr_==true)
		{
			if ((testSetSizeP_-testSetSoFarP_) / static_cast<double>(trainSize_-seenSoFarP_++) < rand_())
				return;  // ignore all but testSetSize_ randomly selected cases
			testSetSoFarP_++;
			xxxyDist_.update(inst);
		}else
		{
			xxxyDist_.update(inst);
			trainSize_++;
		}
	}

	else if(pass_==2)
    {
		if ((testSetSize_-testSetSoFar_) / static_cast<double>(trainSize_-seenSoFar_++) < rand_())
			return;  // ignore all but testSetSize_ randomly selected cases
		testSetSoFar_++;

		LOOCV(inst);
	}
	else
    {
        assert(pass_ == 3);
        LOOCV2(inst);
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


//This will sort the attributes in ascending order
class sortAscendClass {
public:
	sortAscendClass(std::vector<float> *s) {
		val = s;
	}

	bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
		return (*val)[a] < (*val)[b];
	}

private:
	std::vector<float> *val;
};


void a2de::finalisePass() {


	if(pass_==1)
	{

		if (loo_==true) {

			if(mi_==true)
			{
				orderedAtts_.clear();
				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					orderedAtts_.push_back(a);
				}
				//get the mutual information to rank the attrbiutes
				std::vector<float> measure;
				getMutualInformation(xxxyDist_.xxyCounts.xyCounts, measure);

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
				getAttClassCondMutualInf(xxxyDist_.xxyCounts, cmiac);
				getMutualInformation(xxxyDist_.xxyCounts.xyCounts, mi);

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

				//important
				orderedAtts_.clear();

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


		//initialise the weight for non-weighting

		weightaode.assign(noCatAtts_, 1);

		for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				weight[x1][x2] = 1;
				weight[x2][x1] = 1;
			}
		}

		if (weighted) {

			//compute weight for aode
			//there are two method to compute the weight for aode
			//one is using raw instance count,the other alternative is to use m-estimation.
			// m-estimation is used default

			weightaode.assign(noCatAtts_, 0);

			for (CatValue i = 0; i < noCatAtts_; i++) {

				for (CatValue x = 0; x < xxxyDist_.getNoValues(i); x++) {
					for (CatValue y = 0; y < noClasses_; y++) {
						double pXy = xxxyDist_.xxyCounts.xyCounts.jointP(i, x, y);
						double pY = xxxyDist_.xxyCounts.xyCounts.p(y);

						double pX = 0;
						for (CatValue yPrime = 0; yPrime < noClasses_; yPrime++) {
							pX += xxxyDist_.xxyCounts.xyCounts.jointP(i, x, yPrime);
						}
						if (pXy == 0)
							continue;
						double weightXy = pXy * log2(pXy / (pX * pY));
						weightaode[i] += weightXy;
					}
				}
			}

			//use average of the mutual information as weight

			if (avg == true) {
				std::vector<float> w(noCatAtts_, 0);

				xxyDist dist = xxxyDist_.xxyCounts;

				getMutualInformation(xxxyDist_.xxyCounts.xyCounts, w);

				for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
					for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

						double m = (w[x1] + w[x2]) / 2;
						weight[x1][x2] = m;
						weight[x2][x1] = m;
					}
				}
			}
			//use mutualExt to weight
			else {
				xxyDist dist = xxxyDist_.xxyCounts;
				for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
					for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

						double m = 0.0;
						double n = 0.0;

						for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
							for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
								for (CatValue y = 0; y < noClasses_; y++) {

									const double px1x2y =
											xxxyDist_.xxyCounts.jointP(x1, v1, x2,
													v2, y);

									if (verbosity >= 4) {
										printf(
												"%d\t%" ICFMT "\n\t%" ICFMT "\n\t%f\n",
												y,
												xxxyDist_.xxyCounts.xyCounts.getClassCount(
														y), dist.xyCounts.count,
												xxxyDist_.xxyCounts.xyCounts.p(y));
										printf("%d,%d,%d,%f\n", v1, v2, y, px1x2y);
									}

									if (px1x2y) {
										n =
												px1x2y
														* log2(
																px1x2y
																		/ (xxxyDist_.xxyCounts.jointP(
																				x1,
																				v1,
																				x2,
																				v2)
																				* xxxyDist_.xxyCounts.xyCounts.p(
																						y)));
										m += n;
										if (verbosity >= 4)
											if (x1 == 2 && x2 == 0) {
												printf("%e\t%e\t%f\n", px1x2y,
														xxxyDist_.xxyCounts.jointP(
																x1, v1, x2, v2),
														xxxyDist_.xxyCounts.xyCounts.p(
																y));
												printf("%e\n", n);
											}
									}
								}
							}
						}
						assert(m >= -0.00000001);
						// CMI is always positive, but allow for some imprecision
						weight[x1][x2] = m;
						weight[x2][x1] = m;
						//printf("%d,%d\t%f\n", x1, x2, m);
					}
				}
			}
		}

		if (selected) {

			// sort the attributes on symmetrical uncertainty with the class

			std::vector<CategoricalAttribute> order;

			for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
				order.push_back(a);
			}

			if(mRMR_==true){

				//Peng's mRMR to select the attribute set by maximal relivance and minimal redundency
				std::vector<float> MI;
				crosstab<float> PMI(noCatAtts_);
				std::vector<CategoricalAttribute> selectedAtts;
				std::vector<float> maxRMR;
				std::vector<bool> leftAtts;
				float sumSelectedAtts;


				int maxIndex=0,maxAtts=1;


				leftAtts.assign(noCatAtts_, true);

				getMutualInformation(xxxyDist_.xxyCounts.xyCounts, MI);

				getAttMutualInf(xxxyDist_.xxyCounts, PMI);

				//find maximal mutual information
				for(int i=1;i<noCatAtts_;i++)
					if(MI[i]>MI[maxIndex])
						maxIndex=i;

				if (verbosity >= 2) {

						printf("maximal mutual information of atts:%d\n",maxIndex);
				}

				leftAtts[maxIndex]=false;
				selectedAtts.push_back(maxIndex);
				maxRMR.push_back(MI[maxIndex]);

				while(selectedAtts.size()!=noCatAtts_)
				{

					bool first=true;
					float currentValue,maxValue;

					for(int i=0;i<noCatAtts_;i++)
					{
						if(leftAtts[i]==false)
							continue;

						sumSelectedAtts=0;
						for(int j=0;j<selectedAtts.size();j++)
								sumSelectedAtts+=PMI[selectedAtts[j]][i];

						currentValue=MI[i]/(sumSelectedAtts/selectedAtts.size()+0.01);

						if(first==true)
						{
							maxValue=currentValue;
							maxIndex=i;
							first=false;
						}
						else
						{
								if(currentValue>maxValue)
								{
									maxValue=currentValue;
									maxIndex=i;
								}
						}
					}
					leftAtts[maxIndex]=false;
					selectedAtts.push_back(maxIndex);
					maxRMR.push_back(maxValue);

				}

				maxIndex=0;
				//find maximal mRMR
				for(int i=1;i<noCatAtts_;i++)
					if(maxRMR[i]>maxRMR[maxIndex])
						maxIndex=i;

				printf("%d attributes selected from: %d\n",maxIndex+1,noCatAtts_);

				//set the other false
				for(int i=maxIndex+1;i< noCatAtts_;i++)
					active_[selectedAtts[i]]=false;

			}else if (mi_ == true || su_ == true) {
				//calculate the mutual information between each attribute and class

				std::vector<float> measure;

				if (mi_ == true)
					getMutualInformation(xxxyDist_.xxyCounts.xyCounts, measure);
				else if (su_ == true)
					getSymmetricalUncert(xxxyDist_.xxyCounts.xyCounts, measure);

				if (verbosity >= 2) {
					print(measure);
					printf("\n");
				}

				if (!order.empty()) {
					valCmpClass cmp(&measure);

					std::sort(order.begin(), order.end(), cmp);

					if (verbosity >= 2) {
						printf("the attributes order by mi:\n");
						const char * sep = "";
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							printf("%s%d", sep, order[a]);
							sep = ", ";
						}
						printf("\n");
					}
				}
				//select half of the attributes as spodes default

				unsigned int noSelected = noCatAtts_ / factor_; ///< the number of selected attributes

				//set the attribute selected or unselected
				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					if (a >= noSelected)
						active_[order[a]] = false;
				}

				if (verbosity == 2) {
					for (unsigned int i = 0; i < active_.size(); i++) {
						if (active_[i] == true)
							printf("true,");
						else
							printf("false,");
					}
					printf("\n");

				}
			} else if (chisq_ == true) {

				bool flag = true;
				double lowest;
				CategoricalAttribute attLowest;

				for (std::vector<CategoricalAttribute>::const_iterator it =
						order.begin(); it != order.end(); it++) {

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
										xxxyDist_.xxyCounts.xyCounts.getCount(a, r,
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

	}else if(pass_==2)
	{
		if(loo_==true)
		{
            const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)/2;
            const CatValue noOfModels=parentNumber*noCatAtts_;
            double minMean=1;
            CatValue minModel=0;
            if(bayesRace_==true)
            {
                if(noOfModelsThrownOut_==noOfModels-1)
                {
                    for(int i=0;i<noOfModels;i++)
                    {
                        if(modelThrownOut_[i]==false)
                        {
                                minModel=i;
                                break;
                        }
                    }
                  if(verbosity>=2)
                    printf("Only one model %d left.\n",minModel);

                }else
                {

                    for(int i=0;i<noOfModels;i++)
                    {
                        if(modelThrownOut_[i]==true)
                            continue;
                        if(meanOfModels_[i]<minMean)
                        {
                                minModel=i;
                                minMean=meanOfModels_[i];
                        }
                    }

                   if(verbosity>=2)
                        printf("%d models kept of %d models. \n",noOfModels-noOfModelsThrownOut_,noOfModels);
                }


                optChildIndex_=minModel%noCatAtts_+1;
                optParentIndex_=minModel/noCatAtts_+1;
            }
            else
            {

                for (CategoricalAttribute parent = 0; parent < parentNumber;
                        parent++) {
                    if(verbosity>=3)
                        printf("parent:%d:",parent);
                    for (CategoricalAttribute child = 0; child < noCatAtts_;
                            child++) {
                        squaredError_[parent][child] = sqrt(
                                squaredError_[parent][child]
                                        / xxxyDist_.xxyCounts.xyCounts.count);
                        if(verbosity>=3)
                            printf("%f,",squaredError_[parent][child]);
                    }
                    if(verbosity>=3)
                        printf("\n");
                }

                if(ind_==true)
                {
                    //independently select children attributes
                    orderedParent_.clear();

                    for (CatValue parent = 0; parent < parentNumber; parent++) {


                        orderedParent_.push_back(parent);
                        if(allChild_==true)
                        {
                            //keep all child attributes
                            optSquaredError_[parent]=squaredError_[parent][noCatAtts_-1];
                            indOptChildIndex_[parent]=noCatAtts_;


                            if(verbosity>=2)
                            {
                                printf("parent:%d, attributes selected:%d\n",parent,noCatAtts_);
                            }

                        }else
                        {
                            //select child attribute independently
                             unsigned int optColumn=indexOfMinVal(squaredError_[parent]);

                             optSquaredError_[parent]=squaredError_[parent][optColumn];
                             indOptChildIndex_[parent]=optColumn+1;
                             if(verbosity>=2)
                             {
                                printf("parent:%d, attributes selected:%d\n",parent,optColumn+1);

                             }
                        }
                    }

                    // sort the parent on optimal Squared Error_ in each 2de

                    if (!orderedParent_.empty()) {
                        sortAscendClass cmp(&optSquaredError_);
                        std::sort(orderedParent_.begin(), orderedParent_.end(), cmp);
                    }
                    if(verbosity==2)
                    {
                        printf("After pass 2, optimal RMSE in each 2de:\n");
                        print(optSquaredError_);
                        printf("\nAscending order:\n");
                        print(orderedParent_);
                    }
                }else if(vote_=true)
                {
					 //independently select children attributes
                    orderedParent_.clear();

                    for (CatValue parent = 0; parent < parentNumber; parent++) {
						 orderedParent_.push_back(parent);
                        //select child attribute independently
                         unsigned int optColumn=indexOfMinVal(squaredError_[parent]);

                         optSquaredError_[parent]=squaredError_[parent][optColumn];
                         indOptChildIndex_[parent]=optColumn+1;
                         if(verbosity>=2)
                         {
                            printf("parent:%d, attributes selected:%d\n",parent,optColumn+1);
                         }
                    }

					// sort the parent on optimal Squared Error_ in each 2de

                    if (!orderedParent_.empty()) {
                        sortAscendClass cmp(&optSquaredError_);
                        std::sort(orderedParent_.begin(), orderedParent_.end(), cmp);
                    }
                    if(verbosity==2)
                    {
                        printf("After pass 2, optimal RMSE in each 2de:\n");
                        print(optSquaredError_);
                        printf("\nAscending order:\n");
                        print(orderedParent_);
                    }
                }
                else
                {   //children attributes are identical

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
		}
	}else if (pass_==3)
	{
	    const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)/2;

        for (CategoricalAttribute parent = 0; parent < parentNumber; parent++)
        {
            optSquaredError_[parent] = sqrt(
                optSquaredError_[parent]
                / xxxyDist_.xxyCounts.xyCounts.count);
        }

        optParentIndex_=indexOfMinVal(optSquaredError_)+1;
        if(verbosity>=2)
        {
            printf("\nAfter pass 3, RMSE for each parent selection:\n");
            print(optSquaredError_);
            printf("\nOptimal parent: %d\n",optParentIndex_);
        }
	}

	pass_++;


}
void a2de::LOOCV2(const instance &inst)
{


	const InstanceCount totalCount = xxxyDist_.xxyCounts.xyCounts.count-1;
	const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)/2;
	const CatValue trueClass = inst.getClass();
	std::vector<double> classDist;
	classDist.assign(noClasses_,0.0);
	std::vector<double> classDistTemp;
	classDistTemp.assign(noClasses_,0.0);

	//	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
				/ ((noCatAtts_ - 1) * (noCatAtts_) / 2.0);

	CatValue delta = 0;

	xxyDist * xxydist = &xxxyDist_.xxyCounts;
	xyDist * xydist = &xxxyDist_.xxyCounts.xyCounts;

	std::vector<bool> generalizationSet;
	generalizationSet.assign(noCatAtts_, false);
//	compute the generalisation set and substitution set for
//	lazy subsumption resolution

	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					InstanceCount countOfxixj = xxydist->getCount(i,
							inst.getCatVal(i), j, inst.getCatVal(j))-1;
					InstanceCount countOfxj = xydist->getCount(j,
							inst.getCatVal(j))-1;
					InstanceCount countOfxi = xydist->getCount(i,
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
	}


	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<double> > > spodeProbs;
	spodeProbs.resize(noCatAtts_);
	fdarray<InstanceCount> xxyCount(parentNumber, noClasses_);

	// now the activeParent tab is no use for two passes a2de
	crosstab<bool> activeParent(noCatAtts_);


	for (CatValue fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++) {

		spodeProbs[fatherIndex].resize(fatherIndex);
		const CategoricalAttribute father = orderedAtts_[fatherIndex];
		const CatValue fatherVal = inst.getCatVal(father);

		for (CatValue motherIndex = 0; motherIndex < fatherIndex; motherIndex++) {

			spodeProbs[fatherIndex][motherIndex].resize(noClasses_);

			const CategoricalAttribute mother = orderedAtts_[motherIndex];
			const CatValue motherVal = inst.getCatVal(mother);

			const CategoricalAttribute parentIndex=fatherIndex*(fatherIndex-1)/2+motherIndex;
			CatValue parentCount = 0;
			for (CatValue y = 0; y < noClasses_; y++) {

				xxyCount[parentIndex][y]= xxydist->getCount(father, fatherVal, mother,
						motherVal, y);
				if(y==trueClass)
					xxyCount[parentIndex][y]--;
				parentCount +=xxyCount[parentIndex][y];
			}

			if (parentCount > 0) {
				activeParent[father][mother] = true;
				delta++;

				for (CatValue y = 0; y < noClasses_; y++) {

					// scale up by maximum possible factor to reduce risk of numeric underflow
					spodeProbs[fatherIndex][motherIndex][y] = weight[father][mother]
							* scaleFactor * mEstimate(xxyCount[parentIndex][y],
							totalCount,noClasses_ * xxxyDist_.getNoValues(father)
									* xxxyDist_.getNoValues(mother));
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


    for (std::vector<unsigned int>::const_iterator it = orderedParent_.begin(); it != orderedParent_.end(); it++)
    {

        const CatValue parentIndex=*it;
        CatValue fatherIndex,motherIndex;
        for (fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++)
        {
            if(fatherIndex*(fatherIndex-1)/2>parentIndex)
                break;
        }
        fatherIndex--;
        motherIndex=parentIndex-fatherIndex*(fatherIndex-1)/2;

        const CategoricalAttribute father = orderedAtts_[fatherIndex];
        const CatValue fatherVal = inst.getCatVal(father);
        const CategoricalAttribute mother = orderedAtts_[motherIndex];
        const CatValue motherVal = inst.getCatVal(mother);

        if (activeParent[father][mother] == true)
        {

            for (CategoricalAttribute childIndex = 0;
                    childIndex < indOptChildIndex_[parentIndex]; childIndex++)
            {
                const CategoricalAttribute child= orderedAtts_[childIndex];
                const CatValue childVal = inst.getCatVal(child);

                if(generalizationSet[child]==true)
                    continue;

                for (CatValue y = 0; y < noClasses_; y++)
                {
                    if(!generalizationSet[father]&&!generalizationSet[mother]&&!generalizationSet[child])
                    {
                        if (child != father && child != mother)
                        {
                            InstanceCount parentChildYCount =
                            xxxyDist_.getCount(child,childVal, father,fatherVal, mother,motherVal, y);
                            if(y==trueClass)
                                parentChildYCount--;

                            spodeProbs[fatherIndex][motherIndex][y] *= mEstimate(parentChildYCount,
                                    xxyCount[parentIndex][y],xxxyDist_.getNoValues(child));

                        }
                    }

                }
            }

            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] += spodeProbs[fatherIndex][motherIndex][y];
                classDistTemp[y]=classDist[y] ;
            }
            if(sum(classDistTemp)!=0)
            {
                normalise(classDistTemp);
                const double error = 1.0 - classDistTemp[trueClass];
                optSquaredError_[parentIndex] += error * error;
            }
        }
    }
}
void a2de::LOOCV(const instance &inst)
{
	count++;

	if(verbosity>=3)
		printf("instance %d\n",count);

	const InstanceCount totalCount = xxxyDist_.xxyCounts.xyCounts.count-1;
	const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)/2;
	const CatValue trueClass = inst.getClass();
	std::vector<double> classDist;
	classDist.assign(noClasses_,0.0);
	const CatValue noOfModels=parentNumber*noCatAtts_;


    if(noOfModelsThrownOut_==noOfModels-1)
        return;

	//	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
				/ ((noCatAtts_ - 1) * (noCatAtts_) / 2.0);

	CatValue delta = 0;

	xxyDist * xxydist = &xxxyDist_.xxyCounts;
	xyDist * xydist = &xxxyDist_.xxyCounts.xyCounts;

	std::vector<bool> generalizationSet;
	generalizationSet.assign(noCatAtts_, false);
//	compute the generalisation set and substitution set for
//	lazy subsumption resolution

	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					InstanceCount countOfxixj = xxydist->getCount(i,
							inst.getCatVal(i), j, inst.getCatVal(j))-1;
					InstanceCount countOfxj = xydist->getCount(j,
							inst.getCatVal(j))-1;
					InstanceCount countOfxi = xydist->getCount(i,
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
	}


	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<double> > > spodeProbs;
	spodeProbs.resize(noCatAtts_);
	fdarray<InstanceCount> xxyCount(parentNumber, noClasses_);

	// now the activeParent tab is no use for two passes a2de
	crosstab<bool> activeParent(noCatAtts_);

	for (CatValue fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++) {

		spodeProbs[fatherIndex].resize(fatherIndex);
		const CategoricalAttribute father = orderedAtts_[fatherIndex];
		const CatValue fatherVal = inst.getCatVal(father);

		for (CatValue motherIndex = 0; motherIndex < fatherIndex; motherIndex++) {

			spodeProbs[fatherIndex][motherIndex].resize(noClasses_);

			const CategoricalAttribute mother = orderedAtts_[motherIndex];
			const CatValue motherVal = inst.getCatVal(mother);

			const CategoricalAttribute parentIndex=fatherIndex*(fatherIndex-1)/2+motherIndex;
			CatValue parentCount = 0;
			for (CatValue y = 0; y < noClasses_; y++) {

				xxyCount[parentIndex][y]= xxydist->getCount(father, fatherVal, mother,
						motherVal, y);
				if(y==trueClass)
					xxyCount[parentIndex][y]--;
				parentCount +=xxyCount[parentIndex][y];
			}

			if (parentCount > 0) {
				activeParent[father][mother] = true;
				delta++;

				for (CatValue y = 0; y < noClasses_; y++) {

					// scale up by maximum possible factor to reduce risk of numeric underflow
					spodeProbs[fatherIndex][motherIndex][y] = weight[father][mother]
							* scaleFactor * mEstimate(xxyCount[parentIndex][y],
							totalCount,noClasses_ * xxxyDist_.getNoValues(father)
									* xxxyDist_.getNoValues(mother));

					if(verbosity>=3)
					{
						if(count==26&&parentIndex==16&&y==trueClass)
							printf("spodeProb:%f\n",spodeProbs[fatherIndex][motherIndex][y]);

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


    for (CatValue fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++)
    {

        const CategoricalAttribute father = orderedAtts_[fatherIndex];
        const CatValue fatherVal = inst.getCatVal(father);

        for (CatValue motherIndex = 0; motherIndex < fatherIndex; motherIndex++)
        {

            const CategoricalAttribute mother = orderedAtts_[motherIndex];
            const CatValue motherVal = inst.getCatVal(mother);

            CategoricalAttribute parentIndex=fatherIndex*(fatherIndex-1)/2+motherIndex;

            if (activeParent[father][mother] == true)
            {
                for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++)
                {

                    if(modelThrownOut_[parentIndex*noCatAtts_+childIndex]==true)
                        continue;

                    const CategoricalAttribute child= orderedAtts_[childIndex];
                    const CatValue childVal = inst.getCatVal(child);

                    for (CatValue y = 0; y < noClasses_; y++)
                    {

                        if(!generalizationSet[father]&&!generalizationSet[mother]&&!generalizationSet[child])
                        {
                            if (child != father && child != mother)
                            {
                                InstanceCount parentChildYCount =
                                xxxyDist_.getCount(child,childVal, father,fatherVal, mother,motherVal, y);
                                if(y==trueClass)
                                    parentChildYCount--;

                                spodeProbs[fatherIndex][motherIndex][y] *= mEstimate(parentChildYCount,
                                        xxyCount[parentIndex][y],xxxyDist_.getNoValues(child));
                                if(verbosity>=3)
                                {
                                    if(count==26&&parentIndex==16&&y==trueClass)
                                        printf("f:%d,m:%d,child:%d, prob:%f\n",fatherIndex,motherIndex,childIndex,mEstimate(parentChildYCount,
                                                xxyCount[parentIndex][y],xxxyDist_.getNoValues(child)));

                                }
                            }
                        }

                        if(ind_==true || vote_==true)
                        {
                            //independently select children attributes

                            classDist[y] =spodeProbs[fatherIndex][motherIndex][y];
                        }
                        else
                        {
                            //children attributes are identical
                            if(!generalizationSet[father]&&!generalizationSet[mother])
                            {
                                spodeProbsSumOnRow[childIndex][y] +=
                                    spodeProbs[fatherIndex][motherIndex][y];
                                if(verbosity>=3)
                                {
                                    if(count==26&&parentIndex<16&&childIndex==7&&y==trueClass)
                                    {
                                        printf(">>%f\n",spodeProbs[fatherIndex][motherIndex][y]);

                                    }
                                }

                            }
                            classDist[y] = spodeProbsSumOnRow[childIndex][y];
                        }
                    }

                    if(sum(classDist)!=0)
                    {
                        normalise(classDist);
                        const double error = 1.0 - classDist[trueClass];
                        squaredError_[parentIndex][childIndex] += error * error;
                        if(verbosity>=3)
                        {
                            if(parentIndex==0&&childIndex==0)
                                printf("count: %d,parent,%d, child %d,error:%f\n",count,parentIndex,childIndex,error);
                        }

                        const double previousMean=meanOfModels_[parentIndex*noCatAtts_+childIndex];
                        const double variance= varianceOfModels_[parentIndex*noCatAtts_+childIndex];
                        meanOfModels_[parentIndex*noCatAtts_+childIndex]=((count-1)*previousMean+error)/count;
                        if(count==1)
                            ;
                        else if(count==2)
                        {
                            const double nextMean=meanOfModels_[parentIndex*noCatAtts_+childIndex];
                            varianceOfModels_[parentIndex*noCatAtts_+childIndex]=(previousMean-nextMean)*(previousMean-nextMean)+(error-nextMean)*(error-nextMean);
                        }
                        else
                            varianceOfModels_[parentIndex*noCatAtts_+childIndex]=variance*(count-2)/(count-1)+(error- previousMean)*(error- previousMean)/count;
                    }
                }
            }
        }
    }


    if(bayesRace_==true)
    {
        InstanceCount n1=count,n2=count;
        double x1,x2,ss1,ss2;
        double  u1,u2,b;
        long int k;
        if(count>=2&&count%1000==0)
        {
            for(int i=0; i<noOfModels; i++)
            {
                if(modelThrownOut_[i]==true)
                    continue;

                for(int j=0; j<i; j++)
                {
                    if(modelThrownOut_[j]==true)
                        continue;
                    x1=meanOfModels_[i];
                    x2=meanOfModels_[j];
                    ss1=varianceOfModels_[i];
                    ss2=varianceOfModels_[j];
                    u1=ss1/n1, u2=ss2/n2;

                    if((u1+u2)!=0)
                    {
                        b=u1/(u1+u2);

                        k=1/(b*b/(n1-1)+(1-b)*(1-b)/(n2-1));
                        double st=alglib::studenttdistribution(k,(x1-x2)/sqrt(u1+u2));
                        if(verbosity>=3)
                            printf("the probability that model %d is less than model %d: %f.\n",i,j,st);
                        if(st<delta_)
                        {
                            modelThrownOut_[j]=true;
                            noOfModelsThrownOut_++;
                            if(verbosity>=3)
                                printf("model %d is thrown out.\n",j);
                        }
                    }
                }
            }
        }

    }
}
void a2de::classify(const instance &inst, std::vector<double> &classDist) {

	if (verbosity >= 2)
		count++;

	unsigned int check = 101;

	if (verbosity == 3 && count == check) {
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

//	 scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max()
			/ ((noCatAtts_ - 1) * (noCatAtts_) / 2.0);

//	 scale up by maximum possible factor to reduce risk of numeric underflow
//	double scaleFactor = 1;
//

//	// first to assign the spodeProbs array
	std::vector<std::vector<std::vector<double> > > spodeProbs;
	spodeProbs.resize(noCatAtts_);



	// now the activeParent tab is no use for two passes a2de
	crosstab<bool> activeParent(noCatAtts_);



	if(loo_==true)
	{
		const CatValue parentNumber=noCatAtts_*(noCatAtts_-1)/2;
		fdarray<InstanceCount> xxyCount(parentNumber, noClasses_);

		for (CatValue fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++) {

			spodeProbs[fatherIndex].resize(fatherIndex);
			const CategoricalAttribute father = orderedAtts_[fatherIndex];
			const CatValue fatherVal = inst.getCatVal(father);

			for (CatValue motherIndex = 0; motherIndex < fatherIndex; motherIndex++) {

				spodeProbs[fatherIndex][motherIndex].resize(noClasses_);

				const CategoricalAttribute mother = orderedAtts_[motherIndex];
				const CatValue motherVal = inst.getCatVal(mother);

				const CategoricalAttribute parentIndex=fatherIndex*(fatherIndex-1)/2+motherIndex;
				CatValue parentCount = 0;
				for (CatValue y = 0; y < noClasses_; y++) {

					xxyCount[parentIndex][y]= xxydist->getCount(father, fatherVal, mother,
							motherVal, y);
					parentCount +=xxyCount[parentIndex][y];
				}

				if (parentCount > 0) {
					activeParent[father][mother] = true;
					delta++;

					for (CatValue y = 0; y < noClasses_; y++) {

						// scale up by maximum possible factor to reduce risk of numeric underflow
						spodeProbs[fatherIndex][motherIndex][y] = weight[father][mother]
								* scaleFactor * mEstimate(xxyCount[parentIndex][y],
								totalCount,noClasses_ * xxxyDist_.getNoValues(father)
										* xxxyDist_.getNoValues(mother));
					}
				}
			}
		}

		if (delta == 0) {
			aodeClassify(inst, classDist, *xxydist);
			return;
		}



        bool hasParent;
        if(ind_==true)
        {
            //independently set the optimal children in each 2de
            hasParent=true;


            CatValue  parent=0;
            for (std::vector<unsigned int>::const_iterator it = orderedParent_.begin(); it != orderedParent_.end(); it++)
            {
                parent++;


                 const CatValue parentIndex=*it;
                 CatValue fatherIndex,motherIndex;
                 for (fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++)
                 {
                     if(fatherIndex*(fatherIndex-1)/2>parentIndex)
                        break;
                 }
                 fatherIndex--;
                 motherIndex=parentIndex-fatherIndex*(fatherIndex-1)/2;

                if(verbosity>=3)
                {
                    printf("parent number:%d\t father:%d\t mother:%d\n",parentIndex,fatherIndex,motherIndex);

                }
                const CategoricalAttribute father = orderedAtts_[fatherIndex];
                const CatValue fatherVal = inst.getCatVal(father);
                const CategoricalAttribute mother = orderedAtts_[motherIndex];
                const CatValue motherVal = inst.getCatVal(mother);

                if (activeParent[father][mother] == true) {

                    for (CategoricalAttribute childIndex = 0;
                                childIndex < indOptChildIndex_[parentIndex]; childIndex++)
                    {
                        const CategoricalAttribute child= orderedAtts_[childIndex];
                        const CatValue childVal = inst.getCatVal(child);

                        if(generalizationSet[child]==true)
                            continue;

                        for (CatValue y = 0; y < noClasses_; y++) {

                            if (child != father && child != mother) {
                                InstanceCount parentChildYCount =
                                        xxxyDist_.getCount(child,childVal, father,fatherVal, mother,motherVal, y);
                                spodeProbs[fatherIndex][motherIndex][y] *= mEstimate(parentChildYCount,
                                        xxyCount[parentIndex][y],xxxyDist_.getNoValues(child));
                            }
                        }
                    }

                    for (CatValue y = 0; y < noClasses_; y++) {
                        classDist[y] += spodeProbs[fatherIndex][motherIndex][y];
                    }
                }
                if(parent>=optParentIndex_)
                    break;
            }
        }else if(vote_==true)
        {

            //independently set the optimal children in each 2de

            hasParent=true;
            for (std::vector<unsigned int>::const_iterator it = orderedParent_.begin(); it != orderedParent_.end(); it++)
            {

                 const CatValue parentIndex=*it;
                 CatValue fatherIndex,motherIndex;
                 for (fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++)
                 {
                     if(fatherIndex*(fatherIndex-1)/2>parentIndex)
                        break;
                 }
                 fatherIndex--;
                 motherIndex=parentIndex-fatherIndex*(fatherIndex-1)/2;

                if(verbosity>=3)
                {
                    printf("parent number:%d\t father:%d\t mother:%d\n",parentIndex,fatherIndex,motherIndex);

                }
                const CategoricalAttribute father = orderedAtts_[fatherIndex];
                const CatValue fatherVal = inst.getCatVal(father);
                const CategoricalAttribute mother = orderedAtts_[motherIndex];
                const CatValue motherVal = inst.getCatVal(mother);

                if (activeParent[father][mother] == true) {

                    for (CategoricalAttribute childIndex = 0;
                                childIndex < indOptChildIndex_[parentIndex]; childIndex++)
                    {
                        const CategoricalAttribute child= orderedAtts_[childIndex];
                        const CatValue childVal = inst.getCatVal(child);

                        if(generalizationSet[child]==true)
                            continue;

                        for (CatValue y = 0; y < noClasses_; y++) {

                            if (child != father && child != mother) {
                                InstanceCount parentChildYCount =
                                        xxxyDist_.getCount(child,childVal, father,fatherVal, mother,motherVal, y);
                                spodeProbs[fatherIndex][motherIndex][y] *= mEstimate(parentChildYCount,
                                        xxyCount[parentIndex][y],xxxyDist_.getNoValues(child));
                            }
                        }
                    }

                    //predicting the class by each 2de, then vote
                    CatValue predictedClass=indexOfMaxVal(spodeProbs[fatherIndex][motherIndex]);

                    classDist[predictedClass]+=1;
                }

            }
        }
        else
        {// previous loo
            hasParent=false;
            for (CatValue fatherIndex = 1; fatherIndex < noCatAtts_; fatherIndex++) {

                const CategoricalAttribute father = orderedAtts_[fatherIndex];
                const CatValue fatherVal = inst.getCatVal(father);

                if(generalizationSet[father]==true)
                    continue;

                for (CatValue motherIndex = 0; motherIndex < fatherIndex; motherIndex++) {

                    const CategoricalAttribute mother = orderedAtts_[motherIndex];
                    const CatValue motherVal = inst.getCatVal(mother);
                    CategoricalAttribute parentIndex=fatherIndex*(fatherIndex-1)/2+motherIndex;

                    if(generalizationSet[mother]==true)
                        continue;

                    if(parentIndex>=optParentIndex_)
                        goto over;

                    if (activeParent[father][mother] == true) {

                        for (CategoricalAttribute childIndex = 0; childIndex < optChildIndex_; childIndex++) {


                            const CategoricalAttribute child= orderedAtts_[childIndex];
                            const CatValue childVal = inst.getCatVal(child);

                            if(generalizationSet[child]==true)
                                continue;

                            for (CatValue y = 0; y < noClasses_; y++) {

                                if (child != father && child != mother) {
                                    InstanceCount parentChildYCount =
                                            xxxyDist_.getCount(child,childVal, father,fatherVal, mother,motherVal, y);
                                    spodeProbs[fatherIndex][motherIndex][y] *= mEstimate(parentChildYCount,
                                            xxyCount[parentIndex][y],xxxyDist_.getNoValues(child));
                                }
                            }
                        }

                        hasParent=true;
                        for (CatValue y = 0; y < noClasses_; y++) {
                            classDist[y] += spodeProbs[fatherIndex][motherIndex][y];
                        }

                    }
                }
            }


        }

		//deal with the possible no parent case
over:   if(hasParent==false)
		{
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] =xxxyDist_.xxyCounts.xyCounts.p(y) ;
			}
		}
		normalise(classDist);
		return;
	}

	//normal A2DE
	for (CatValue father = 1; father < noCatAtts_; father++) {
		spodeProbs[father].resize(father);
		for (CatValue mother = 0; mother < father; mother++)
			spodeProbs[father][mother].assign(noClasses_, 0);
	}

	for (CatValue father = 1; father < noCatAtts_; father++) {

		//select the attribute according to centain measure
		if (active_[father] == false)
			continue;
		//selecct attribute for subsumption resolution
		if (generalizationSet[father])
			continue;
		const CatValue fatherVal = inst.getCatVal(father);

		for (CatValue mother = 0; mother < father; mother++) {
			//select the attribute according to centain measure
			if (active_[mother] == false)
				continue;
			//selecct attribute for subsumption resolution
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
					spodeProbs[father][mother][y] = weight[father][mother]
							* scaleFactor; // scale up by maximum possible factor to reduce risk of numeric underflow

					InstanceCount parentYCount = xxydist->getCount(father,
							fatherVal, mother, motherVal, y);
					if (empiricalMEst_) {
						spodeProbs[father][mother][y] *= empiricalMEstimate(
								parentYCount, totalCount,
								xydist->p(y) * xydist->p(father, fatherVal)
										* xydist->p(mother, motherVal));

					} else {

						spodeProbs[father][mother][y] *= mEstimate(parentYCount,
								totalCount,
								noClasses_ * xxxyDist_.getNoValues(father)
										* xxxyDist_.getNoValues(mother));
					}

					if (verbosity == 4)
						printf("%f,", spodeProbs[father][mother][y]);
				}

			}
		}

	}

	if (verbosity == 3 && count == check) {
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

	if (verbosity == 3 && count == check) {
		printf("print every prob for each parents:\n");
	}

	for (CatValue father = 2; father < noCatAtts_; father++) {

		//for a2de2, selecting must be in the most inner loop
		// because we want to select only the parents, which means we should let
		//child be any value different from parent.
		//if we select here, child can only have values in this set

		//select the attribute according to centain measure
//		if (active_[father] == false)
//			continue;

		//selecct attribute for subsumption resolution
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
//			if (father == 13 && mother == 6 && count == check)
//				printf("%u\n", check);

			// the same to father
//			if (active_[mother] == false)
//				continue;
			//selecct attribute for subsumption resolution
			if (generalizationSet[mother])
				continue;
			const CatValue motherVal = inst.getCatVal(mother);

//			CatValue parent = 0;
//			for (CatValue y = 0; y < noClasses_; y++) {
//				parent += xxydist->getCount(father, fatherVal, mother,
//						motherVal, y);
//			}
//
//			if (parent != 0) {

			//		XYSubDist xySubDist(xxxyDist_.getXYSubDist(father,fatherVal,mother,motherVal), noClasses_);
			XYSubDist xySubDist(xxySubDist.getXYSubDist(mother, motherVal),
					noClasses_);

			XYSubDist xySubDistMother(xxydist->getXYSubDist(mother, motherVal),
					noClasses_);

			for (CatValue child = 0; child < mother; child++) {
				//select the attribute according to centain measure
//					if (active_[child] == false)
//						continue;

				if (!generalizationSet[child]) {
					const CatValue childVal = inst.getCatVal(child);
					if (child != father && child != mother) {

						for (CatValue y = 0; y < noClasses_; y++) {

//								InstanceCount parentChildYCount =
//										xxxyDist_.getCount(father, fatherVal, mother,
//												motherVal, child, childVal,y);
//								InstanceCount parentYCount =
//										xxydist->getCount(father,
//												fatherVal, mother, motherVal,
//												y);
//								InstanceCount fatherYChildCount =
//										xxydist->getCount(father,
//												fatherVal, child, childVal, y);

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
									if (active_[father] == true
											&& active_[mother] == true) {
										temp1 = empiricalMEstimate(
												parentChildYCount, parentYCount,
												xydist->p(child, childVal));
										spodeProbs[father][mother][y] *= temp1;

									}
								}
								if (xxydist->getCount(father, fatherVal, child,
										childVal) > 0) {
									if (active_[father] == true
											&& active_[child] == true) {
										temp2 = empiricalMEstimate(
												parentChildYCount,
												fatherYChildCount,
												xydist->p(mother, motherVal));
										spodeProbs[father][child][y] *= temp2;

									}
								}
								if (xxydist->getCount(mother, motherVal, child,
										childVal) > 0) {
									if (active_[mother] == true
											&& active_[child] == true) {
										temp3 = empiricalMEstimate(
												parentChildYCount,
												motherYChildCount,
												xydist->p(father, fatherVal));
										spodeProbs[mother][child][y] *= temp3;

									}
								}

							} else {

								if (xxydist->getCount(father, fatherVal, mother,
										motherVal) > 0) {
									//check whether the parents is qualified or not
									if (active_[father] == true
											&& active_[mother] == true)
											//if selecting child as well
											//if(active_[father]==true&&active_[mother]==true&&active_[child]==true)
													{
										temp1 = mEstimate(parentChildYCount,
												parentYCount,
												xxxyDist_.getNoValues(child));
										spodeProbs[father][mother][y] *= temp1;

										if (verbosity == 3 && count == 1
												&& y == 0) {
											if (father == 3 && mother == 1
													&& child == 5) {
												printf("%u,%u,%u\n",
														parentChildYCount,
														parentYCount,
														xxxyDist_.getNoValues(
																child));
											}

										}
										if (verbosity == 3 && count == check
												&& y == 0) {
											printf("%u,%u,%u,%u,%f\n", father,
													mother, child, y, temp1);
										}

									}
								}
								if (xxydist->getCount(father, fatherVal, child,
										childVal) > 0) {
									if (active_[father] == true
											&& active_[child] == true)
											//if selecting child as well
											//if(active_[father]==true&&active_[child]==true&&active_[mother]==true)
													{
										temp2 = mEstimate(parentChildYCount,
												fatherYChildCount,
												xxxyDist_.getNoValues(mother));
										spodeProbs[father][child][y] *= temp2;

										if (verbosity == 3 && count == check
												&& y == 0) {
											if (father == 3 && child == 1
													&& mother == 5) {
												printf("%u,%u,%u\n",
														parentChildYCount,
														fatherYChildCount,
														xxxyDist_.getNoValues(
																mother));
											}

										}

										if (verbosity == 3 && count == check
												&& y == 0) {
											printf("%u,%u,%u,%u,%f\n", father,
													child, mother, y, temp2);
										}

									}
								}
								if (xxydist->getCount(mother, motherVal, child,
										childVal) > 0) {
									if (active_[mother] == true
											&& active_[child] == true)
											//if selecting child as well
											//if(active_[mother]==true&&active_[child]==true&&active_[father]==true)
													{
										temp3 = mEstimate(parentChildYCount,
												motherYChildCount,
												xxxyDist_.getNoValues(father));
										spodeProbs[mother][child][y] *= temp3;

										if (verbosity == 3 && count == check
												&& y == 0) {
											if (mother == 3 && child == 1
													&& father == 5) {
												printf("%u,%u,%u\n",
														parentChildYCount,
														motherYChildCount,
														xxxyDist_.getNoValues(
																father));
											}

										}
										if (verbosity == 3 && count == check
												&& y == 0) {
											printf("%u,%u,%u,%u,%f\n", mother,
													child, father, y, temp3);
										}

									}
								}

							}

						}
					}
				}

			}

		}

	}

	for (CatValue father = 1; father < noCatAtts_; father++) {
		if (active_[father] == false)
			continue;
		for (CatValue mother = 0; mother < father; mother++) {
			if (active_[mother] == false)
				continue;
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] += spodeProbs[father][mother][y];

			}
		}
	}

	if (verbosity >= 3) {
		if (count == check) {
			printf("distribute of instance %u for each parent.\n", count);
			for (CatValue father = 1; father < noCatAtts_; father++) {
				if (active_[father] == false)
					continue;
				for (CatValue mother = 0; mother < father; mother++) {
					if (active_[mother] == false)
						continue;
					printf("%u,%u: ", father, mother);
					print(spodeProbs[father][mother]);
					printf("\n");
				}
			}

		}
	}

	if (verbosity == 4) {
		if (count == check) {
			printf("the class dist of instance %u before normalizing:\n",
					count);
			for (unsigned int i = 0; i < classDist.size(); i++)
				printf("%0.20f,", classDist[i]);
			printf("\n");
		}
	}

	normalise(classDist);

	if (verbosity >= 3) {

		if (count == check) {
			printf("the class dist of instance %u:\n", count);
			for (unsigned int i = 0; i < classDist.size(); i++)
				printf("%0.14f,", classDist[i]);
			printf("\n");
		}
	}
}

void a2de::aodeClassify(const instance &inst, std::vector<double> &classDist,
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

			if (active_[parent]) {
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
	}

	if (delta == 0) {
		count++;
		if (verbosity == 2)
			printf("nb is called for %u times\n", count);
		nbClassify(inst, classDist, xxyDist_.xyCounts);
		return;
	}

	for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
		//
		//		std::vector<std::vector<std::vector<double> > > * parentsProbs =
		//				&xxyDist_.condiProbs[x1][x1Val];

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

	count++;
	if (verbosity == 2 && count == 1) {
		printf("the class distribution is :\n");
		print(classDist);
		printf("\n");

	}
}
void a2de::nbClassify(const instance &inst, std::vector<double> &classDist,
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

