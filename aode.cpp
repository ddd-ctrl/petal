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

#include "aode.h"
#include <assert.h>
#include "utils.h"
#include <algorithm>
#include "correlationMeasures.h"
#include "globals.h"
#include "utils.h"
#include "crosstab.h"
#include "learnerRegistry.h"


static LearnerRegistrar registrar("aode", constructor<aode>);


aode::aode(char* const *& argv, char* const * end) {
	name_ = "AODE";

	directRank_=false;

	trainSize_ = 0;


	testSetSize_ = std::numeric_limits<InstanceCount>::max();   // by default include all the training data in the test set for pas 2
	testSetSizeP_ = std::numeric_limits<InstanceCount>::max();   // by default include all the training data in the test set for pas 1

	weighting_=WT_UNIFORM;
	minCount = 100;
	subsumptionResolution = false;
	selected = false;
	su_ = false;
	mi_ = false;
	acmi_=false;
	mRMR_=false;
	chisq_ = false;
	Jmi=false;
	Icap=false;
	expCritVal_=2; //default value

	empiricalMEst_ = false;
	empiricalMEst2_ = false;

	correlationFilter_=false;
	useThreshold_=false;
	threshold_=0;
	factor_=1.0;
	loo_=false;
	raw_=false;
	sample4prmtr_=false;



	useAttribSelec_=false;

	attribSelected_=0;


	// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
 		} else if (streq(argv[0] + 1, "loo")) {
			loo_ = true;
		} else if (streq(argv[0] + 1, "raw")) {
			raw_ = true;
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
		} else if (streq(argv[0] + 1, "sub")) {
			subsumptionResolution = true;
		} else if (argv[0][1] == 'n') {
			getUIntFromStr(argv[0] + 2, minCount, "n");
		} else if (streq(argv[0] + 1, "wmi")) {
			weighting_ = WT_MI;
		} else if (streq(argv[0] + 1, "wlh")) {
			weighting_ = WT_LH;
		} else if (streq(argv[0] + 1, "selective")) {
			selected = true;
		} else if (streq(argv[0] + 1, "acmi")) {
			selected = true;
			acmi_ = true;
		} else if (streq(argv[0] + 1, "jmi")) {
			selected=true;
			Jmi=true;
		}else if (streq(argv[0] + 1, "icap")) {
			selected=true;
			Icap=true;
		}else if (streq(argv[0] + 1, "mi")) {
			selected = true;
			mi_ = true;
		} else if (streq(argv[0] + 1, "mrmr")) {
			mRMR_ = true;
		} else if (argv[0][1] == 'a') {
			getUIntFromStr(argv[0] + 2, attribSelected_, "a");
			useAttribSelec_=true;
		} else if (argv[0][1] == 'f') {
			unsigned int factor;
			getUIntFromStr(argv[0] + 2, factor, "f");
			factor_ = factor / 10.0;
			while (factor_ >= 1)
				factor_ /= 10;
		}else if (streq(argv[0] + 1, "su")) {
			selected = true;
			su_ = true;
		}else if (streq(argv[0] + 1, "cf")) {
			correlationFilter_ = true;
		} else if (argv[0][1] == 't') {
			unsigned int thres;
			getUIntFromStr(argv[0] + 2, thres, "threshold");
			threshold_=thres/10.0;
			while(threshold_>=1)
				threshold_/=10;
			useThreshold_=true;
		} else if (streq(argv[0] + 1, "chisq")) {
			selected = true;
			chisq_ = true;
		} else if (argv[0][1] == 'e') {
			getUIntFromStr(argv[0] + 2, expCritVal_, "expOfCriVal");
		}
		else {
			error("Aode does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}

	if (selected == true) {
		if (mi_ == false && su_ == false && chisq_ == false&& mRMR_==false &&Icap==false)
			chisq_ = true;
	}

	if(mRMR_ == true&&loo_ == false)
	{
		 selected =true;
	}
	if(loo_==true && directRank_==false&&raw_==false&& mRMR_==false &&Jmi==false &&Icap==false)
	{
		mi_=true;
	}
	printf("args in running is:Jmi=%d loo=%d ICAP=%d MI=%d chisq=%d\n",Jmi,loo_,Icap,mi_,chisq_);
}

aode::aode(const aode& l) {
  name_ = l.name_;
  directRank_=l.directRank_;
  trainSize_ = l.trainSize_;
  testSetSize_ = l.testSetSize_;
  testSetSizeP_=l.testSetSizeP_;
  sample4prmtr_=l.sample4prmtr_;
  weighting_=l.weighting_;
  minCount = l.minCount;
  subsumptionResolution = l.subsumptionResolution;
  selected = l.selected;
  su_ = l.su_;
  mi_ = l.mi_;
  acmi_=l.acmi_;
  mRMR_=l.mRMR_;
  Icap=l.Icap;
  Jmi=l.Jmi;

  chisq_ = l.chisq_;
  expCritVal_=l.expCritVal_;
  empiricalMEst_ = l.empiricalMEst_;
  empiricalMEst2_ = l.empiricalMEst2_;
  correlationFilter_=l.correlationFilter_;
  useThreshold_=l.useThreshold_;
  threshold_=l.threshold_;
  factor_=l.factor_;
  loo_=l.loo_;
  raw_=l.raw_;
  useAttribSelec_=l.useAttribSelec_;
  attribSelected_=l.attribSelected_;
}

learner* aode::clone() const {
  return new aode(*this);
}

aode::~aode(void) {
}

void aode::getCapabilities(capabilities &c) {
	c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void aode::reset(InstanceStream &is) {
	xxyDist_.reset(is);
	inactiveCnt_ = 0;
	count_=0;
	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
	noChild_.resize(noCatAtts_,noCatAtts_);

	//weight_.assign(noCatAtts_, 1.0);
	weight_.assign(noCatAtts_, 1.0/noCatAtts_);

	instanceStream_ = &is;
	pass_ = 1;

	trainSize_ = 0;

	active_.assign(noCatAtts_, true);

	squaredError1D_.assign(noCatAtts_,0.0);

	//delete and initialise the rmse vector
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].clear();
	}
	squaredError_.clear();
	squaredError_.resize(noCatAtts_);
	for (CategoricalAttribute x =0; x <squaredError_.size(); x++) {
		squaredError_[x].assign(noCatAtts_,0.0);
	}
}

void aode::initialisePass() {
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
}

void aode::train(const instance &inst) {

	if (pass_ == 1){
		if(sample4prmtr_==true)
		{
			if ((testSetSizeP_-testSetSoFarP_) / static_cast<double>(trainSize_-seenSoFarP_++) < rand_())
				return;  // ignore all but testSetSize_ randomly selected cases
			testSetSoFarP_++;
			xxyDist_.update(inst);
		}else
		{
			xxyDist_.update(inst);
			trainSize_++;
		}
	}
	else {
		assert(pass_ == 2);

		if(weighting_==WT_LH)
		{
			calLikelihood(inst);
		}
		else
		{
			if ((testSetSize_-testSetSoFar_) / static_cast<double>(trainSize_-seenSoFar_++) < rand_())
				return;  // ignore all but testSetSize_ randomly selected cases
			testSetSoFar_++;
			LOOCV(inst);
		}
	}
}

void aode::calLikelihood(const instance &inst) {

	//[1]李楠,姜远,周志华.基于模型似然的超1-依赖贝叶斯分类器集成方法[J].模式识别与人工智能,2007,20(06):727-731.

	const CatValue trueClass = inst.getClass();
 	double likelihood;

	for(CategoricalAttribute parent = 0; parent < noCatAtts_; parent++) {
		const CatValue parentVal = inst.getCatVal(parent);

		likelihood=xxyDist_.xyCounts.p(trueClass);
		likelihood*=xxyDist_.xyCounts.p(parent,parentVal,trueClass);


		for(CategoricalAttribute child = 0; child < noCatAtts_; child++) {
			if(child==parent)
				continue;

			const CatValue childVal = inst.getCatVal(child);
			likelihood*=xxyDist_.p(child,childVal,parent,parentVal,trueClass);

		}
		weight_[parent]+=likelihood;
	}
}


/// true iff no more passes are required. updated by finalisePass()
bool aode::trainingIsFinished() {
	if (loo_==true||weighting_==WT_LH)
		return pass_ > 2;
	else
		return pass_ > 1;
}

// creates a comparator for two attributes based on their
//relative value with the class,such as mutual information, symmetrical uncertainty
//This will sort the attributes in descending order

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
class valCmpClassAscend {
public:
	valCmpClassAscend(std::vector<float> *s) {
		val = s;
	}

	bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
		return (*val)[a] < (*val)[b];
	}

private:
	std::vector<float> *val;
};


int aode::getNextElement(std::vector<CategoricalAttribute> &order,CategoricalAttribute ca, unsigned int noSelected) {
	CategoricalAttribute c = ca + 1;
	while (active_[order[c]] == false&&c < noSelected)
		c++;
	if (c < noSelected)
		return c;
	else
		return -1;
}

void aode::finalisePass() {

	if(pass_==1)
	{
		if (loo_==true) {
			if(mi_==true)
			{
				orderedAtts.clear();
				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					orderedAtts.push_back(a);
				}
				//get the mutual information to rank the attrbiutes
				std::vector<float> measure;
				getMutualInformation(xxyDist_.xyCounts, measure);

				// sort the attributes on mutual information with the class

				if (!orderedAtts.empty()) {
					valCmpClass cmp(&measure);
					std::sort(orderedAtts.begin(), orderedAtts.end(), cmp);

					if (verbosity >= 3) {
						printf("The order of attributes ordered by the measure:\n");
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							printf("%d:\t%f\t%u\n",  orderedAtts[a],measure[orderedAtts[a]],instanceStream_->getNoValues(orderedAtts[a]));
						}
					}
				}

			}
			else if(directRank_==true){

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

				//this is added
				orderedAtts.clear();
				orderedAtts.push_back(first);
				seletedAttributes[first]=true;

				while(orderedAtts.size()<noCatAtts_)
				{
					maxFirst=true;
					for(CategoricalAttribute a = 0; a < noCatAtts_; a++) {

						if(seletedAttributes[a]==false)
						{
							minimalCMI[a]=cmiac[a][orderedAtts[0]];
							for (CategoricalAttribute i = 1; i < orderedAtts.size(); i++) {
								if( cmiac[a][orderedAtts[i]]< minimalCMI[a] )
								{
									minimalCMI[a] =cmiac[a][orderedAtts[i]];
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
					orderedAtts.push_back(maxIndex);
					seletedAttributes[maxIndex]=true;
				}

				if (verbosity >= 2) {
					const char * sep = "";
					printf("The order of attributes:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						printf("%s%d",sep, orderedAtts[a] );
						sep = ", ";
					}
					printf("\n");
				}
			}else if(mRMR_==true){
				//Peng's mRMR to select the attribute set by maximal relivance and minimal redundency
				std::vector<float> MI;
				crosstab<float> PMI(noCatAtts_);
				std::vector<float> maxRMR;
				std::vector<bool> leftAtts;
				float sumSelectedAtts;

				int maxIndex=0,maxAtts=1;

				leftAtts.assign(noCatAtts_, true);

				getMutualInformation(xxyDist_.xyCounts, MI);

				getAttMutualInf(xxyDist_, PMI);


				orderedAtts.clear();
				//find maximal mutual information
				for(int i=1;i<noCatAtts_;i++)
					if(MI[i]>MI[maxIndex])
						maxIndex=i;

				if (verbosity >= 2) {

						printf("maximal mutual information of atts:%d\n",maxIndex);
				}

				leftAtts[maxIndex]=false;
				orderedAtts.push_back(maxIndex);
				maxRMR.push_back(MI[maxIndex]);

				while(orderedAtts.size()!=noCatAtts_)
				{

					bool first=true;
					float currentValue,maxValue;

					for(int i=0;i<noCatAtts_;i++)
					{
						if(leftAtts[i]==false)
							continue;

						sumSelectedAtts=0;
						for(int j=0;j<orderedAtts.size();j++)
								sumSelectedAtts+=PMI[orderedAtts[j]][i];

						currentValue=MI[i]/(sumSelectedAtts/orderedAtts.size()+0.01);

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
					orderedAtts.push_back(maxIndex);
					maxRMR.push_back(maxValue);
				}


			}
			else if(Jmi==true)
			{
                //Joint mutual information  by Meyer(2008) and Gavin Brown (2012)
                orderedAtts.clear();
				std::vector<float> measure;
                getMutualInformation(xxyDist_.xyCounts, measure);
				printf("JMI is in Ruinning\n");
                crosstab<float> PMI(noCatAtts_);
                getPairMutualInf(xxyDist_, PMI);

                int max_ind=indexOfMaxVal(measure);
                orderedAtts.push_back(max_ind);
                if(verbosity>=2)
                {
                    printf("%f,%d\n",measure[max_ind],max_ind);
                }

                float J,Jmax;
                bool first;


                while(orderedAtts.size()<noCatAtts_)
                {
                    first=true;
                    unsigned k;
                    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {

                        //search if a has been added in orderedAtts_
                        for( k=0;k<orderedAtts.size();k++)
                        {
                            if(a==orderedAtts[k])
                                break;
                        }
                        //a has already been added, skip
                        if(k<orderedAtts.size())
                            continue;

                        J=0;
                        for(CategoricalAttribute s=0;s<orderedAtts.size();s++)
                        {
                            J+=PMI[orderedAtts[s]][a];
                        }

                        if(first==true)
                        {
                            first=false;
                            Jmax=J;
                            max_ind=a;
                            if(verbosity>=3)
                            {
                                printf("==%f,%d\n",Jmax,max_ind);
                            }

                        }
                        else{

                            if(J>Jmax){
                                Jmax=J;
                                max_ind=a;
                            }
                            if(verbosity>=3)
                            {
                                printf("==%f,%d\n",Jmax,max_ind);
                            }
                        }
                    }
                    if(verbosity>=2)
                    {
                        printf("%f,%d\n",Jmax,max_ind);
                    }
                    orderedAtts.push_back(max_ind);
                }
            }
            else if(Icap==true)
			{
                //Joint mutual information  by Meyer(2008) and Gavin Brown (2012)
                orderedAtts.clear();
				std::vector<float> measure;
                getMutualInformation(xxyDist_.xyCounts, measure);
				crosstab<float> Att_measure(noCatAtts_);
				getAttMutualInf(xxyDist_,Att_measure);
                crosstab<float> CMI(noCatAtts_);
                getCondMutualInf(xxyDist_, CMI);
				
                int max_ind=indexOfMaxVal(measure);
                orderedAtts.push_back(max_ind);
                if(verbosity>=2)
                {
                    printf("%f,%d\n",measure[max_ind],max_ind);
                }

                float J,Jmax;
                bool first;


                while(orderedAtts.size()<noCatAtts_)
                {
                    first=true;
                    unsigned k;
                    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {

                        //search if a has been added in orderedAtts_
                        for( k=0;k<orderedAtts.size();k++)
                        {
                            if(a==orderedAtts[k])
                                break;
                        }
                        //a has already been added, skip
                        if(k<orderedAtts.size())
                            continue;

                        J=measure[a];
                        for(CategoricalAttribute s=0;s<orderedAtts.size();s++)
                        {
                            J-=max(0.0f,Att_measure[a][orderedAtts[s]]-CMI[a][orderedAtts[s]]);
                        }

                        if(first==true)
                        {
                            first=false;
                            Jmax=J;
                            max_ind=a;
                            if(verbosity>=3)
                            {
                                printf("==%f,%d\n",Jmax,max_ind);
                            }

                        }
                        else{

                            if(J>Jmax){
                                Jmax=J;
                                max_ind=a;
                            }
                            if(verbosity>=3)
                            {
                                printf("==%f,%d\n",Jmax,max_ind);
                            }
                        }
                    }
                    /*if(verbosity>=2)
                    {
                        printf("%f,%d\n",Jmax,max_ind);
                    }*/
                    orderedAtts.push_back(max_ind);
                }
            }
			else
			{
				assert(raw_==true);
				orderedAtts.clear();
				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					orderedAtts.push_back(a);
				}
			}

		}

		if (weighting_==WT_MI) {
			weight_.assign(noCatAtts_, 0);
			getMutualInformation(xxyDist_.xyCounts, weight_);

			if (verbosity >= 3) {
				for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
					printf("%f\n", weight_[a]);
				}
			}
		}
		else if(weighting_==WT_LH)
		{
			weight_.assign(noCatAtts_, 0);
		}


		if (selected) {

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

				getMutualInformation(xxyDist_.xyCounts, MI);

				getAttMutualInf(xxyDist_, PMI);

				//find maximal mutual information
				for(int i=1;i<noCatAtts_;i++)
					if(MI[i]>MI[maxIndex])
						maxIndex=i;

				if (verbosity >= 2) {

						printf("attribute sequence:\n%d",maxIndex);
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

                    if (verbosity >= 2) {

                            printf(",%d",maxIndex);
                    }

				}

				maxIndex=0;
				//find maximal mRMR
				for(int i=1;i<noCatAtts_;i++)
					if(maxRMR[i]>maxRMR[maxIndex])
						maxIndex=i;

				printf("\n%d attributes selected from: %d\n",maxIndex+1,noCatAtts_);

				//set the other false
				for(int i=maxIndex+1;i< noCatAtts_;i++)
					active_[selectedAtts[i]]=false;

			}else if (mi_ == true || su_ == true || acmi_==true ) {
				//calculate the symmetrical uncertainty between each attribute and class
				std::vector<float> measure;
				crosstab<float> acmi(noCatAtts_);

				if (mi_ == true)
					getMutualInformation(xxyDist_.xyCounts, measure);
				else if (su_ == true)
					getSymmetricalUncert(xxyDist_.xyCounts, measure);
				else if (acmi_ == true) {
					getAttClassCondMutualInf(xxyDist_, acmi);
					measure.assign(noCatAtts_, 0.0);

					for (CategoricalAttribute x1 = 1; x1 < noCatAtts_; x1++) {
						for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

							// the way to sum acmi has been corrected.
							measure[x1] += acmi[x1][x2];
							measure[x2] += acmi[x2][x1];

							if(verbosity>=4)
							{
								if(x1==2)
								printf("%u,",x2);
								if(x2==2)
									printf("%u,",x1);
							}
						}


					}
				}



				if (verbosity >= 2) {
					if (mi_ == true)
						printf("Selecting according to mutual information:\n");
					else if (su_ == true)
						printf("Selecting according to symmetrical uncertainty:\n");
					else if (acmi_ == true)
						printf("Selecting according to attribute and class conditional mutual information:\n");
					print(measure);
					printf("\n");
				}

				// sort the attributes on symmetrical uncertainty with the class

				if (!order.empty()) {
					valCmpClass cmp(&measure);

					std::sort(order.begin(), order.end(), cmp);

					if (verbosity >= 2) {

						printf("The order of attributes ordered by the measure:\n");
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							printf("%d:\t%f\t%u\n",  order[a],measure[order[a]],instanceStream_->getNoValues(order[a]));
						}
						printf("\n");

					}
				}

				//select half of the attributes as spodes default

				unsigned int noSelected; ///< the number of selected attributes


				// set the attribute selected or unselected

				if (useThreshold_ == true) {
					noSelected = noCatAtts_;
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						if (measure[a] < threshold_) {
							active_[a] = false;
							noSelected--;
						}

					}
					if (verbosity >= 2) {
						const char * sep = "";
						printf(
								"The attributes being selected according to the threshold:\n");
						for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
							if (measure[a] >= threshold_) {
								printf("%s%d", sep, order[a]);
								sep = ", ";
							}
						}
						printf("\n");
					}
				} else if (useAttribSelec_ == true) {
					noSelected = 1;
					if (attribSelected_ >= noCatAtts_)
						error("the attribute is out of range!\n");

					for (CategoricalAttribute a = 0; a <noCatAtts_; a++) {
						if (a != attribSelected_) {
							active_[a] = false;
						}
					}
					printf(
							"The user specified attribute is :  %u\n",
							attribSelected_);

				}
				else {

					noSelected = static_cast<unsigned int>(noCatAtts_ * factor_);

					for (CategoricalAttribute a = noSelected; a < noCatAtts_; a++) {
						active_[order[a]] = false;
					}

					if (verbosity >= 2) {
						const char * sep = "";
						printf(
								"The attributes being selected according to the number:\n");
						for (CategoricalAttribute a = 0; a < noSelected; a++) {
							printf("%s%d", sep, order[a]);
							sep = ", ";
						}
						printf("\n");
					}

				}

				if (correlationFilter_ == true) {
					if (!(su_ == true && mi_ == false)) {
						printf(
								"Correlation filter can only be used with symmetrical uncertainty!\n");
						return;
					}
					int Fp = 0;
					int Fq;

					do {
						Fq = getNextElement(order, Fp, noSelected);

						if (Fq != -1) {
							do {

								double SUpq = getSymmetricalUncert(xxyDist_,
										order[Fp], order[Fq]);

								if (SUpq >= measure[order[Fq]]) {
									active_[order[Fq]] = false;
								}
								Fq = getNextElement(order, Fq, noSelected);

							} while (Fq != -1);

						}
						Fp = getNextElement(order, Fp, noSelected);
					} while (Fp != -1);

					printf(
							"The following attributes have been selected by correlation filter:\n");
					for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
						if (active_[order[a]])
							printf("%d,%f\t", order[a], measure[order[a]]);
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
								tab[r * cols + c] += xxyDist_.xyCounts.getCount(a,
										r, c);

								if (verbosity >= 2) {

									if (a == 2) {
										printf("%d ", tab[r * cols + c]);
									}

								}

							}
							if (verbosity >= 2) {
								if (a == 2) {
									printf("\n");
								}
							}
						}

						//double critVal = 0.05 / noCatAtts_;
						//double critVal = 0.0000000005 / noCatAtts_;

						//double critVal =5*pow(0.1,(int)expCritVal_) / noCatAtts_;

						double critVal =5*pow(0.1,(int)expCritVal_);

						double chisqVal = chiSquare(tab, rows, cols);

						if (verbosity >= 2){

							printf("the chi-square value of attribute %s: %40.40f\n",instanceStream_->getCatAttName(a), chisqVal);

						}

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
		if(weighting_==WT_LH)
		{
			normalise(weight_);
		}

		if(loo_==true)
		{
			for (CategoricalAttribute parent = 0; parent < noCatAtts_;
					parent++) {
				for (CategoricalAttribute child = 0; child < noCatAtts_;
						child++) {
					squaredError_[parent][child] = sqrt(
							squaredError_[parent][child]
									/ xxyDist_.xyCounts.count);
				}
			}

			optParentIndex_=0;
			optChildIndex_=indexOfMinVal(squaredError_[optParentIndex_]);
			double minVal=squaredError_[optParentIndex_][optChildIndex_];

			for (CatValue parent = 1; parent < noCatAtts_; parent++) {
				unsigned int optRow=indexOfMinVal(squaredError_[parent]);
				double  minValRow=squaredError_[parent][optRow];
				if(minVal>minValRow)
				{
					optParentIndex_=parent;
					optChildIndex_=optRow;
					minVal=minValRow;
				}
			}
			optParentIndex_++;
			optChildIndex_++;
 			printf("The best model is:(%u,%u) in (%u,%u).\n",optParentIndex_,optChildIndex_,noCatAtts_,noCatAtts_);

			if(verbosity>=4)
			{
				for (CatValue parent = 0; parent < noCatAtts_; parent++) {
					print(squaredError_[parent]);
					printf("\n");
				}
			}
		}
	}
	++pass_;
}

void aode::LOOCV(const instance &inst)
{
	const InstanceCount totalCount = xxyDist_.xyCounts.count-1;
	std::vector<double> classDist;

	const CatValue trueClass = inst.getClass();

	// scale up by maximum possible factor to reduce risk of numeric underflow
	double scaleFactor = std::numeric_limits<double>::max() / noCatAtts_;
	// scaleFactor =1;

	CatValue delta = 0;

	//try to increase the efficiency

	fdarray<double> spodeProbs(noCatAtts_, noClasses_);
	fdarray<InstanceCount> xyCount(noCatAtts_, noClasses_);
	std::vector<bool> active(noCatAtts_, false);


	std::vector<bool> generalizationSet;
	generalizationSet.assign(noCatAtts_, false);
//	compute the generalisation set and substitution set for
//	lazy subsumption resolution
	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			const CatValue iVal = inst.getCatVal(i);
			const InstanceCount countOfxi = xxyDist_.xyCounts.getCount(i, iVal)-1;

			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					const CatValue jVal = inst.getCatVal(j);
					const InstanceCount countOfxixj = xxyDist_.getCount(i, iVal,
							j, jVal)-1;
					const InstanceCount countOfxj = xxyDist_.xyCounts.getCount(
							j, jVal)-1;

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


	// initial spode assignment of joint probability of parent and class
	for (CatValue parentIndex = 0; parentIndex < noCatAtts_; parentIndex++) {

		CategoricalAttribute parent = orderedAtts[parentIndex];
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

		if (active_[parent]) {
			if (parentCount > 0) {
				delta++;
				active[parent] = true;
//					//need to modify for loocv
//					if (empiricalMEst_) {
//						for (CatValue y = 0; y < noClasses_; y++) {
//							spodeProbs[parentIndex][y] = weight[parent]
//									* empiricalMEstimate(xyCount[parent][y],
//											totalCount,	xxyDist_.xyCounts.p(y)
//													* xxyDist_.xyCounts.p(parent, parentVal))
//									* scaleFactor;
//						}
//					} else

				{
					for (CatValue y = 0; y < noClasses_; y++) {
						spodeProbs[parentIndex][y] = weight_[parent]
								* mEstimate(xyCount[parent][y], totalCount,
										noClasses_* xxyDist_.getNoValues(parent))
								* scaleFactor;

					}
				}
			}
		}
	}

	if (delta == 0) {
		printf("there are no eligible parents.\n");
		return;
	}

	std::vector<std::vector<std::vector<double> > > model;
	model.resize(noCatAtts_);


	for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
			parentIndex++) {

		CategoricalAttribute parent = orderedAtts[parentIndex];
		model[parentIndex].resize(noCatAtts_);

		if (active[parent] == true) {

			const CatValue parentVal = inst.getCatVal(parent);
			for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_; childIndex++) {

				CategoricalAttribute child= orderedAtts[childIndex];

				model[parentIndex][childIndex].resize(noClasses_);
				CatValue childVal = inst.getCatVal(child);


				for (CatValue y = 0; y < noClasses_; y++) {



					if(!generalizationSet[parent]&&!generalizationSet[child])
					{
						if (child != parent) {
							InstanceCount x1x2yCount;
							if (y == trueClass) {
								x1x2yCount = xxyDist_.getCount(parent,
										parentVal, child, childVal, y) - 1;
							} else {
								x1x2yCount = xxyDist_.getCount(parent,
										parentVal, child, childVal, y);
							}

							spodeProbs[parentIndex][y] *= mEstimate(
									x1x2yCount, xyCount[parent][y],
									xxyDist_.getNoValues(child));

						}

					}
					model[parentIndex][childIndex][y] = spodeProbs[parentIndex][y];
				}
			}
		}
	}
	if (verbosity >= 4) {
		printf("true class is %u\n", trueClass);
	}

//	for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_;
//			childIndex++) {
//
//		std::vector<double> spodeProbsSumOnRow;
//		spodeProbsSumOnRow.resize(noClasses_, 0.0);
//
//		for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
//				parentIndex++) {
//
//			CategoricalAttribute parent = orderedAtts[parentIndex];
//			if (active[parent] == true) {
//
//				for (CatValue y = 0; y < noClasses_; y++) {
//					spodeProbsSumOnRow[y] += model[parentIndex][childIndex][y];
//					classDist[parentIndex][y] = spodeProbsSumOnRow[y];
//				}
//				normalise(classDist[parentIndex]);
//				const double error = 1.0 - classDist[parentIndex][trueClass];
//				squaredError_[parentIndex][childIndex] += error * error;
//			}
//		}
//	}

	classDist.assign(noClasses_,0.0);

	std::vector<std::vector<double> >spodeProbsSumOnRow;
	spodeProbsSumOnRow.resize(noCatAtts_);
	for (CatValue y = 0; y < noCatAtts_; y++) {
		spodeProbsSumOnRow[y].assign(noClasses_,0.0);
	}

	for (CategoricalAttribute parentIndex = 0; parentIndex < noCatAtts_;
			parentIndex++) {
		CategoricalAttribute parent = orderedAtts[parentIndex];

		if (active[parent] == true) {

			for (CategoricalAttribute childIndex = 0; childIndex < noCatAtts_;
					childIndex++) {

				for (CatValue y = 0; y < noClasses_; y++) {

					if(!generalizationSet[parent])
					{
						spodeProbsSumOnRow[childIndex][y] +=
								model[parentIndex][childIndex][y];
					}
					classDist[y] =
							spodeProbsSumOnRow[childIndex][y];
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

	if(verbosity>=4)
	{
		for (CatValue parent = 0; parent < noCatAtts_; parent++) {
			print(squaredError_[parent]);
			printf("\n");
		}
	}

}


void aode::classify(const instance &inst, std::vector<double> &classDist) {

	if(verbosity>=3)
		count_++;
	std::vector<bool> generalizationSet;
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


	generalizationSet.assign(noCatAtts_, false);
//	compute the generalisation set and substitution set for
//	lazy subsumption resolution
	if (subsumptionResolution == true) {
		for (CategoricalAttribute i = 1; i < noCatAtts_; i++) {
			const CatValue iVal = inst.getCatVal(i);
			const InstanceCount countOfxi = xxyDist_.xyCounts.getCount(i, iVal);

			for (CategoricalAttribute j = 0; j < i; j++) {
				if (!generalizationSet[j]) {
					const CatValue jVal = inst.getCatVal(j);
					const InstanceCount countOfxixj = xxyDist_.getCount(i, iVal,
							j, jVal);
					const InstanceCount countOfxj = xxyDist_.xyCounts.getCount(
							j, jVal);

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

	if (verbosity >= 4) {
		for (CatValue i = 0; i < noCatAtts_; i++) {
			printf("%f\n", weight_[i]);
		}
	}


	if(loo_==true)
	{

		for (CatValue parentIndex = 0; parentIndex < optParentIndex_; parentIndex++) {


			CategoricalAttribute parent ;

			parent= orderedAtts[parentIndex];

			if(generalizationSet[parent]==true)
				continue;

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
							spodeProbs[parentIndex][y] = weight_[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] = weight_[parent]
									* mEstimate(xyCount[parent][y], totalCount,
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


		}

		if (delta == 0) {
			nbClassify(inst, classDist, xxyDist_.xyCounts);
			return;
		}


		for (CatValue parentIndex = 0; parentIndex < optParentIndex_; parentIndex++) {

			CategoricalAttribute parent = orderedAtts[parentIndex];
			const CatValue parentVal = inst.getCatVal(parent);
			if(generalizationSet[parent]==true)
				continue;

			if (active[parent] == true) {

				for (CategoricalAttribute childIndex = 0;
						childIndex < optChildIndex_; childIndex++) {

					CategoricalAttribute child= orderedAtts[childIndex];
					CatValue childVal = inst.getCatVal(child);
					if(generalizationSet[child]==true)
						continue;

					if (child != parent) {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parentIndex][y] *= xxyDist_.p(child,
									childVal, parent, parentVal, y);
						}
					}
				}
			}
		}

		bool hasParent=false;
		for (CatValue parent = 0; parent < optParentIndex_; parent++) {

			if(generalizationSet[orderedAtts[parent]]==true)
				continue;

			if (active[orderedAtts[parent]]) {
				hasParent=true;
				for (CatValue y = 0; y < noClasses_; y++) {
					classDist[y] += spodeProbs[parent][y];
				}
			}
		}
		//deal with the possible no parent case
		if(hasParent==false)
		{
			for (CatValue y = 0; y < noClasses_; y++) {
				classDist[y] =xxyDist_.xyCounts.p(y) ;
			}
		}
		normalise(classDist);
		return;
	}




	//normal AODE
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
							spodeProbs[parent][y] = weight_[parent]
									* empiricalMEstimate(xyCount[parent][y],
											totalCount,
											xxyDist_.xyCounts.p(y)
													* xxyDist_.xyCounts.p(
															parent, parentVal))
									* scaleFactor;
						}
					} else {
						for (CatValue y = 0; y < noClasses_; y++) {
							spodeProbs[parent][y] = weight_[parent]
									* mEstimate(xyCount[parent][y], totalCount,
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
			const InstanceCount x1Count = xxyDist_.xyCounts.getCount(x1,
						x1Val);

			for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {

				if (!generalizationSet[x2]) {
					const bool x2Active = active[x2];

					if (x1Active || x2Active ) {
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

							if (x1Active) {
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
							if (x2Active ) {
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
		}
	}

	for (CatValue parent = 0; parent < noCatAtts_; parent++) {
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


void aode::nbClassify(const instance &inst, std::vector<double> &classDist,
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

