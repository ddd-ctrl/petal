/*
 * a2de.h
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#pragma once

#include "incrementalLearner.h"
#include "xxxyDist.h"
#include "crosstab.h"
#include "utils.h"

class a2de: public IncrementalLearner {
public:

	a2de(char* const *& argv, char* const * end);

	a2de(const a2de& l);            ///< copy constructor

	learner* clone() const;         ///< create a copy of the learner

	virtual ~a2de();

	void reset(InstanceStream &is);   ///< reset the learner prior to training

	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

	/**
	 * Inisialises the pass indicated by the parametre.
	 *
	 * @param pass  Current pass.
	 */
	void initialisePass();
	/**
	 * Train an aode with instance inst.
	 *
	 * @param inst Training instance
	 */
	void train(const instance &inst);

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param inst The instance to be classified
	 * @param classDist Predicted class probability distribution
	 */
	void classify(const instance &inst, std::vector<double> &classDist);
	/**
	 * Calculates the weight for wa2de
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:
	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &dist);
	void aodeClassify(const instance &inst, std::vector<double> &classDist,
			xxyDist & dist);

	void LOOCV(const instance &inst);

	void LOOCV2(const instance &inst);


	std::vector<CategoricalAttribute> orderedAtts_;   ///<the rank of the attrbiutes for loocv selection
	bool loo_;				///< true if performing the second pass to do leave-one-out cross validation
	unsigned int pass_;
	bool directRank_;   ///<true iff directly rank the attributes
	std::vector<std::vector<double> > squaredError_;

    std::vector<float>   optSquaredError_; ///< optimal squared error for each 2de

	unsigned int optParentIndex_; ///< indicate the how many attributes have been selected as parent
	unsigned int optChildIndex_; ///< indicate the how many attributes have been selected as children

	std::vector<CategoricalAttribute> indOptChildIndex_;
	///< indicates how many attributes have been selected in each 2de
    std::vector<CategoricalAttribute> orderedParent_;


	unsigned int minCount;

	bool subsumptionResolution; ///< true if using lazy subsumption resolution
	bool weighted; 			///< true  if  using weighted a2de
	bool avg;  				///< true if using averaged mutual information as weight

	bool selected;   			///< true iif using selective a2de

	bool su_;			///<true if using symmetric uncertainty to select active attributes
	bool mi_;			///<true if using mutual information to select active attributes
	bool chisq_;		///< true if using chi square test to selects active attributes
	bool mRMR_;
    bool  ind_;         ///<true if selecting attributes within each 2de independently
    bool allChild_;     ///<true iff keeping all child attributes in the second pass, selecting only 2de
    bool vote_;         ///<true iff voting the instance by optimized 2de


    bool empiricalMEst_;  ///< true if using empirical m-estimation
    bool empiricalMEst2_;  ///< true if using empirical m-estimation of attribute given parent



    bool bayesRace_;     ///< true if using Bayesian race
    std::vector<double> meanOfModels_;  ///< stores mean of all models
    std::vector<double> varianceOfModels_;  ///< stores variance of all models
    std::vector<bool>  modelThrownOut_;  ///< true if the model has been thrown out
	double delta_;     ///< confidence parameter in Bayesian race
    int noOfModelsThrownOut_;  ///< no of models thrown out


	InstanceStream* instanceStream_;
	crosstab<double> weight; ///<stores the mutual information between each attribute and class as weight

	std::vector<double> weightaode; ///<stores the mutual information between each attribute and class as weight

//	std::vector<int> generalizationSet;
//	std::vector<int> substitutionSet;

	std::vector<bool> generalizationSet; 		///< indicate if att should be delete according to subsumption resolution

	std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
	unsigned int inactiveCnt_; ///< number of attributes not selected -- flags: chisq, selective, selectiveTest
	unsigned int factor_;      ///< number of  mutual information selected attributes is the original number by factor_




	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
	InstanceCount count;
	xxxyDist xxxyDist_;





	InstanceCount trainSize_;  ///< number of examples for training
	MTRand_open rand_;                  ///< random number generator for selecting test set for pass 2
	InstanceCount testSetSize_;         ///< the number of test cases to use in pass 2
	InstanceCount testSetSoFar_;        ///< used in pass 2 to count how many test cases have been used so far
	InstanceCount seenSoFar_;           ///< used in pass 2 to count how many cases from the input stream have been seen so far

	InstanceCount testSetSizeP_;       ///< the number of test cases to use in pass 1
	InstanceCount testSetSoFarP_;        ///< used in pass 1 to count how many test cases have been used so far
	InstanceCount seenSoFarP_;           ///< used in pass 1 to count how many cases from the input stream have been seen so far
	bool sample4prmtr_;     ///true iff sampling for parameterization of the model

};

