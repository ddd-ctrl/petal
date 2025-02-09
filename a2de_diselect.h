/*
 * a2de3.h
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#pragma once

#include "incrementalLearner.h"
#include "xxxyDist4.h"
#include "crosstab.h"
#include "utils.h"

class a2de_diselect: public IncrementalLearner {
public:

	a2de_diselect(char* const *& argv, char* const * end);

	virtual ~a2de_diselect();

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
	 * Calculates the weight for wa2de3
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:

	bool loo_;				///< true if performing the third pass to do leave-one-out cross validation

	unsigned int minCount;  ///for subsumption resolution

	bool subsumptionResolution; ///< true iff using lazy subsumption resolution
	bool weighted; 			///< true  iff  using weighted a2de3

	bool factored_;    	///<true iff user specify how many attributes to select
	float factor_;      ///<the factor of how many attributes to select  that user specified

	bool random_;		///<true iff random shuffle the attributes
	bool su_;			///<true iff using symmetric uncertainty to rank attributes
	bool mi_;			///<true iff using mutual information to rank attributes
	bool acmi_;			///<true iff using conditional mutual information to rank attributes
	bool directRank_;   ///<true iff directly rank the attributes

	bool avg_;
	bool sum_;

    bool empiricalMEst_;  	///< true iff using empirical m-estimation
    bool empiricalMEst2_; 	///< true iff using empirical m-estimation of attribute given parent

	bool trainingIsFinished_; 	///< true iff the learner is trained

	InstanceStream* instanceStream_;

	unsigned int pass_;

	crosstab<float> weight; 		///<stores the mutual information between each attribute and class as weight
	std::vector<float> weightaode; 	///<stores the mutual information between each attribute and class as weight
	std::vector<bool> generalizationSet; 	///< indicate if att should be delete according to subsumption resolution

	std::vector<bool> active_; 				///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
	std::vector<CategoricalAttribute> order_;  ///< the rank of attributes according to different measure

	std::vector<std::vector<double> > squaredError_;
	unsigned int optParentIndex_; ///< indicate the how many attributes have been selected as parent
	unsigned int optChildIndex_; ///< indicate the how many attributes have been selected as children

	unsigned int noSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noUnSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes

	InstanceCount trainSize_;  ///< number of examples for training
	MTRand_open rand_;                  ///< random number generator for selecting test set for pass 2
	InstanceCount testSetSize_;         ///< the number of test cases to use in pass 2
	InstanceCount testSetSoFar_;        ///< used in pass 2 to count how many test cases have been used so far
	InstanceCount seenSoFar_;           ///< used in pass 2 to count how many cases from the input stream have been seen so far


	InstanceCount count;
	xxxyDist4 xxxyDist_;
	xxyDist xxyDist_;

	void LOOCV(const instance &inst);

	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &dist);
	void aodeClassify(const instance &inst, std::vector<double> &classDist,
			xxyDist &dist);
};

