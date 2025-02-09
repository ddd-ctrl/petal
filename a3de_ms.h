/*
 * a3de3.h
 *
 *  Created on: 28/09/2012
 *      Author: shengleichen
 */

#pragma once

#include "incrementalLearner.h"
#include "xxxxyDist3.h"
#include "crosstab.h"
#include "crosstab3d.h"

class a3de_ms: public IncrementalLearner {
public:

	a3de_ms(char* const *& argv, char* const * end);

	virtual ~a3de_ms();

	void reset(InstanceStream &is);   ///< reset the learner prior to training

	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()

	/**
	 * Inisialises the pass indicated by the parametre.
	 *
	 * @param pass  Current pass.
	 */
	void initialisePass();
	/**
	 * Train an a3de3 with instance inst.
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
	 * Calculates the weight for wa3de3
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:

	unsigned int minCount;

	bool subsumptionResolution; ///< true if using lazy subsumption resolution
	bool weighted; 			///< true  if  using weighted aode

	bool factored_;    	///<true iff user specify how many attributes to select
	float factor_;      ///< number of  mutual information selected attributes is the original number by factor_

	bool random_;		///<true iff random shuffle the attributes
	bool su_;			///<true iff using symmetric uncertainty to rank attributes
	bool mi_;			///<true iff using mutual information to rank attributes
	bool acmi_;			///<true iff using conditional mutual information to rank attributes
	bool directRank_;   ///<true iff directly rank the attributes

	bool avg_;
	bool sum_;

    bool empiricalMEst_;  	///< true iff using empirical m-estimation
    bool empiricalMEst2_; 	///< true iff using empirical m-estimation of attribute given parent

	InstanceStream* instanceStream_;

	bool oneSelective_;
	bool twoSelective_;

	std::vector<bool> active_; ///< true for active[att] if att is selected -- flags: chisq, selective, selectiveTest
	std::vector<CategoricalAttribute> order_;
	unsigned int pass_;


	std::vector<bool> generalizationSet;
	crosstab<float> weight_a2de; ///<stores the mutual information between each attribute and class as weight
	std::vector<float> weight_aode; ///<stores the mutual information between each attribute and class as weight
	crosstab3D<float> weight; ///<stores the mutual information between each attribute and class as weight


	unsigned int noSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.
	unsigned int noUnSelectedCatAtts_; ///< the number of categorical CategoricalAttributes.

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes

	InstanceCount count_;
    xxxxyDist3 xxxxyDist_;
	xxxyDist xxxyDist_;

	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &dist);
	void aodeClassify(const instance &inst, std::vector<double> &classDist,
			xxyDist & dist);
	void a2deClassify(const instance &inst, std::vector<double> &classDist,
			xxxyDist & dist);

};

