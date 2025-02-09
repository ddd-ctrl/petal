/*
 * Feating2Learner.h
 *
 *  Created on: 26/03/2013
 *      Author: shengleichen
 */

#include "learner.h"
#include "FilterSet.h"

#include "instanceStreamDiscretiser.h"


class Feating3Learner: public learner{
public:
	Feating3Learner(char*const*& argv, char*const* end);
	virtual ~Feating3Learner();


	void getCapabilities(capabilities &c);

	/**
	 * trains the feating learners.
	 *
	 * @param is The training set
	 */
	virtual void train(InstanceStream &is);

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param inst The instance to be classified
	 * @param classDist Predicted class probability distribution
	 */
	virtual void classify(const instance &inst, std::vector<double> &classDist);


    void printClassifier();       ///< print details of the classifier that has been created

private:

	char const* learnerName_;        		///< the name of the learner
	char* const * learnerArgv_;  			///< the start of the arguments for the learner
	char* const * learnerArgEnd_;   		///< the end of the arguments ot the learner

	CategoricalAttribute k_;      			///< select top k pairs of attributes
	std::vector<std::vector<std::vector<bool> > > selected_;
											///< true iff the pair of attributes is selected

	bool randomSelected_;         			///< true iff random selected k pairs
 	bool topSelected_;						///< true iff selected top k pairs
 	unsigned int noCatAtts_;				///< number of categorical attributes
 	unsigned int noNumAtts_;				///< number of numerical attributes
	unsigned int noAtts_;					///< number of all attributes

	capabilities capabilities_;				///< learner capabilities

	bool useMajorityVoting_;   				///< true iff the classifier uses majority voting
	FilterSet filter;						///< the filter for discretizing the data
	InstanceStreamDiscretiser *discretiser_;///< the pointer to the discretizer

	std::vector<std::vector<std::vector<std::vector<learner*> > > > classifiers_; ///< the classifiers for both type of attributes, indexed by attribute then discretised value
};


