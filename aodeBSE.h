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

#pragma once

#include "incrementalLearner.h"
#include "xxyDist.h"
#include "utils.h"

class aodeBSE: public IncrementalLearner {
public:
	/**
	 * @param argv Options for the aode classifier
	 * @param argc Number of options for aode
	 * @param m    Metadata information
	 */
	aodeBSE(char* const *& argv, char* const * end);

        aodeBSE(const aodeBSE& l);

        learner* clone() const;         ///< create a copy of the learner

	virtual ~aodeBSE(void);

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
	 * Calculates the weight for waode
	 */
	void finalisePass();

	void getCapabilities(capabilities &c);

private:
	/**
	 * Naive Bayes classifier to which aode will deteriorate when there are no eligible parent attribute (also as SPODE)
	 *
	 *@param inst The instance to be classified
	 *@param classDist Predicted class probability distribution
	 *@param dist  class object pointer of xyDist describing the distribution of attribute and class
	 */
	void nbClassify(const instance &inst, std::vector<double> &classDist,
			xyDist &xyDist_);

	int getNextElement(std::vector<CategoricalAttribute> &order,CategoricalAttribute ca, unsigned int noSelected);

	/**
	 * Leave-one-out cross validation function
	 * @param inst The instance to be used
	 *
	 */
	void LOOCV(const instance &inst);

	bool poce_;  			///< true iif parent or child elimination is done
	bool pace_;  			///< true iif parent and child elimination is done
	bool ce_;  			///< true iif child elimination is done
	bool pe_;			///< true iif parent elimination is done

	bool trainingIsFinished_; ///< true iff the learner is trained
	std::vector<bool> predictedBest_;
	std::vector<std::vector<bool> > predicted_;
	unsigned int win_;
	unsigned int loss_;

	unsigned int count_;
	unsigned int pass_;                                        ///< the number of passes for the learner

    bool empiricalMEst_;  ///< true if using empirical m-estimation
    bool empiricalMEst2_;  ///< true if using empirical m-estimation of attribute given parent

	bool loo_;				///< true if performing the second pass to do leave-one-out cross validation using strategy 5

	unsigned int optParentIndex_; ///< indicate the how many attributes have been selected as parent
	unsigned int optChildIndex_; ///< indicate the how many attributes have been selected as children
	std::vector<unsigned int> noChild_;

	std::vector<double>  squaredErrorChild_;
	std::vector<double>  squaredErrorParent_;
	std::vector<double>  squaredErrorParentChild_;

	std::vector<double>  minError_;
	double  squaredErrorAode_;

	//squaredError1D_
	int noActiveChild_;
	int noActiveParent_;
	int noActivePC_;
	std::vector<bool> activeChild_; 	///< true for active[att] if att is selected as child
	std::vector<bool> activeParent_; 	///< true for active[att] if att is selected as parent
	std::vector<bool> activeParentChild_; 	///< true for active[att] if att is selected as parent and child

	InstanceStream* instanceStream_;

	unsigned int noCatAtts_;  ///< the number of categorical attributes.
	unsigned int noClasses_;  ///< the number of classes
	xxyDist xxyDist_; ///< the xxy distribution that aode learns from the instance stream and uses for classification
};

