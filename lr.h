/* Open source system for classification learning from very large data
 ** Class for a decision tree learner
 **
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
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "learner.h"
#include "instanceSample.h"
#include "utils.h"

#include "xyDist.h"

#include "lbfgs.h"

class lr: public learner {
public:
	lr(char* const *& argv, char* const * end);
	virtual ~lr();

	void getCapabilities(capabilities &c);

	/**
	 * trains the logistic regression classifier using LBFGS
	 * quasi-Newton optimization method.
	 *
	 * @param is The training set.
	 */
	virtual void train(InstanceStream &is);

	/**
	 * Calculates the class membership probabilities for the given instance.
	 *
	 * @param inst The instance to be classified.
	 * @param classDist Predicted class probability distribution.
	 */
	virtual void classify(const instance &inst, std::vector<double> &classDist);

	void inplaceLogGradientFromDistribution(const instance &inst, lbfgsfloatval_t *g);

	void printRegressor();

	void printx();

	void printg(lbfgsfloatval_t *g);

	int ind(int i, int j);

	void logDomainNormalize();

	void logDomainExp();

	int getPosNominal(int c, int att, int val);

	void setNBProbabilities(InstanceStream &is);

	void distributionForInstance(const instance &inst);

	void logdistributionForInstance(const instance &inst);

	static lbfgsfloatval_t _evaluate(
			void *instance,
			const lbfgsfloatval_t *x,
			lbfgsfloatval_t *g,
			const int n,
			const lbfgsfloatval_t step
	)
	{
		return reinterpret_cast<lr*>(instance)->evaluate(x, g, n, step);
	}

	lbfgsfloatval_t evaluate(
			const lbfgsfloatval_t *x,
			lbfgsfloatval_t *g,
			const int num,
			const lbfgsfloatval_t step
	);

	static int _progress(
			void *instance,
			const lbfgsfloatval_t *x,
			const lbfgsfloatval_t *g,
			const lbfgsfloatval_t fx,
			const lbfgsfloatval_t xnorm,
			const lbfgsfloatval_t gnorm,
			const lbfgsfloatval_t step,
			int n,
			int k,
			int ls
	)
	{
		return reinterpret_cast<lr*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
	}

	int progress(
			const lbfgsfloatval_t *x,
			const lbfgsfloatval_t *g,
			const lbfgsfloatval_t fx,
			const lbfgsfloatval_t xnorm,
			const lbfgsfloatval_t gnorm,
			const lbfgsfloatval_t step,
			int n,
			int k,
			int ls
	);


private:
	char const* learnerName_;       ///< the name of the learner
	char* const* learnerArgv_;  	///< the start of the arguments for the learner
	char* const* learnerArgEnd_;   	///< the end of the arguments to the learner

	capabilities capabilities_;

	bool doWANBIA;
	char* objectiveFunction;
	float initialParameters;

	// Model LR parameters
	std::vector<int> paramsPerAtt;
	std::vector<int> startAtt;
	int paramsPerClass;
	int n, nc, N, np;

	std::vector<double> logtheta_C;
	std::vector<std::vector<std::vector<double> > > logtheta_UC;

	xyDist xyDist_;

	InstanceStream *data;
	std::vector<double> regressor;

	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x;
};

