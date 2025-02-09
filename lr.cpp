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
 **
 **
 */

#include "lr.h"
#include "learnerRegistry.h"

static LearnerRegistrar registrar("lr", constructor<lr>);

int lr::ind(int i, int j) {
	return (i==j) ? 1 : 0;
}

int lr::getPosNominal(int c, int att, int val) {
	return c * paramsPerClass + startAtt[att] + val;
}

int lr::progress(
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
	printf("%f, ", fx);
	//	printf("Iteration %d:\n", k);
	//	printf("  fx = %f, Parameter value is: \n", fx);
	//	for (int i = 0; i < n; i++) {
	//		printf("%f, ", x[i]);
	//	}
	//	printf("\n, Gradient vector is: \n");
	//	for (int i = 0; i < n; i++) {
	//		printf("%f, ", g[i]);
	//	}
	//	printf("\n");
	//	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	//	printf("\n");
	return 0;
}

void lr::logDomainNormalize() {
	double logSum = 0;
	double maxLog = regressor[0];
	int idxMax = 0;
	for (int c = 1; c < nc; c++) {
		if (maxLog < regressor[c]) {
			maxLog = regressor[c];
			idxMax = c;
		}
	}
	double sum = 0;
	for (int c = 0; c < nc; c++) {
		if (c == idxMax) {
			sum++;
		} else {
			sum += exp(regressor[c] - maxLog);
		}
	}
	logSum = maxLog + log(sum);

	for (int c = 0; c < nc; c++) {
		regressor[c] -= logSum;
	}
}

void lr::logDomainExp() {
	for (int c = 0; c < nc; c++) {
		regressor[c] = exp(regressor[c]);
	}
}

void lr::inplaceLogGradientFromDistribution(const instance &inst, lbfgsfloatval_t *g) {

	int x_C = inst.getClass();
	int pos = 0;
	for (int k = 0; k < nc; k++) {
		g[pos] += (doWANBIA) ? -1 * (ind(k,x_C) - regressor[k]) * logtheta_C[k] : -1 * (ind(k,x_C) - regressor[k]);
		pos += paramsPerClass;
	}

	for (int u = 0; u < n; u++) {
		double x_u = inst.getCatVal(u);

		for (int k = 0; k < nc; k++) {
			int pos = getPosNominal(k, u, (int)x_u);
			g[pos] += (doWANBIA) ? (-1) * (ind(k,x_C) - regressor[k]) * logtheta_UC[u][k][(int)x_u] : (-1) * (ind(k,x_C) - regressor[k]);
		}
	}

}

void lr::logdistributionForInstance(const instance &inst) {

	for (int i = 0; i < nc; i++)
		regressor[i] = 0;

	for (int c = 0; c < nc; c++) {
		regressor[c] = (doWANBIA) ? x[c * paramsPerClass] * logtheta_C[c] : x[c * paramsPerClass];
	}

	for (int u = 0; u < n; u++) {
		double x_u = inst.getCatVal(u);

		for (int c = 0; c < nc; c++) {
			int pos = getPosNominal(c, u,(int)x_u);
			regressor[c] += (doWANBIA) ? x[pos] * logtheta_UC[u][c][(int)x_u] : x[pos];
		}
	}

	logDomainNormalize();
}

void lr::distributionForInstance(const instance &inst) {
	logdistributionForInstance(inst);
	logDomainExp();
}

lbfgsfloatval_t lr::evaluate(
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int num,
		const lbfgsfloatval_t step
)
{
	int nc = data->getNoClasses();

	// Calculate function
	lbfgsfloatval_t fx = 0.0;
	for (int i = 0; i < np; i++) {
		g[i] = 0;
	}
	double ctt;
	ctt = -log(static_cast<double>(nc));

	instance inst(*data);
	data->rewind();
	while (!data->isAtEnd()) {
		if (data->advance(inst)) {

			logdistributionForInstance(inst);
			int x_C = inst.getClass();
			fx += (ctt - regressor[x_C]);
			logDomainExp();

			inplaceLogGradientFromDistribution(inst, g);
		}
	}

	return fx;
}

void lr::train(InstanceStream &is) {

	data = &is;

	N = is.size();
	n = is.getNoCatAtts(); // + is.getNoNumAtts();
	nc = is.getNoClasses();

	printf("N = %d, n = %d, nc = %d \n", N, n, nc);

	paramsPerAtt.resize(n);
	startAtt.resize(n);
	regressor.resize(nc);

	paramsPerClass = 1;

	int attIndex;
	attIndex = 0;
	for (int u = 0; u < n; u++) {
		paramsPerAtt[attIndex] = is.getNoValues(u);
		startAtt[attIndex] = paramsPerClass;
		paramsPerClass += paramsPerAtt[attIndex];
		attIndex++;
	}

	printf("ParamsPerClass = %d \n",paramsPerClass);
	np = nc * paramsPerClass;

	if (doWANBIA) {
		setNBProbabilities(is);
	}

	// LBFGS optimization routine starts here.
	//lbfgsfloatval_t fx;
	//lbfgsfloatval_t *x = lbfgs_malloc(np);
	//param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
	//param.max_iterations = 20;

	x = lbfgs_malloc(np);
	for (int i = 0; i < np; i++) {
		x[i] = initialParameters;
	}

	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	//param.max_iterations = 50;

	printf("fx = [");
	int ret = lbfgs(np, x, &fx, _evaluate, _progress, this, &param);
	printf("]\n");

	/* Report the result. */
	printf("L-BFGS optimization terminated with status code = %d\n", ret);
	printf(" -------------------------------------------------------- \n");

}

void lr::classify(const instance &inst, std::vector<double> &classDist) {
	distributionForInstance(inst);
	classDist = regressor;
}

lr::lr(char* const *& argv, char* const * end) {
	name_ = "LogisticRegression";

	doWANBIA = false;
	objectiveFunction = "CLL";
	initialParameters = 0;

	// get argument
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		}
		char *p = argv[0] + 1;
		//printf("%s \n", p);

		switch (*p) {
		case 'O':
			objectiveFunction = p + 1;
			break;
		case 'P':
			initialParameters = atof(p + 1);
			break;
		case 'W':
			doWANBIA = true;
			break;
		default:
			error("+%c flag is not supported by Logistic Regression", *p);
		}

		++argv;
	}
}

lr::~lr() {
	lbfgs_free(x);
}

void lr::setNBProbabilities(InstanceStream &is) {
	xyDist_.reset(&is);

	instance inst(is);
	is.rewind();
	while (!is.isAtEnd()) {
		if (is.advance(inst)) {
			xyDist_.update(inst);
		}
	}

	// Initialize logtheta_C and logtheta_UC
	logtheta_C.resize(nc);

	logtheta_UC.resize(n);
	for (int i = 0; i < n; i++) {
		logtheta_UC[i].resize(nc);
	}

	for (int u = 0; u < n; u++) {
		for (int c = 0; c < nc; c++) {
			logtheta_UC[u][c].resize(is.getNoValues(u));
		}
	}

	// Populate logtheta_C and logtheta_UC
	for (int c = 0; c < nc; c++) {
		logtheta_C[c] = log(xyDist_.p(c));
		//printf("%f ", logtheta_C[c]);
	}
	//printf("\n");

	for (int u = 0; u < n; u++) {
		for (int c = 0; c < nc; c++) {
			for (CatValue i = 0; i < is.getNoValues(u); i++) {
				logtheta_UC[u][c][i] = log(xyDist_.p(u,i,c));
				//printf("%f, ", logtheta_UC[u][c][i]);
			}
			//printf("\n");
		}
		//printf(" ------------------------------------------------------- \n");
	}
	//printf("All Done");

}

void lr::getCapabilities(capabilities &c) {
	c.setCatAtts(true);
}

void lr::printRegressor() {
	for (int i = 0; i < nc; i++) {
		printf("%f, ", regressor[i]);
	}
	printf("\n");
}

void lr::printx() {
	for (int i = 0; i < np; i++) {
		printf("%f, ", x[i]);
	}
	printf("\n");
}

void lr::printg(lbfgsfloatval_t *g) {
	for (int i = 0; i < np; i++) {
		printf("%f, ", g[i]);
	}
	printf("\n");
}

