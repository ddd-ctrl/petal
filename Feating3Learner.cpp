/*
 * Feating3Learner.cpp
 *
 *  Created on: 26/03/2013
 *      Author: shengleichen
 */

#include "Feating3Learner.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "learnerRegistry.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"
#include "crosstab3d.h"

static char*const* argBegin = NULL;
static char*const* argEnd = argBegin;

Feating3Learner::Feating3Learner(char*const*& argv, char*const* end) {

	learner *theLearner = NULL;

	discretiser_=NULL;


	name_ = "FEATED3";
	k_=100;
	randomSelected_=false;
	topSelected_=false;

	// defaults
	useMajorityVoting_ = true;

	noCatAtts_=0;
	noNumAtts_=0;
	noAtts_=0;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		} else if (argv[0][1] == 'd') {
			useMajorityVoting_ = false;
		} else if (argv[0][1] == 'k') {
			getUIntFromStr(argv[0] + 2, k_, "k");
		} else if (streq(argv[0] + 1, "top")) {
			topSelected_=true;
		} else if (streq(argv[0] + 1, "rand")) {
			randomSelected_=true;
		} else if (argv[0][1] == 'b') {
			// specify the base learner
			learnerName_ = argv[0] + 2;
			learnerArgv_ = ++argv;

			// create the learner
			theLearner = createLearner(learnerName_, argv, end);

			if (theLearner == NULL) {
				error("Learner %s is not supported", learnerName_);
			}
			learnerArgEnd_ = argv;

			name_ += "_";
			name_ += *theLearner->getName();
			break;
		} else {
			break;
		}

		name_ += argv[0];

		++argv;
	}

	if (theLearner == NULL)
		error("No base learner specified");
	else {
		theLearner->getCapabilities(capabilities_);
		delete theLearner;
	}


}

void Feating3Learner::printClassifier()
{
	printf("Classifier %s is constructed.\n", name_.c_str());
}
Feating3Learner::~Feating3Learner() {
	// TODO Auto-generated destructor stub
	for (unsigned int a1 = 0; a1 < classifiers_.size(); ++a1) {
		for (unsigned int a2 = 0; a2 < classifiers_[a1].size(); ++a2){
				for (unsigned int a3 = 0; a3 < classifiers_[a1][a2].size(); ++a3) {
					for (unsigned int a4 = 0; a4 < classifiers_[a1][a2][a3].size(); ++a4) {

							delete classifiers_[a1][a2][a3][a4];
				}
			}
		}
	}
}

void  Feating3Learner::getCapabilities(capabilities &c){
  c = capabilities_;
}

void Feating3Learner::train(InstanceStream &is) {
	// load the data into a store
	is.rewind();
	StoredInstanceStream store;
	StoredIndirectInstanceStream thisStream; ///< the bagged stream for learning the next classifier
	AddressableInstanceStream* aStream =
			dynamic_cast<AddressableInstanceStream*>(&is);

	//store the instance in the memory
	if (aStream == NULL) {
		store.setSource(is);
		aStream = &store;
	}

	noCatAtts_=is.getMetaData()->getNoCatAtts();
	noNumAtts_=is.getMetaData()->getNoNumAtts();
	noAtts_ = noCatAtts_ + noNumAtts_;



	// discretize for calculating mutual information
	// and constructing level tree
	filter.clear();
	discretiser_=new InstanceStreamDiscretiser("mdl2", argBegin, argEnd);

	filter.push_back(discretiser_);
	InstanceStream *discStream=filter.apply(&is);



	//clear and initialize the vector
	for (unsigned int i = 0; i < selected_.size(); i++) {
		for (unsigned int j = 0; j < selected_[i].size(); j++)
			selected_[i][j].clear();
		selected_[i].clear();
	}
	selected_.clear();


	selected_.resize(noAtts_);
	//no selection is performed
	if(topSelected_==false&&randomSelected_==false)
	{
		for (CategoricalAttribute a1 = 2; a1 < noAtts_; ++a1) {
			selected_[a1].resize(a1);
			for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {
				selected_[a1][a2].resize(a2, true);
			}
		}
		printf("All pairs have been selected.\n");

	}else
	{
		for (CategoricalAttribute a1 = 2; a1 < noAtts_; ++a1) {
			selected_[a1].resize(a1);
			for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {
				selected_[a1][a2].resize(a2, false);
			}
		}

		if(noAtts_<3)
			error("feating3 needs at least 3 attributes.\n");

		unsigned int length = noAtts_ * (noAtts_ - 1)*(noAtts_ - 2) / 6;


		//no need to select
		if (length < k_) {
			printf("Attempt to select %u pairs from %u pairs.\n"
					"All pairs have been selected.\n", k_, length);
			for (CategoricalAttribute a1 = 2; a1 < noAtts_; ++a1) {
				for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {
					for (CategoricalAttribute a3 = 0; a3 < a2; ++a3) {
						selected_[a1][a2][a3]= true;
					}
				}
			}
		} else {
			if (randomSelected_ == true) {

				printf("Random select %u attribute pairs.\n",k_);
				//randomly select k_ pairs without replacement
				MTRand rand;     ///< random number generator for selecting bags

				unsigned int selected = 0;
				unsigned int i = 1;
				while (selected < k_ && i <= length) {
					double random = rand();
					double thres = static_cast<double>(k_ - selected)
							/ (length - i + 1);
					if (random < thres) {
						//select the pair i
						//calculate the three indexes for the selected pair
						CategoricalAttribute a1 = 2;
						CategoricalAttribute a2 = 1;
						CategoricalAttribute a3;
						while (a1 * (a1 - 1) * (a1 - 2) / 6 < i) {
							++a1;
						}
						a1--;

						unsigned int rest = i- a1 * (a1 - 1) * (a1 - 2) / 6;

						while (a2 * (a2 - 1) / 2 < rest) {
							++a2;
						}
						a2--;

						a3 = rest - (a2 * (a2 - 1) / 2);
						selected_[a1][a2][a3] = true;

						if (verbosity >= 2) {
							printf("%u,%u,%u,%u\n", i, a1, a2, a3);
						}
						selected++;
					}

					i++;
				}
			} else {
				assert(topSelected_==true);

				printf("Select top %u attribute pairs.\n",k_);
				xxxyDist dist;
				crosstab3D<float> tmi(noAtts_);

				//scan the data set and update the distribution
				instance inst(*discStream);
				dist.reset(*discStream);
				discStream->rewind();
				while (discStream->advance(inst)) {
					dist.update(inst);
				}

				//calculate the mutual information
				getTripleMutualInf(dist, tmi);

				if (verbosity >= 2) {
					printf("output the triple mutual information:\n");

					for (CategoricalAttribute a1 = 2; a1 < noAtts_; ++a1) {
						for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {
							print(tmi[a1][a2]);
						}
					}
				}

				float min;
				CategoricalAttribute minIndex1, minIndex2,minIndex3;

				//find the top k pairs of attributes
				for (CategoricalAttribute a1 = 2; a1 < noAtts_; ++a1) {
					for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {
						for (CategoricalAttribute a3 = 0; a3 < a2; ++a3) {

							//convert the 3 dimensional coordinates to one dimensional coordinate
							CategoricalAttribute i = a1 * (a1 - 1) * (a1 - 2)
									/ 6 + (a2 - 1) * a2 / 2 + a3;
							if (i < k_) {
								selected_[a1][a2][a3] = true;

								//record the min value with their indexes for the first k pairs
								if (a1 == 2 && a2 == 1&& a3 == 0) {
									min = tmi[a1][a2][a3];
									minIndex1 = a1;
									minIndex2 = a2;
									minIndex3 = a3;
								} else {
									if (min > tmi[a1][a2][a3]) {
										min = tmi[a1][a2][a3];
										minIndex1 = a1;
										minIndex2 = a2;
										minIndex3 = a3;
									}
								}
							} else {
								if (tmi[a1][a2][a3] > min) {

									selected_[a1][a2][a3] = true;
									selected_[minIndex1][minIndex2][minIndex3]  = false;

									//update the min value and the indexes
									bool first = true;
									unsigned int k = 0;
									for (CategoricalAttribute i = 2; i <= a1; ++i) {
										for (CategoricalAttribute j = 1; j < i; ++j) {
											for (CategoricalAttribute r = 0; r < j; ++r) {

												if (selected_[i][j][r]
														== true) {
													k++;
													if (first == true) {
														min = tmi[i][j][r];
														minIndex1 = i;
														minIndex2 = j;
														minIndex3 = r;
														first = false;
													} else {
														if (min > tmi[i][j][r]) {
															min = tmi[i][j][r];
															minIndex1 = i;
															minIndex2 = j;
															minIndex3 = r;
														}
													}
												}
											}
										}
									}
									assert(k==k_);
								}
							}

						}
					}
				}
				if (verbosity >= 2) {
					printf("output the top k pairs mutual information:\n");

					for (CategoricalAttribute a1 = 2; a1 < noAtts_; ++a1) {
						for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {
							for (CategoricalAttribute a3 = 0; a3 < a2; ++a3) {
								if (selected_[a1][a2][a3] == true) {

									printf("%u,%u,%u,%f\n", a1, a2, a3,
											tmi[a1][a2][a3]);
								}
							}
						}
					}
				}
			}
		}
	}

	printf("The number of attributes: %u, categorical: %u, numerical: %u\n",noAtts_,noCatAtts_,noNumAtts_);

	for (unsigned int a1 = 0; a1 < classifiers_.size(); ++a1) {
		for (unsigned int a2 = 0; a2 < classifiers_[a1].size(); ++a2){
			for (unsigned int a3 = 0; a3 < classifiers_[a1][a2].size(); ++a3) {
				for (unsigned int a4 = 0; a4 < classifiers_[a1][a2][a3].size();
						++a4) {
					delete classifiers_[a1][a2][a3][a4];
				}
			}
		}
	}
	classifiers_.clear();

	//construct the level tree and train each classifier
	classifiers_.resize(noAtts_);
	for (unsigned int a1 = 2; a1 < noAtts_; ++a1) {

		unsigned int noVals1;
		//categorical attributes
		if (a1 < noCatAtts_)
			noVals1=aStream->getNoValues(a1);
		else
		{
			//numerical attributes
			//one for missing value
			noVals1 = discretiser_->getMetaData()->cuts[a1-noCatAtts_].size()
					+ 1;
			if (aStream->getMetaData()->hasNumMissing(a1 - noCatAtts_))
				noVals1 +=1;
		}

		classifiers_[a1].resize(noVals1 * a1);
		for (unsigned int v1 = 0; v1 < noVals1; ++v1) {
			for (CategoricalAttribute a2 = 1; a2 < a1; ++a2) {

				unsigned int noVals2;
				//categorical attributes
				if (a2 < noCatAtts_)
					noVals2=aStream->getNoValues(a2);
				else
				{
					//numerical attributes
					//one for missing value
					noVals2 = discretiser_->getMetaData()->cuts[a2-noCatAtts_].size()
							+ 1;
					if (aStream->getMetaData()->hasNumMissing(a2 - noCatAtts_))
						noVals2 += 1;

				}

				classifiers_[a1][v1*a1+a2].resize(noVals2 * a2);

				for (unsigned int v2 = 0; v2 < noVals2; ++v2) {
					for (CategoricalAttribute a3 = 0; a3 < a2; ++a3) {

						if (selected_[a1][a2][a3] == false)
								continue;

						unsigned int noVals3;
						if (a3 < noCatAtts_)
							noVals3=aStream->getNoValues(a3);
						else
						{
							//numerical attributes
							//one for missing value
							noVals3 = discretiser_->getMetaData()->cuts[a3-noCatAtts_].size()
									+ 1;
							if (aStream->getMetaData()->hasNumMissing(a3 - noCatAtts_))
								noVals3 += 1;
						}

						for (unsigned int v3 = 0; v3 < noVals3; ++v3) {

							thisStream.setSourceWithoutLoading(*aStream); // clear the stream

							aStream->rewind();

							while (aStream->advance()) {
								if(a1 < noCatAtts_)
								{
									if ((aStream->current()->getCatVal(a1) == v1)
										&& (aStream->current()->getCatVal(a2) == v2)
										&& (aStream->current()->getCatVal(a3) == v3))

									thisStream.add(aStream->current());
								}else
								{
									if(a2 < noCatAtts_)
									{
										if ((discretiser_->discretise(aStream->current()->getNumVal(a1 - noCatAtts_),a1 - noCatAtts_) == v1)
											&& (aStream->current()->getCatVal(a2) == v2)
											&& (aStream->current()->getCatVal(a3) == v3))

										thisStream.add(aStream->current());
									}else
									{
										if(a3 < noCatAtts_)
										{
											if ((discretiser_->discretise(aStream->current()->getNumVal(a1 - noCatAtts_),a1 - noCatAtts_) == v1)
												&& (discretiser_->discretise(aStream->current()->getNumVal(a2 - noCatAtts_),a2 - noCatAtts_) == v2)
												&& (aStream->current()->getCatVal(a3) == v3))

											thisStream.add(aStream->current());

										}else
										{
											if ((discretiser_->discretise(aStream->current()->getNumVal(a1 - noCatAtts_),a1 - noCatAtts_) == v1)
												&&(discretiser_->discretise(aStream->current()->getNumVal(a2 - noCatAtts_),a2 - noCatAtts_) == v2)
												&&(discretiser_->discretise(aStream->current()->getNumVal(a3 - noCatAtts_),a3 - noCatAtts_) == v3))

											{
												//printf("add the instance.\n");
												thisStream.add(aStream->current());
											}
										}
									}
								}
							}
							//printf("%u,%u,%u,%u.\n",a1,v1 * a1 + a2,v2 * a2 + a3,v3);
							classifiers_[a1][v1 * a1 + a2][v2 * a2 + a3].push_back(
									createLearner(learnerName_, learnerArgv_,
											learnerArgEnd_));
							classifiers_[a1][v1 * a1 + a2][v2 * a2 + a3].back()->train(
									thisStream);
						}
					}
				}
			}
		}
	}
}

void Feating3Learner::classify(const instance &inst, std::vector<double> &classDist) {
  std::vector<double> thisClassDist(classDist.size());

	classDist.assign(classDist.size(), 0.0);

	assert(classifiers_.size()==noAtts_);
	for (unsigned int a1 = 2; a1 < noAtts_; ++a1) {
		unsigned int v1;

		if (a1 < noCatAtts_)
			v1= inst.getCatVal(a1);
		else
			v1 = discretiser_->discretise(
								inst.getNumVal(a1 - noCatAtts_), a1 - noCatAtts_);

		for (unsigned int a2 = 1; a2 < a1; ++a2) {
			unsigned int v2;

			if (a2 < noCatAtts_)
				v2= inst.getCatVal(a2);
			else
				v2 = discretiser_->discretise(
									inst.getNumVal(a2 - noCatAtts_), a2 - noCatAtts_);

			for (unsigned int a3 = 0; a3 < a2; ++a3) {

				if(selected_[a1][a2][a3]==false)
					continue;

				unsigned int v3;

				if (a3 < noCatAtts_)
					v3= inst.getCatVal(a3);
				else
					v3 = discretiser_->discretise(
										inst.getNumVal(a3 - noCatAtts_), a3 - noCatAtts_);

				classifiers_[a1][v1 * a1 + a2][v2 * a2 + a3][ v3 ]->classify(
						inst, thisClassDist);

				if (useMajorityVoting_) {
					classDist[indexOfMaxVal(thisClassDist)] += 1.0;
				} else {
					for (CatValue y = 0; y < classDist.size(); ++y) {
						classDist[y] += thisClassDist[y];
					}
				}
			}
		}
	}
	normalise(classDist);
}
