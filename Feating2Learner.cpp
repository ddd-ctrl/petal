/*
 * Feating2Learner.cpp
 *
 *  Created on: 26/03/2013
 *      Author: shengleichen
 */

#include "Feating2Learner.h"
#include "StoredInstanceStream.h"
#include "StoredIndirectInstanceStream.h"
#include "learnerRegistry.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"


static char*const* argBegin = NULL;
static char*const* argEnd = argBegin;

Feating2Learner::Feating2Learner(char*const*& argv, char*const* end) {

	learner *theLearner = NULL;

	discretiser_=NULL;


	name_ = "FEATED2";
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

	printf("Classifier %s is constructed.\n", name_.c_str());
}

Feating2Learner::~Feating2Learner() {
	// TODO Auto-generated destructor stub
	for (unsigned int a1 = 0; a1 < classifiers_.size(); ++a1) {
		for (unsigned int a2 = 0; a2 < classifiers_[a1].size(); ++a2)
				for (unsigned int a3 = 0; a3 < classifiers_[a1][a2].size(); ++a3) {

							delete classifiers_[a1][a2][a3];
				}
			}
}

void  Feating2Learner::getCapabilities(capabilities &c){
  c = capabilities_;
}

void Feating2Learner::train(InstanceStream &is) {
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
//	for(unsigned int i=0;i<filter.size();i++)
//		delete filter[i];
	filter.clear();
	discretiser_=new InstanceStreamDiscretiser("mdl2", argBegin, argEnd);

	filter.push_back(discretiser_);
	InstanceStream *discStream=filter.apply(&is);



	//clear and initialize the vector
	for (unsigned int i = 0; i < selected_.size(); i++)
		selected_[i].clear();
	selected_.clear();

	selected_.resize(noAtts_);

	//no selection is performed
	if(topSelected_==false&&randomSelected_==false)
	{
		for (CategoricalAttribute a1 = 1; a1 < noAtts_; ++a1) {
			selected_[a1].resize(a1, true);
		}
		printf("All pairs have been selected.\n");

	}else
	{
		for (CategoricalAttribute a1 = 1; a1 < noAtts_; ++a1) {
			selected_[a1].resize(a1, false);
		}

		unsigned int length = noAtts_ * (noAtts_ - 1) / 2;


		//no need to select
		if (length < k_) {
			printf("Attempt to select %u pairs from %u pairs.\n"
					"All pairs have been selected.\n", k_, length);
			for (CategoricalAttribute a1 = 1; a1 < noAtts_; ++a1) {
				for (CategoricalAttribute a2 = 0; a2 < a1; ++a2) {
					selected_[a1][a2] = true;
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
						//calculate the two indexes for the selected pair
						CategoricalAttribute a1 = 1;
						CategoricalAttribute a2;
						while (a1 * (a1 - 1) / 2 < i) {
							++a1;
						}
						a1--;
						a2 = i - (a1 * (a1 - 1) / 2);
						selected_[a1][a2] = true;

						if (verbosity >= 2) {
							printf("%u,%u,%u\n", i, a1, a2);
						}
						selected++;
					}

					i++;
				}

				//		MTRand_int32 rand;           ///< random number generator for selecting bags
				//
				//		//randomly select k_ pairs with replacement
				//		for (unsigned int k = 0; k < k_; k++) {
				//			unsigned int random = rand(length);
				//
				//			//calculate the two indexes through random value
				//			CategoricalAttribute a1 = 1;
				//			CategoricalAttribute a2;
				//			while (a1 * (a1 - 1) / 2 < random) {
				//				++a1;
				//			}
				//			a1--;
				//			a2 = random - (a1 * (a1 - 1) / 2);
				//			selected_[a1][a2] = true;
				//			printf("%u,%u,%u\n",random,a1,a2);
				//		}

			} else if(topSelected_==true) {
				printf("Select top %u attribute pairs.\n",k_);
				xxyDist dist;
				crosstab<float> pmi(noAtts_);

				//scan the data set and update the distribution
				instance inst(*discStream);
				dist.reset(*discStream);
				discStream->rewind();
				while (discStream->advance(inst)) {
					dist.update(inst);
				}

				//calculate the mutual information
				getPairMutualInf(dist, pmi);

				if (verbosity >= 2) {
					printf("output the pair mutual information:\n");

					for (CategoricalAttribute a1 = 1; a1 < noAtts_; ++a1) {
						for (CategoricalAttribute a2 = 0; a2 < a1; ++a2) {
							printf("%u,%u,%f\n", a1, a2, pmi[a1][a2]);
						}
					}
				}

				float min;
				CategoricalAttribute minIndex1, minIndex2;

				//find the top k pairs of attributes
				for (CategoricalAttribute a1 = 1; a1 < noAtts_; ++a1) {

					for (CategoricalAttribute a2 = 0; a2 < a1; ++a2) {
						CategoricalAttribute i = (a1 - 1) * a1 / 2 + a2;
						if (i < k_) {
							selected_[a1][a2] = true;

							//record the min value with their indexes for the first k pairs
							if (a1 == 1 && a2 == 0) {
								min = pmi[a1][a2];
								minIndex1 = a1;
								minIndex2 = a2;
							} else {
								if (min > pmi[a1][a2]) {
									min = pmi[a1][a2];
									minIndex1 = a1;
									minIndex2 = a2;
								}
							}
						} else {
							if (pmi[a1][a2] > min) {

								selected_[a1][a2] = true;
								selected_[minIndex1][minIndex2] = false;

								//update the min value and the indexes
								bool first = true;
								unsigned int k = 0;
								for (CategoricalAttribute i = 1; i <= a1; ++i) {
									for (CategoricalAttribute j = 0; j < i; ++j) {

										if (i == a1 && j > a2)
											continue;

										if (selected_[i][j] == true) {
											k++;
											if (first == true) {
												min = pmi[i][j];
												minIndex1 = i;
												minIndex2 = j;
												first = false;
											} else {
												if (min > pmi[i][j]) {
													min = pmi[i][j];
													minIndex1 = i;
													minIndex2 = j;
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
				if (verbosity >= 2) {
					printf("output the top k pairs mutual information:\n");

					for (CategoricalAttribute a1 = 1; a1 < noAtts_; ++a1) {
						for (CategoricalAttribute a2 = 0; a2 < a1; ++a2) {
							if (selected_[a1][a2] == true) {

								printf("%u,%u,%f\n", a1, a2, pmi[a1][a2]);
							}
						}
					}

				}

			}
			else
			{
				error("error of selecting method.\n");
			}
		}
	}

	printf("The number of attributes: %u, categorical: %u, numerical: %u\n",noAtts_,noCatAtts_,noNumAtts_);

	for (unsigned int a1 = 0; a1 < classifiers_.size(); ++a1) {
		for (unsigned int a2 = 0; a2 < classifiers_[a1].size(); ++a2)
				for (unsigned int a3 = 0; a3 < classifiers_[a1][a2].size(); ++a3) {

							delete classifiers_[a1][a2][a3];
				}
			}
	classifiers_.clear();

	//construct the level tree and train each classifier
	classifiers_.resize(noAtts_);
	for (unsigned int a1 = 1; a1 < noAtts_; ++a1) {

		if (a1 < noCatAtts_) {
			//categorical attributes
			classifiers_[a1].resize(aStream->getNoValues(a1) * a1);
			for (unsigned int v1 = 0; v1 < aStream->getNoValues(a1); ++v1) {
				for (CategoricalAttribute a2 = 0; a2 < a1; ++a2) {

					if (selected_[a1][a2] == false)
						continue;

					for (CatValue v2 = 0; v2 < aStream->getNoValues(a2); ++v2) {

						thisStream.setSourceWithoutLoading(*aStream); // clear the stream

						aStream->rewind();

						while (aStream->advance()) {
							if ((aStream->current()->getCatVal(a1) == v1)
									&& (aStream->current()->getCatVal(a2) == v2)) {
								thisStream.add(aStream->current());
							}
						}

						classifiers_[a1][v1 * a1 + a2].push_back(
								createLearner(learnerName_, learnerArgv_,
										learnerArgEnd_));
						classifiers_[a1][v1 * a1 + a2].back()->train(
								thisStream);
					}
				}
			}

		} else {

			//numerical attributes
			//one for missing value
			unsigned int noVals;
			if (aStream->getMetaData()->hasNumMissing(a1 - noCatAtts_))
				noVals =
						discretiser_->getMetaData()->cuts[a1 - noCatAtts_].size()
								+ 2;
			else
				noVals =
						discretiser_->getMetaData()->cuts[a1-noCatAtts_].size()
								+ 1;

			classifiers_[a1].resize(noVals * a1);
			for (unsigned int v1 = 0; v1 < noVals; ++v1) {
				for (unsigned int a2 = 0; a2 < a1; ++a2) {

					if (selected_[a1][a2] == false)
						continue;
					if (a2 < noCatAtts_) {

						for (CatValue v2 = 0; v2 < aStream->getNoValues(a2);
								++v2) {

							thisStream.setSourceWithoutLoading(*aStream); // clear the stream

							aStream->rewind();

							while (aStream->advance()) {
								CatValue val1 = discretiser_->discretise(
										aStream->current()->getNumVal(
												a1 - noCatAtts_),
										a1 - noCatAtts_);
								if ((val1 == v1)
										&& (aStream->current()->getCatVal(a2)
												== v2)) {
									thisStream.add(aStream->current());
								}
							}

							classifiers_[a1][v1 * a1 + a2].push_back(
									createLearner(learnerName_, learnerArgv_,
											learnerArgEnd_));
							classifiers_[a1][v1 * a1 + a2].back()->train(
									thisStream);
						}

					} else {
						unsigned int noVals2;
						if (aStream->getMetaData()->hasNumMissing(
								a2 - noCatAtts_))

							noVals2 = discretiser_->getMetaData()->cuts[a2
									- noCatAtts_].size() + 2;
						else
							noVals2 = discretiser_->getMetaData()->cuts[a2
									- noCatAtts_].size() + 1;


						for (CatValue v2 = 0; v2 < noVals2; ++v2) {

							thisStream.setSourceWithoutLoading(*aStream); // clear the stream

							aStream->rewind();

							while (aStream->advance()) {
								CatValue val1 = discretiser_->discretise(
										aStream->current()->getNumVal(
												a1 - noCatAtts_),
										a1 - noCatAtts_);
								CatValue val2 = discretiser_->discretise(
										aStream->current()->getNumVal(
												a2 - noCatAtts_),
										a2 - noCatAtts_);

								if ((val1 == v1) && (val2 == v2)) {
									thisStream.add(aStream->current());
								}
							}

							classifiers_[a1][v1 * a1 + a2].push_back(
									createLearner(learnerName_, learnerArgv_,
											learnerArgEnd_));
							classifiers_[a1][v1 * a1 + a2].back()->train(
									thisStream);
						}
					}
				}
			}
		}
	}
}

void Feating2Learner::classify(const instance &inst, std::vector<double> &classDist) {
  std::vector<double> thisClassDist(classDist.size());

	classDist.assign(classDist.size(), 0.0);

	assert(classifiers_.size()==noAtts_);
	for (unsigned int a1 = 1; a1 < classifiers_.size(); ++a1) {

		if (a1 < noCatAtts_) {
			unsigned int v1 = inst.getCatVal(a1);
			for (unsigned int a2 = 0; a2 < a1; ++a2) {

			if(selected_[a1][a2]==false)
				continue;

				classifiers_[a1][v1 * a1 + a2][inst.getCatVal(a2)]->classify(
						inst, thisClassDist);

				if (useMajorityVoting_) {
					classDist[indexOfMaxVal(thisClassDist)] += 1.0;
				} else {
					for (CatValue y = 0; y < classDist.size(); ++y) {
						classDist[y] += thisClassDist[y];
					}
				}
			}
		} else {
			unsigned int v1 = discretiser_->discretise(
					inst.getNumVal(a1 - noCatAtts_), a1 - noCatAtts_);
			for (unsigned int a2 = 0; a2 < a1; ++a2) {

				if(selected_[a1][a2]==false)
					continue;

				if (a2 < noCatAtts_) {

					classifiers_[a1][v1 * a1 + a2][inst.getCatVal(a2)]->classify(
							inst, thisClassDist);

					if (useMajorityVoting_) {
						classDist[indexOfMaxVal(thisClassDist)] += 1.0;
					} else {
						for (CatValue y = 0; y < classDist.size(); ++y) {
							classDist[y] += thisClassDist[y];
						}
					}

				} else {
					unsigned int v2 = discretiser_->discretise(
							inst.getNumVal(a2 - noCatAtts_), a2 - noCatAtts_);

					classifiers_[a1][v1 * a1 + a2][v2]->classify(inst,
							thisClassDist);

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
	}
	normalise(classDist);
}
