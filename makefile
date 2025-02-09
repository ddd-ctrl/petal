CC      = g++
CFLAGS  = -O3 -DNDEBUG -c -Wall
#CFLAGS = -c -Wall
LDFLAGS =

## PLEASE KEEP THE LIST OF SOURCE FILES IN ALPHABETIC ORDER
SOURCE  = a2de.cpp a2de_diselect.cpp a2de_ms.cpp a2de2.cpp a2je.cpp a3de.cpp a3de_ms.cpp AdaBoost.cpp ALGLIB_ap.cpp ALGLIB_specialfunctions.cpp alglibinternal.cpp aode.cpp aodeBSE.cpp aodeDist.cpp aodeEager.cpp aodeExt.cpp baggedLearner.cpp biasvariance.cpp biasVarianceInstanceStream.cpp capabilities.cpp cbn.cpp correlationMeasures.cpp dataStatistics.cpp dataStatisticsAction.cpp discretiser.cpp distributionTree.cpp distributionTreeEager.cpp DTree.cpp ensembleLearner.cpp eqDepthDiscretiser.cpp externLearner.cpp externXVal.cpp extLearnLibSVM.cpp extLearnOMCLPBoost.cpp extLearnPetal.cpp extLearnSVMSGD.cpp extLearnVW.cpp extLearnWeka.cpp Feating2Learner.cpp Feating3Learner.cpp featingLearner.cpp filteredLearner.cpp FilterSet.cpp globals.cpp gnb.cpp incrementalLearner.cpp IndirectInstanceSample.cpp IndirectInstanceStream.cpp IndirectInstanceSubstream.cpp instance.cpp instanceFile.cpp instanceSample.cpp instanceStream.cpp instanceStreamClassFilter.cpp instanceStreamDiscretiser.cpp instanceStreamDynamicDiscretiser.cpp instanceStreamDynamicPIDDiscretiser.cpp instanceStreamFilter.cpp instanceStreamNormalisationFilter.cpp instanceStreamQuadraticFilter.cpp kdb.cpp kdbExt.cpp kdbGaussian.cpp kdbSelective.cpp kdbSelectiveClean.cpp lbfgs.c learner.cpp learnerRegistry.cpp learningCurves.cpp lr.cpp LR_SGD.cpp MDLBinaryDiscretiser.cpp MDLDiscretiser.cpp mtrand.cpp nb.cpp petal.cpp random.cpp RFDTree.cpp sampler.cpp ShuffledInstanceStream.cpp stackedLearner.cpp StoredIndirectInstanceStream.cpp StoredInstanceStream.cpp streamTest.cpp subspaceLRSGD.cpp syntheticInstanceStream.cpp tan.cpp trainTest.cpp utils.cpp xVal.cpp xValInstanceStream.cpp xxxxyDist.cpp xxxxyDist3.cpp xxxyDist.cpp xxxyDist2.cpp xxxyDist3.cpp xxxyDist4.cpp  xxyDist.cpp xxyDistEager.cpp xyDist.cpp xyGaussDist.cpp yDist.cpp
# Removed experimental learners: kdbCDdisc.cpp kdbCDRAM.cpp kdbEager.cpp 

OBJECTS=$(SOURCE:.cpp=.o)
EXECUTABLE=petal

default: $(EXECUTABLE)

depend: .depend

.depend: $(SOURCE)
	rm -f ./.depend
	$(CC) $(CFLAGS) -MM $^ >> ./.depend;

include .depend

petal: ${OBJECTS}
	$(CC) -o $@ ${OBJECTS} $(LDFLAGS)

petal64: ${SOURCE}
	$(CC) -o $@ ${SOURCE} $(CFLAGS) -DSIXTYFOURBITCOUNTS

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
