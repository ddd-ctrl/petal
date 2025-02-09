/** \mainpage Petal: An open source system for classification learning from very large data
 * Copyright (C) 2012 Geoffrey I Webb
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
//#ifndef DBG_NEW
//#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
//#define new DBG_NEW
//#endif
#include <stdlib.h>
#include <crtdbg.h>
#endif
#endif

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <new>

#include "instanceFile.h"
#include "instanceStreamDiscretiser.h"
#include "instanceStreamDynamicDiscretiser.h"
#include "instanceStreamDynamicPIDDiscretiser.h"
#include "instanceStreamClassFilter.h"
#include "instanceStreamNormalisationFilter.h"
#include "instanceStreamFilter.h"
#include "learningCurves.h"
#include "learner.h"
#include "mtrand.h"
#include "utils.h"
#include "globals.h"
#include "FILEtype.h"
#include "learnerRegistry.h"
#include "ALGLIB_ap.h"
#include "FilterSet.h"
#include "syntheticInstanceStream.h"
#include "dataStatisticsAction.h"

// Train & test utilities
#include "trainTest.h"
#include "streamTest.h"
#include "xVal.h"
#include "biasvariance.h"
#include "externXVal.h"

/** 
 * Type of experiment and evaluation method.
 */
enum experimentType {
	etNone, /**< Nothing is done by default. */
	etStreamTest, /**< Use a stream test (specified with -s) */
	etTrainTest, /**< Use training set for testing (specified with -t) */
  etXVal, /**< Cross-validation (-x10 by default). */
	etBiasVariance, /**< Bias/variance experiments. */
	etExternXVal, /**< Learn a external classifier (e.g. libsvm) with petal
	 * folds. */
	etLearningCurves, /**< Do a learning curves experiment */
  etDataStats /**< Collect data statistics */
};

/**
 * @param argv Options for the experiment
 * @param argc Number of options
 * @return An integer 0 upon exit success
 */
int main(int argc, char* const argv[]) {
	MTRand rand;
	char* testfilename = NULL;
	char* metafilename = NULL;
	experimentType et = etNone;
	char* expArgs = NULL;
	std::vector<learner*> theLearners;
	char* const * eXValArgv = NULL;
	int eXValArgc = 0;
	char* const * argvEnd = argv + argc;
	FilterSet filters;
	LearningCurveArgs lcArgs;
	TrainTestArgs ttArgs;
	StreamTestArgs stArgs;
  DataStatisticsActionArgs dsArgs;
  InstanceStream* instanceStream = NULL;
  InstanceStream* testStream = NULL;

#ifdef _MSC_VER
#ifdef _DEBUG
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
#endif

	// First parse the command line arguments
	try {
		printf("======================\n"
				"Petal: the system for learning from big data\nVersion 0.2\n");
		for (int i = 0; i < argc; i++) {
			printf("%s ", argv[i]);
		}
		putchar('\n');
		putchar('\n');

		if (argc < 3) {
			error("Usage: %s <metafile> <trainingfile> [-p<posClassName>]"
					" [<test method args>] -l<learner> [<learner args>]",
					argv[0]);
		}

    if (streq(argv[1], "-syn")) {
      // a synthetic data stream test
      argv += 2; // skip the program name, and this argument
      instanceStream = new SyntheticInstanceStream(argv, argvEnd);
    }
    else {
      // 来自数据文件的标准输入
		  metafilename = argv[1];
      
      instanceStream = new InstanceFile(argv[1], argv[2]);

		  argv += 3; // skip the program name, the meta file name and the data file name
    }

    ArgParensCheck parensCheck;

    while (argv != argvEnd) {
      if (parensCheck.check(argv)) {
        ++argv;
      }
      else {
        if (**argv != '-') {
          error("Argument '%s' requires '-'", *argv);
        }

        char *p = argv[0] + 1;

        switch (*p) {
        case 'b':
          // 使用偏差-方差实验
		  if (et != etNone) error("Only one action can be specified");//只能指定一个操作
		  et = etBiasVariance;
          expArgs = p + 1;
          ++argv;
          break;
        case 'c':
          // 学习曲线
			if (et != etNone) error("Only one action can be specified");
			et = etLearningCurves;
          lcArgs.getArgs(++argv, argvEnd);
          break;
        case 'd':
          // discretise
          if (streq(p+1, "dynamic")) {
            filters.push_back(new InstanceStreamDynamicDiscretiser(++argv, argvEnd));
          }
          else if (streq(p+1, "dynamicPID")) {
            filters.push_back(new InstanceStreamDynamicPIDDiscretiser(++argv, argvEnd));
          }
          else {
            filters.push_back(new InstanceStreamDiscretiser(p + 1, ++argv, argvEnd));
          }
          break;
        case 'e':
          // 使用外部交叉验证实验
		  if (et != etNone) error("Only one action can be specified");
		  et = etExternXVal;

          expArgs = p + 1;

          // 收集所有参数并将其传递给 xval
          eXValArgv = ++argv;
          eXValArgc = argvEnd - argv;
          argv = argvEnd;
          break;
        case 'l':
          // 指定学习器

          // 创建学习器
          theLearners.push_back(createLearner(p + 1, ++argv, argvEnd));

          if (theLearners.back() == NULL) {
            std::vector<std::string> learners;
            
            errorMsg("Learner %s is not supported.\nAvailable learners are:", p + 1);
            getLearnerRegistry().getLearnerList(learners);

            for (std::vector<std::string>::const_iterator it = learners.begin(); it != learners.end(); it++) {
              errorMsg("- %s", it->c_str());
            }
            exit(0);
          }
          break;
        case 'n':
          filters.push_back(new InstanceStreamNormalisationFilter(++argv, argvEnd));
          break;
        case 'p':
          // 将类别筛选为二元分类
          // 不能添加到筛选器中，因为这会改变类的数量
          instanceStream = new InstanceStreamClassFilter(instanceStream, p + 1, ++argv, argvEnd);
          break;
        case 's':
          // 使用流测试实验
		  if (et != etNone) error("Only one action can be specified");
		  et = etStreamTest;
          stArgs.getArgs(++argv, argvEnd);
          break;
        case 't':
          // 使用训练文件-测试文件实验
          // 测试文件名必须跟在 t后面
		  if (et != etNone) error("Only one action can be specified");
		  et = etTrainTest;
          testfilename = p + 1;
          testStream = new InstanceFile(metafilename, testfilename);

          ttArgs.getArgs(++argv, argvEnd);
          break;
        case 'v':
          // set the verbosity level - the default is 1
          getUIntFromStr(p + 1, verbosity, "verbosity");
          ++argv;
          break;
        case 'x':
          // 使用交叉验证实验
		  if (et != etNone) error("Only one action can be specified");
		  et = etXVal;
          expArgs = p + 1;
          ++argv;
          break;
        case 'z':
          // 获取数据统计
		  if (et != etNone) error("Only one action can be specified");
		  et = etDataStats;
          dsArgs.getArgs(++argv, argvEnd);
          break;
        default:
          error("-%c flag is not supported", *p);
        }
      }
    }

    if (!parensCheck.balanced()) {
      error("Parentheses do not balance.");
    }

		if (et == etExternXVal) {
			externXVal(instanceStream, filters, expArgs, eXValArgv, eXValArgc);
		}
    else {
			if (theLearners.empty() && et != etDataStats) {
				error("No learner specified");
			}

			// 进行实验
			switch (et) {
			case etStreamTest:
				if (theLearners.size() > 1)
					error("Stream test only accepts a single learner");

				streamTest(theLearners[0], *instanceStream, filters, stArgs);
				break;
			case etTrainTest:
				if (theLearners.size() > 1)
					error("Train/test only accepts a single learner");

				trainTest(theLearners[0], *instanceStream, *testStream, filters, ttArgs);
				break;
			case etXVal:
				if (theLearners.size() > 1)
					error("Cross validation only accepts a single learner");

				xVal(theLearners[0], *instanceStream, filters, expArgs);
				break;
			case etBiasVariance:
				if (theLearners.size() > 1)
					error("Bias Variance only accepts a single learner");

				biasVariance(theLearners[0], *instanceStream, filters, expArgs);
				break;
			case etLearningCurves:
				genLearningCurves(theLearners, *instanceStream, filters, &lcArgs);
				break;
      case etDataStats:
        dataStatisticsAction(*instanceStream, filters, dsArgs);
        break;
			default:
				error("No action specified");
				break;
			}

			for (std::vector<learner*>::iterator it = theLearners.begin();
					it != theLearners.end(); it++) {
				delete *it;
			}
		}
	} catch (std::bad_alloc) {
		error("Out of memory");
	} catch (alglib::ap_error err) {
		error(err.msg.c_str());
	}

	if (verbosity >= 1)
		summariseUsage();

	if (instanceStream != NULL) delete instanceStream;
	if (testStream != NULL) delete testStream;
	
	return 0;
}

