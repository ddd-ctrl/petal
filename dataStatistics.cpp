/* Open source system for classification learning from very large data
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

#include <assert.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#include "utils.h"
#include "dataStatistics.h"
#include "learnerRegistry.h"


static LearnerRegistrar registrar("dataStat", constructor<dataStatistics>);

dataStatistics::dataStatistics(char* const *& argv, char* const * end) :
		trainingIsFinished_(false) {
	name_ = "dataStatistics";
	trainingIsFinished_ = false;
        
    dt_=dtXy;

	// get arguments
	while (argv != end) {
		if (*argv[0] != '+') {
			break;
		} else if (streq(argv[0] + 1, "xy")) {
			dt_=dtXy;
		} else if (streq(argv[0] + 1, "xxy")) {
			dt_=dtXxy;
		} else if (streq(argv[0] + 1, "xxxy")) {
			dt_=dtXxxy;
		} else if (streq(argv[0] + 1, "xxxxy")) {
			dt_=dtXxxxy;
		} else {
			error("dataStat does not support argument %s\n", argv[0]);
			break;
		}

		name_ += *argv;

		++argv;
	}
}

dataStatistics::~dataStatistics(void) {
}

void dataStatistics::getCapabilities(capabilities &c) {
	c.setCatAtts(true); 
        c.setNumAtts(true);
}

void dataStatistics::reset(InstanceStream &is) {

	switch (dt_){
	case dtXy:
		xyDist_.reset(&is);
		break;
	case dtXxy:
		xxyDist_.reset(is);
		break;
	case dtXxxy:
		xxxyDist_.reset(is);
		break;
	case dtXxxxy:
		xxxxyDist_.reset(is);
		break;
	}
	noCatAtts_ = is.getNoCatAtts();
	noClasses_ = is.getNoClasses();
        noNumAtts_ = is.getNoNumAtts();
        noInstances_ = 0;

        printf("No of values for each attributes:\n");
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            avNoAttValues_ += is.getNoValues(a);
            printf("%u:\t%u\n",a,is.getNoValues(a));
	}
        dataName_ = is.getName();
        classCount.assign(noClasses_,0);
	trainingIsFinished_ = false;
}

void dataStatistics::train(const instance &inst) {

	switch (dt_){
	case dtXy:
		xyDist_.update(inst);
		break;
	case dtXxy:
		xxyDist_.update(inst);
		break;
	case dtXxxy:
		xxxyDist_.update(inst);
		break;
	case dtXxxxy:
		xxxxyDist_.update(inst);
		break;
	}

	noInstances_++;
    classCount[inst.getClass()]++;
}

void dataStatistics::initialisePass() {
}

void dataStatistics::finalisePass() {


	trainingIsFinished_ = true;
        printf("\n\n--- Statistics for %s ---\n\n",dataName_.c_str());
        printf("# instances: %d\n", noInstances_);
        printf("# attributes: %d\n", noNumAtts_+noCatAtts_);
          printf("\t# categorical atts: %d\n", noCatAtts_);
            printf("\taverage attribute values: %.2f\n", avNoAttValues_/noCatAtts_);
          printf("\t# Numerical atts: %d\n", noNumAtts_);
        printf("# classes: %d\n", noClasses_);
        printf("class distribution:\n");
        const char *sep = "";
          for (std::vector<unsigned int>::const_iterator it = classCount.begin(); it != classCount.end(); it++) {
            printf("%s%0.4f", sep, static_cast<double>(*it)/noInstances_);
            sep = ", ";
          }
        putchar('\n');

        //output the distribution
        printf("instance distribution:\n");

    	switch (dt_){
    	case dtXy:
    		xyDist_.outputDist();
    		break;
    	case dtXxy:
    		xxyDist_.outputDist();
    		break;
		case dtXxxy:
			xxxyDist_.outputDist();
			break;
		case dtXxxxy:
			xxxxyDist_.outputDist();
			break;
    	}
}

bool dataStatistics::trainingIsFinished() {

	return trainingIsFinished_;
}

void dataStatistics::classify(const instance &inst, std::vector<double> &classDist) {

}

