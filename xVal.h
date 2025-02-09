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
#include "instanceStream.h"
#include "FilterSet.h"
#include "learner.h"

class XValArgs {
public:
	XValArgs() : logProgression_(true), testSetSize_(1000), noOfTrials_(10), noOfPoints_(20), startingPoint_(8), endingPoint_(std::numeric_limits<InstanceCount>::max()), csvFormat_(false) {}

	void getArgs(char*const*& argv, char*const* end);  // get settings from command line arguments

	bool logProgression_;
	InstanceCount testSetSize_;
	unsigned int noOfTrials_;
	unsigned int noOfPoints_;
	InstanceCount startingPoint_;
	InstanceCount endingPoint_;
	bool csvFormat_;   ///< the learning curves should be written in CSV format
};

void xVal(learner *theLearner, InstanceStream &instStream, FilterSet &filters, char* args);
