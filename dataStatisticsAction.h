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

/**
<!-- globalinfo-start -->
 * Procedure for printing dataset's statistics such as # of instances, attributes, classes, class distribution, etc.<br/>
 * <br/>
 <!-- globalinfo-end -->
 *
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu) 
 */

#include "instanceFile.h"
#include "instanceStream.h"
#include "FilterSet.h"


class DataStatisticsActionArgs {
public:
  DataStatisticsActionArgs() : numericRuns_(false) {}

  void getArgs(char*const*& argv, char*const* end);  // get settings from command line arguments

  bool numericRuns_;
};

/// train a learner from a training set and test against a test set read from a file
/// @param instStream the instance stream to collect stats from
/// @param filters any filters to apply before collecting stats
void dataStatisticsAction(InstanceStream &instStream, FilterSet &filters, const DataStatisticsActionArgs &args);
