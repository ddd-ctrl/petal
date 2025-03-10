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
** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
*/


/**
<!-- globalinfo-start -->
 * Class for discretiser that dynamically dicretises into bins containing as close as possible to equal numbers of instances.<p>
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

#pragma once

#include "discretiser.h"

#include <vector>


class eqDepthDiscretiser : public discretiser
{
public:
  eqDepthDiscretiser(char*const*& argv, char*const* end);
  ~eqDepthDiscretiser(void);

  virtual void discretise(std::vector<NumValue> &vals, const std::vector<CatValue> &classes, unsigned int noOfClasses, std::vector<NumValue> &cuts); // discretize Attribute att with values vals

protected:
  typedef enum {specifiedIntervals, pkid, pkidCubed, npkid} discretisationType;

  unsigned int intervals; ///< the number of intervals into which the data should be discretised
  unsigned int root_;     ///< the n in the nth-root used to determine the pkdd interval size
  bool exact_;            ///< if true then discretise into the exact number of intervals, even if some are empty
  discretisationType discType;
};
