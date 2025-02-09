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

#include "xyDist.h"
#include "utils.h"

#include <memory.h>

xyDist::xyDist() {
}

xyDist::xyDist(InstanceStream *is)
{
  reset(is);

  instance inst(*is);

  while (!is->isAtEnd()) {
    if (is->advance(inst)) update(inst);
  }
}

xyDist::~xyDist(void)
{
}

void xyDist::reset(InstanceStream *is) {

  metaData_ = is->getMetaData();
  noOfClasses_ = is->getNoClasses();
  count = 0;

  counts_.resize(is->getNoCatAtts());

  for (CategoricalAttribute a = 0; a < is->getNoCatAtts(); a++) {
    counts_[a].assign(is->getNoValues(a)*noOfClasses_, 0);
  }

  classCounts.assign(noOfClasses_, 0);
}

void xyDist::reset(InstanceStream *is, unsigned int noOrigCatAtts) {

  metaData_ = is->getMetaData();
  noOfClasses_ = is->getNoClasses();
  count = 0;

  counts_.resize(noOrigCatAtts);

  for (CategoricalAttribute a = 0; a < noOrigCatAtts; a++) {
    counts_[a].assign(is->getNoValues(a)*noOfClasses_, 0);
  }

  classCounts.assign(noOfClasses_, 0);
}

void xyDist::update(const instance &inst) {
  count++;

  const CatValue y = inst.getClass();

  classCounts[y]++;

  for (CategoricalAttribute a = 0; a < metaData_->getNoCatAtts(); a++) {
    counts_[a][inst.getCatVal(a)*noOfClasses_+y]++;
  }
}


void xyDist::outputDist()
{

	//0: zero instance
	//1: one instance
	//2:two instances
	//3: [3,254] instances
	//4: [0xFF,0xFFFFFFFE] instances
	//5:  [0xFFFFFFFF,0xFFFFFFFFFFFFFFFF]
	//6:  else
	std::vector<InstanceCount> dist(7);
	InstanceCount count;

	for(unsigned short i=0;i<7;i++ )
		dist[i]=0;

	for (CategoricalAttribute x1 = 0; x1 < metaData_->getNoCatAtts(); x1++) {

		for (CatValue v1 = 0; v1 < metaData_->getNoValues(x1); ++v1) {
					for (CatValue y = 0; y < noOfClasses_; y++) {

						count=getCount(x1,v1,y);

						if(count==0)
							dist[0]++;
						else if(count==1)
							dist[1]++;
						else if(count==2)
							dist[2]++;
						else if(count<255)
							dist[3]++;
						else if(count<0xFFFFFFFF)
							dist[4]++;
						else if(count<0xFFFFFFFFFFFFFFFF)
							dist[5]++;
						else
							dist[6]++;

					}
		}
	}

	printf("0:\t%u\t%s\n",dist[0],"0");
	printf("1:\t%u\t%s\n",dist[1],"1");
	printf("2:\t%u\t%s\n",dist[2],"2");
	printf("3:\t%u\t%s\n",dist[3],"[3,254]");
	printf("4:\t%u\t%s\n",dist[4],"[0xFF,0xFFFFFFFE]");
	printf("5:\t%u\t%s\n",dist[5],"[0xFFFFFFFF,0xFFFFFFFFFFFFFFFF]");
	printf("6:\t%u\t%s\n",dist[6],"else");

}

void xyDist::clear(){
  classCounts.clear();
  for (CategoricalAttribute a = 0; a < getNoAtts(); a++) {
    counts_[a].assign(metaData_->getNoValues(a)*noOfClasses_, 0);
  }
}
