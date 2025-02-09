/* Open source system for classification learning from very large data
** Class for an input source for instances
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

#include "instanceStream.h"
#include "utils.h"

InstanceStream::InstanceStream()
{
}

InstanceStream::~InstanceStream(void)
{
}


// output the petal format metadata description to a file
// attributes are ordered categorical first then numeric then the class
void InstanceStream::printMetadata(const char* filename) const {
  FILE *f = fopen(filename, "w");

  if (f == NULL) error("Cannot open metadata output file %s", filename);

  for (CategoricalAttribute a = 0; a < getNoCatAtts(); a++) {
    fprintf(f, "%s: ", getCatAttName(a));
    for (CatValue v = 0; v < getNoValues(a); v++) {
      if (v) fprintf(f, ", ");
      fputs(getCatAttValName(a, v), f);
    }
    putc('\n', f);
  }

  for (NumericAttribute a = 0; a < getNoNumAtts(); a++) {
    fprintf(f, "%s: numeric\n", getNumAttName(a));
  }

  fprintf(f, ":class:%s: ", getClassAttName());
  for (CatValue y = 0; y < getNoClasses(); y++) {
    if (y) fprintf(f, ", ");
    fputs(getClassName(y), f);
  }
  putc('\n', f);
  fclose(f);
}

InstanceStream::MetaData::~MetaData() {
}

InstanceStream::MetaDataFilter::~MetaDataFilter() {
}

unsigned int InstanceStream::MetaDataFilter::getNoClasses() const {
  return source_->getNoClasses();
}

const char* InstanceStream::MetaDataFilter::getClassName(const CatValue y) const {
  return source_->getClassName(y);
}

const char* InstanceStream::MetaDataFilter::getClassAttName() const {
  return source_->getClassAttName();
}

unsigned int InstanceStream::MetaDataFilter::getNoCatAtts() const {
  return source_->getNoCatAtts();
}

unsigned int InstanceStream::MetaDataFilter::getNoValues(const CategoricalAttribute att) const {
  return source_->getNoValues(att);
}

const char* InstanceStream::MetaDataFilter::getCatAttName(const CategoricalAttribute att) const {
  return source_->getCatAttName(att);
}

const char* InstanceStream::MetaDataFilter::getCatAttValName(const CategoricalAttribute att, const CatValue val) const {
  return source_->getCatAttValName(att, val);
}

unsigned int InstanceStream::MetaDataFilter::getNoNumAtts() const {
  return source_->getNoNumAtts();
}

const char* InstanceStream::MetaDataFilter::getNumAttName(const NumericAttribute att) const {
  return source_->getNumAttName(att);
}

unsigned int InstanceStream::MetaDataFilter::getPrecision(const NumericAttribute att) const {
  return source_->getPrecision(att);
}

const char* InstanceStream::MetaDataFilter::getName() const {
  return source_->getName();
}

bool InstanceStream::MetaDataFilter::areNamesCaseSensitive() const {
  return source_->areNamesCaseSensitive();
}

bool InstanceStream::MetaDataFilter::hasCatMissing(const CategoricalAttribute att) const {
  return source_->hasCatMissing(att);
}

bool InstanceStream::MetaDataFilter::hasNumMissing(const NumericAttribute att) const {
  return source_->hasNumMissing(att);
}

void InstanceStream::MetaDataFilter::setAllAttsMissing() {
  source_->setAllAttsMissing();
}
