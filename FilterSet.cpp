
#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#endif

#include "FilterSet.h"

FilterSet::FilterSet(void)
{
}

FilterSet::~FilterSet(void)
{
}

/// create a new InstanceStream by applying the set of filters
InstanceStream* FilterSet::apply(InstanceStream* source) {
  FilterSet::iterator it = begin();

  while (it != end()) {
    (*it)->setSource(*source);
    source = *it;
    ++it;
  }

  return source;
}
