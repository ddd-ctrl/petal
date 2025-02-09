/* Open source system for classification learning from very large data
** Abstract class for a filter for an instance stream
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
#pragma once

#include "instanceStream.h"

/**
<!-- globalinfo-start -->
 * Class for a filtered instance stream.<p>
 * The filter can be used in either of two modes<br>
 * - directly as an instance stream, in which case it extracts the instances from the source stream and converts them to serve up to the consumer, or<br>
 * - the convert() method can directly translate an instance from the base stream into an instance for the output stream.<p>
 * The default methods pass all calls through to the source stream.<p>
 * The ClassSafeInstanceStreamFilter subclass is for stream filters that do not change the number of classes<p>
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

class InstanceStreamFilter: public InstanceStream
{
public:
  InstanceStreamFilter();
  virtual ~InstanceStreamFilter(void);

  virtual void rewind();                                    ///< return to the first instance in the stream
  virtual bool advance();                                   ///< advance, discarding the next instance in the stream.  Return true iff successful.
  virtual bool advance(instance &inst);                     ///< advance to the next instance in the stream.  Return true iff successful. @param inst the instance record to receive the new instance. 
  virtual bool isAtEnd() const;                                   ///< true if we have advanced past the last instance
  virtual InstanceCount size();                             ///< the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.

  virtual void setSource(InstanceStream &source);           ///< set the source for the filter

  virtual void convert(const instance &in, instance &out);  ///< create a new instance for the output stream by converting an instance from the source stream

protected:
  InstanceStream *source_; ///< the source instance stream
};

// An InstanceStreamFilter that does not change the number of classes
class ClassSafeInstanceStreamFilter: public InstanceStreamFilter
{
};

//class FilterRegistry {
//public:
//  inline void addFilter(char* n, InstanceStreamFilter* (*c)()) { filters_.push_back(*(new FilterRec(n,c))); }
//
//private:
//  class FilterRec {
//  public:
//    FilterRec(char* n, InstanceStreamFilter* (*c)()) : name(n), constructor(c) {};
//
//  private:
//    char* name;
//    InstanceStreamFilter* (*constructor)();
//  };
//
//  std::vector<FilterRec> filters_;
//};
//
//// must encapsulate registry in a function to ensure initialisation before first use (as it is called during static initialisation)
//FilterRegistry& theFilterRegistry();
//
//class RegisterFilter {
//  RegisterFilter(char* n, InstanceStreamFilter* (*c)()) { theFilterRegistry().addFilter(n, c); }
//}
