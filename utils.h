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
// Utility functions
#pragma once

#include "mtrand.h"
#include "FILEtype.h"
#include "crosstab.h"
#include "crosstab3d.h"
#include "instanceStream.h"

#include <time.h>
#include <vector>
#include <deque>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <string>
#include <algorithm>

/// print an error mesage and exit.  Supports printf style format and arguments
#ifdef _MSC_VER
__declspec(noreturn)
#endif
  void error(const char *fmt, ...)
#ifdef __GNUC__
  __attribute__ ((noreturn))
#endif
  ;

/// print an error mesage without exiting.  Supports printf style format and arguments
void errorMsg(const char *fmt, ...);

template <typename T>
inline int printWidth(T val) {
  return static_cast<int>(ceil(log(static_cast<double>(val))/log(10.0)));
}

void printResults(crosstab<InstanceCount> &xtab, const InstanceStream &instanceStream);

template <typename T>
inline T max(std::vector<T> v) {
  assert(v.size() > 0);

  T maxVal = v[0];

  for (unsigned int i = 1; i < v.size(); i++) {
    if (v[i] > maxVal) maxVal = v[i];
  }

  return maxVal;
}

template <typename T>
inline unsigned int indexOfMinVal(std::vector<T> v) {
  unsigned int mini = 0;

  for (unsigned int i = 1; i < v.size(); i++) {
    if (v[i] < v[mini]) mini = i;
  }

  return mini;
}

template <typename T>
inline unsigned int indexOfMaxVal(std::vector<T> v) {
  unsigned int maxi = 0;

  for (unsigned int i = 1; i < v.size(); i++) {
    if (v[i] > v[maxi]) maxi = i;
  }

  return maxi;
}


template <typename T>
inline void normalise(crosstab3D<T> &v) {
	T sum = 0;
	unsigned int length = v.size();
	for (unsigned int i = 0; i < length; i++) {
		for (unsigned int j = 0; j < length; j++) {
			for (unsigned int k = 0; k < length; k++) {
				sum += v[i][j][k];
			}
		}
	}

	assert(sum!=0);
	for (unsigned int i = 0; i < length; i++) {
		for (unsigned int j = 0; j < length; j++) {
			for (unsigned int k = 0; k < length; k++) {
				v[i][j][k]  /= sum;
			}
		}
	}
}
template <typename T>
inline void normalise(crosstab<T> &v) {
	T sum = 0;
	unsigned int length = v.size();
	for (unsigned int i = 0; i < length; i++) {
		for (unsigned int j = 0; j < length; j++) {
			sum += v[i][j];
		}
	}

	assert(sum!=0);
	for (unsigned int i = 0; i < length; i++) {
		for (unsigned int j = 0; j < length; j++) {
			v[i][j] /= sum;
		}
	}
}


template <typename T>
inline void normalise(std::vector<T> &v) {
  T sum = v[0];

  for (unsigned int i = 1; i < v.size(); i++) {
    sum += v[i];
  }

  assert(sum!=0);

  for (unsigned int i = 0; i < v.size(); i++) {
    v[i] /= sum;
  }
}

template <typename T>
inline T sum(const std::vector<T> &v) {
  T sum = 0;

  for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
    sum += *it;
  }

  return sum;
}

template <typename T>
inline T mean(std::vector<T> &v) {
  return sum(v) / static_cast<T>(v.size());
}


template <typename T>
inline T stddev(std::vector<T> &v) {
  T m = mean(v);
  T devsq = 0;
  for (unsigned int i = 0; i < v.size(); i++) {
    const T dev = (v[i]-m);
    devsq += dev * dev;
  }

  return sqrt(devsq/(v.size()-1));
}

template <typename T>
inline T sum(const std::deque<T> &v) {
  T sum = 0;

  for (std::deque<double>::const_iterator it = v.begin(); it != v.end(); ++it) {
    sum += *it;
  }

  return sum;
}

template <typename T>
inline T mean(std::deque<T> &v) {
  return sum(v) / static_cast<T>(v.size());
}

// 从字符串中解析一个无符号 int。
// 模板，以便允许任何大小的无符号 ints
template <typename T>
void getUIntFromStr(char const *s, T &val, char const *context) {
  if (s == NULL) return;

  if (*s != '\0') {
    if (*s < '0' || *s > '9') {
      error("Encountered '%s' when expecting an unsigned integer for %s", s, context);
    }
    unsigned int v = 0;
    while (*s >= '0' && *s <= '9') {
      v *= 10;
      v += *s++ - '0';
    }
    val = v;
    if (*s != '\0') {
      error("Encountered '%s' when expecting an unsigned integer for %s", s, context);
    }
  }
}

unsigned int getUIntListFromStr(char *s, std::vector<unsigned int*> &vals, char const *context);

// parse a floating point number from a string.
// template used so as to allow floats of any size
template <typename T>
void getFloatFromStr(char const *s, T &val, char const *context) {
  if (s == NULL) return;

  if (*s == '\0') {
    error("Missing number for %s", context);
  }
  else {
    if ((*s < '0' || *s > '9') && *s != '.') {
      error("Encountered '%s' when expecting a floating point value for %s", s, context);
    }
    double v = 0;
    while (*s >= '0' && *s <= '9') {
      v *= 10;
      v += *s++ - '0';
    }
    val = v;

    if (*s == '.') {
      double p = 1;
      s++;
      while (*s >= '0' && *s <= '9') {
        p *= 10;
        val += (*s++ - '0') / p;
      }
    }

    if (*s != '\0') {
      error("Encountered '%s' when expecting a floating point value for %s", s, context);
    }
  }
}

InstanceCount countCases(FILEtype *f);

template <typename T>
inline void randomise(std::vector<T> &order) {
  MTRand_int32 rand(static_cast<unsigned long>(time(NULL)));

  std::vector<T> newOrder;

  while (!order.empty()) {
    const unsigned int i = rand() % order.size();
    newOrder.push_back(order[i]);
    order.erase(order.begin()+i);
  }

  order = newOrder;
}

template <typename T>
inline T max(const T v1, const T v2) {
  if (v1 > v2) return v1;
  else return v2;
}

template <typename T>
inline T min(const T v1, const T v2) {
  if (v1 < v2) return v1;
  else return v2;
}

// allocate a single object
template <typename T>
inline void safeAlloc(T *&object) {
#ifdef _DEBUG
  try {
    object = new T;
  }
  catch (std::bad_alloc) {
    error("Out of memory");
  }
#else // _DEBUG
  object = new T;
#endif // _DEBUG
}

// allocate an array of objects
template <typename T>
inline void safeAlloc(T *&object, const unsigned int size) {
#ifdef _DEBUG
  try {
    object = new T[size];
  }
  catch (std::bad_alloc) {
    error("Out of memory");
  }
#else // _DEBUG
    object = new T[size];
#endif // _DEBUG
}

// allocate an array of objects and initialise to 0
template <typename T>
inline void allocAndClear(T *&object, const unsigned int size) {
#ifdef _DEBUG
  try {
    object = new T[size];
    memset(object, 0, size * sizeof(T));
  }
  catch (std::bad_alloc) {
    error("Out of memory");
  }
#else // _DEBUG
  object = new T[size];
  memset(object, 0, size * sizeof(T));
#endif // _DEBUG
}

// compare two attributes on the values stored in an array
// appropriate comparator for descending order
template <typename T>
class IndirectCmpClass {
public:
  IndirectCmpClass(T *p) : pv(p) {
  }

  bool operator() (unsigned int a, unsigned int b) {
    return pv[a] > pv[b];
  }

private:
  const T *pv;
};

// compare two attributes on the values stored in an array, resolving draws using a second array
template <typename T, typename T2>
class IndirectCmpClass2 {
public:
  IndirectCmpClass2(std::vector<T> &p, std::vector<T2> &p2) : pv(p), pv2(p2) {
  }

  bool operator() (unsigned int a, unsigned int b) {
    return pv[a] > pv[b] || (pv[a] == pv[b] && pv2[a] > pv2[b]);
  }

private:
  const std::vector<T> &pv;
  const std::vector<T2> &pv2;
};

// compare two attributes on the values stored in an array
// appropriate comparator for ascending order
template <typename T>
class IndirectCmpClassAscending {
public:
  IndirectCmpClassAscending(T *p) : pv(p) {
  }

  bool operator() (unsigned int a, unsigned int b) const {
    return pv[a] < pv[b];
  }

private:
  const T *pv;
};


// true iff two strings are identical, case insensitive
bool streq(char const *s1, char const *s2, const bool caseSensistive = false);

// output summary of process usage
void summariseUsage();

// a vector of ptrs to objects.  On destruction, the class is responsible for deleting the object to which it points
// Used instead of a vector of auto_ptrs because we want to be able to utilise temporary pointers to the same object
// Used instead of shared_ptr because we do not need the overhead of tracking multiple pointers
template <typename T>
class ptrVec : public std::vector<T*> {
public:
  ptrVec<T>() {}

  ~ptrVec<T>() {
    typedef typename ptrVec<T>::iterator iterator;
    for (iterator i = this->begin(); i != this->end(); i++) delete *i;
    //for (int i = 0; i < this->size(); i++) delete (*this)[i];
  }

  void clear()  {
    typedef typename ptrVec<T>::iterator iterator;
    for (iterator i = this->begin(); i != this->end(); i++) delete *i;
    //for (int i = 0; i < this->size(); i++) delete (*this)[i];
    std::vector<T*>::clear();
  }
};

// a two-dimensional fixed size array whose dimensions are not known at compile time
template <typename T>
class fdarray {
public:
  fdarray<T>() : d2size_(0) {
  }

  fdarray<T>(const unsigned int d1sz, const unsigned int d2sz) : store_(d1sz*d2sz), d2size_(d2sz)
    #ifndef NDEBUG
      , d1size_(d1sz)
    #endif
  {
  }

  ~fdarray<T>() {
  }

  inline void clear() {
    store_.assign(store_.size(), 0);
  }

  inline void resize(const unsigned int d1sz, const unsigned int d2sz) {
    store_.resize(d1sz*d2sz);
    d2size_ = d2sz;
    #ifndef NDEBUG
      d1size_ = d1sz;
    #endif
  }

  inline void assign(const unsigned int d1sz, const unsigned int d2sz, T v)  {
    store_.assign(d1sz*d2sz, v);
    d2size_ = d2sz;
    #ifndef NDEBUG
      d1size_ = d1sz;
    #endif
  }

  inline T& ref(unsigned int a, unsigned int b) {
    #ifndef NDEBUG
      assert(a >=0 && a < d1size_);
    #endif
    assert(b >= 0 && b < d2size_);
    return store_[a*d2size_+b];
  }

  inline T* operator[](unsigned int a) {
    #ifndef NDEBUG
      assert(a >=0 && a < d1size_);
    #endif
    return &store_[a*d2size_];
  }

  inline unsigned int getDim(){
    return store_.size();
  }

private:
  std::vector<T> store_;
  unsigned int d2size_;
#ifndef NDEBUG
  unsigned int d1size_;
#endif
};

void print(const std::vector<double> &vals);

void print(const std::vector<float> &vals);

void print(const std::vector<long int> &vals);

void print(const std::vector<unsigned int> &vals);

/// print a string replacing ' ' with '_'
inline void print_(FILE* f, std::string &s) {
  for (std::string::iterator it = s.begin(); it != s.end(); it++) {
    if (*it == ' ' || *it == '-') fputc('_', f);
    else fputc(*it, f);
  }
}

CatValue discretise(const NumValue val, std::vector<NumValue> &cuts);

#ifdef _MSC_VER
inline double log2( const double n ) {
    // log2(n) = log10(n)/log10(2)
	//change log to log10, since log actully represent the log-base e.
    return log10( n ) / 0.30102999566398119521373889472449;
}
#endif

void calcAUPRC(
        std::vector<std::vector<float> >& probs, //< the sequence of predicted probabilitys for each class
        const std::vector<CatValue>& trueClasses, //< the sequence of true classes
        InstanceStream::MetaData& metadata
        );


double calcBinaryAUC(
        std::vector<std::vector<double> >& probs, //< the sequence of predicted probabilities for each class
        const std::vector<CatValue>& trueClasses //< the sequence of true classes
        );

double calcMultiAUC(
        std::vector<std::vector<double> >& probs, //< the sequence of predicted probabilities for each class
        const std::vector<CatValue>& trueClasses //< the sequence of true classes
        );


/// This class is used to do parenthesis checking during argument processing
class ArgParensCheck {
public:
  ArgParensCheck() : count_(0) {};

  inline bool check(char* const *& argv) {
    if (**argv == '[') {
      count_++;
      return true;
    }
    else if (**argv == ']' && count_ > 0) {
      count_--;
      return true;
    }
    else return false;
  }

  inline bool balanced() { return count_ == 0; }

private:
  int count_; ///< a count of the number of open parentheses
};

class RandNorm {
public:
  // returns a random value that is normally distributed with the specified mean and standard deviation
  double operator ()(const double mean, const double stddev) {
    const double norm = sqrt(-2*log(static_cast<double>(random_()))) * cos(2*3.14159265359*random_());
    return mean + stddev * norm;
  }

  void reset(const unsigned long seed) { random_.seed(seed); }

private:
  // returns a random value in the interval (0,1]
  MTRand random_;
};
