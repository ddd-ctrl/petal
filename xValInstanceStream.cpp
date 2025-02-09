
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

#include "xValInstanceStream.h"

XValInstanceStream::XValInstanceStream(InstanceStream *source, const unsigned int noOfFolds, const unsigned int seed)
  : source_(source), seed_(seed), noOfFolds_(noOfFolds)
{ metaData_ = source->getMetaData();
  startSubstream(0, true);
}

XValInstanceStream::~XValInstanceStream(void)
{
}

/// 开始新的训练或测试
void XValInstanceStream::startSubstream(const unsigned int fold, const bool training) {
  fold_ = fold;
  training_ = training;
  rewind();
}

/// 返回数据流中的第一个实例
void XValInstanceStream::rewind() {
  source_->rewind();
  rand_.seed(seed_);
  count_ = 0;
}

/// advance, discarding the next instance in the stream.  Return true iff successful.
bool XValInstanceStream::advance() {
  while (source_->advance()) {
    if (rand_() % noOfFolds_ == fold_) {
      // test instance
      if (!training_) {
        count_++;
        return true;
      }
    }
    else {
      // training instance
      if (training_) {
        count_++;
        return true;
      }
    }
  }
  return false;
}

/// advance to the next instance in the stream.Return true iff successful. @param inst the instance record to receive the new instance. 
bool XValInstanceStream::advance(instance &inst) {
  while (!source_->isAtEnd()) {
    if (rand_() % noOfFolds_ == fold_) {
      // test instance
      if (training_) {
        source_->advance();
      }
      else {
        // testing
        if (source_->advance(inst)) {
          count_++;
          return true;
        }
        else return false;
      }
    }
    else {
      // training instance
      if (training_) {
        if (source_->advance(inst)) {
          count_++;
          return true;
        }
        else return false;
      }
      else {
        // testing
        source_->advance();
      }
    }
  }
  return false;
}

/// true if we have advanced past the last instance
bool XValInstanceStream::isAtEnd() const {
  return source_->isAtEnd();
}

/// the number of instances in the stream. This may require a pass through the stream to determine so should be used only if absolutely necessary.  The stream state is undefined after a call to size(), so a rewind shouldbe performed before the next advance.
InstanceCount XValInstanceStream::size() {
  if (!isAtEnd()) {
    instance inst(*this);

    while (!isAtEnd()) advance(inst);
  }

  return count_;
}

