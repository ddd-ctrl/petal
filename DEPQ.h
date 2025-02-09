#include <vector>
#include <functional>   // std::less
#include <limits>
#include <assert.h>
#include <exception>

/**
<!-- globalinfo-start -->
 * Class for a Double ended priority queue implemented as an interval heap.<p>
 * Dynamic discretisation creates the cut points incrementally and modifies them as more data are seen.<p>
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */

template<class _Ty,
         class _Pr = std::less<_Ty> >  // _Pr is the comparator
	class DEPQ {	
public:
  // construct empty queue with default comparator
  DEPQ() : isOdd_(false)
  {
  }

  // construct empty queue, specify comparator
  explicit DEPQ(const _Pr& _Pred)
		: isOdd_(false), comp_(_Pred)
	{	
	}

  // empty the queue
  void clear() {
    container_.clear();
    isOdd_ = false;
  }
  
  _Ty min() const {
    assert(!empty());

    return container_[0].lower;
  }
  
  _Ty max() const {
    assert(!empty());

    if (size() == 1) return container_[0].lower;
    else return container_[0].upper;
  }

  // add an element
  void push(_Ty v) {
    if (isOdd_) {
      // insert v into the last interval which contains only one value
      if (container_.back().lower > v) {
        container_.back().upper = container_.back().lower;
        bubbleUpLower(container_.size()-1, v);
      }
      else {
        bubbleUpUpper(container_.size()-1, v);
      }
      isOdd_ = false;
    }
    else {
      // create a new interval containing only one element
      const unsigned int n = container_.size();          // the index of the new container

      container_.resize(n+1);
      isOdd_ = true;
      if (container_.size() == 1) {
        // container was empty so job done
        container_.front().lower = v;
      }
      else {
        // check whether we need to bubble up
        const unsigned int p = parent(n);

        if (v < container_[p].lower) {
          container_[n].lower = container_[p].lower;
          bubbleUpLower(p, v);
        }
        else if (v > container_[p].upper) {
          container_[n].lower = container_[p].upper;
          bubbleUpUpper(p, v);
        }
        else {
          container_[n].lower = v;
        }
      }
    }
  }

  // remove the lowest value element
  _Ty pop_min() {
    if (empty()) throw new popFromEmpty;

    const _Ty v = container_[0].lower;

    bubbleDownLower(0);

    return v;
  }

  // remove the highest value element
  _Ty pop_max() {
    if (empty()) throw new popFromEmpty;

    if (size() == 1) {
      const _Ty v = container_[0].lower;

      container_.clear();
      isOdd_ = false;
      return v;
    }
    else {
      const _Ty v = container_[0].upper;

      bubbleDownUpper(0);

      return v;
    }
  }

  // true iff the queue is empty
  bool empty() const {
    return container_.empty();
  }

  inline unsigned int size() const {
    unsigned int size = 2 * container_.size();

    if (isOdd_) return size - 1;
    else return size;
  }

  // true iff the interval heap obeys the inverval heap rules
  bool test() const {
    interval bounds(std::numeric_limits<_Ty>::min(), std::numeric_limits<_Ty>::max());

    return test(bounds, 0);
  }

  // remove an element by index (intended to remove a random element where the random selection has been done externally
  void remove(const unsigned int index) {
    const unsigned int interval = index / 2;
    if (interval >= container_.size()) throw new removeIndexOutOfBounds;
    if (interval == container_.size()-1) {
      // last container - special case

      if (isOdd_) {
        // the very last element, so just remove it

        if (is_odd(index)) throw new removeIndexOutOfBounds;

        isOdd_ = false;
        container_.pop_back();
      }
      else {
        isOdd_ = true; // remove the upper bound
        
        if (!is_odd(index)) {
          // need to remove the lower bound, so move the upper bound into its slot
          container_[interval].lower = container_[interval].upper;
        }
      }
    }
    else {
      // borrow a value from the last interval
      _Ty borrowed;

      if (isOdd_) {
        // last interval has only one value, so delete it
        borrowed = container_.back().lower;
        isOdd_ = false;
        container_.pop_back();
      }
      else {
        // borrow the upper value
        borrowed = container_.back().upper;
        isOdd_ = true;
      }

      if (is_odd(index)) {
        // deleting the upper value
        if (borrowed != container_[interval].upper) {
          // no need to do anything if the borrowed value is the same as the removed value

          if (borrowed > container_[interval].upper) {
            // need to bubble up upper
            bubbleUpUpper(interval, borrowed);
          }
          else if (borrowed >= container_[interval].lower) {
            // bubble the borrowed value downwards
            bubbleDownUpper(interval, borrowed);
          }
          else {
            // need to make the lower value into the new upper and bubble down
            bubbleDownUpper(interval, container_[interval].lower);
            // and bubble up the new lower
            bubbleUpLower(interval, borrowed);
          }
        }
      }
      else {
        // deleting the lower value

        if (borrowed != container_[interval].lower) {
          // no need to do anything if the borrowed value is the same as the removed value
          if (borrowed < container_[interval].lower) {
            // need to bubble up lower
            bubbleUpLower(interval, borrowed);
          }
          else if (borrowed <= container_[interval].upper) {
            // bubble the borrowed value downwards
            bubbleDownLower(interval, borrowed);
          }
          else {
            // need to make the upper value into the new lower and bubble down
            bubbleDownLower(interval, container_[interval].upper);
            // and bubble up the new upper
            bubbleUpUpper(interval, borrowed);
          }
        }
      }
    }
  }

  // replace an element by index (intended to replace a random element where the random selection has been done externally
  void replace(const unsigned int index, const _Ty v) {
    const unsigned int interval = index / 2;
    if (interval >= container_.size()) throw new removeIndexOutOfBounds;
    if (interval == container_.size()-1 && isOdd_) {
      // last container with 1 element - special case
      if (is_odd(index)) throw new removeIndexOutOfBounds;

      if (interval == 0) {
        // replacing the only value
        container_[0].lower = v;
      }
      else {
        const unsigned int p = parent(interval);
        if (v > container_[p].upper) {
          container_[interval].lower = container_[p].upper;
          bubbleUpUpper(p, v);
        }
        else if (v < container_[p].lower) {
          container_[interval].lower = container_[p].lower;
          bubbleUpLower(p, v);
        }
        else {
          container_[interval].lower = v;
        }
      }
    }
    else {
      if (is_odd(index)) {
        // deleting the upper value
        if (v != container_[interval].upper) {
          // no need to do anything if the value is the same as the removed value

          if (v > container_[interval].upper) {
            // need to bubble up upper
            bubbleUpUpper(interval, v);
          }
          else if (v >= container_[interval].lower) {
            // bubble the value downwards
            bubbleDownUpper(interval, v);
          }
          else {
            // need to make v the new lower and bubble up
            const _Ty tmp = container_[interval].lower;
            bubbleUpLower(interval, v);
            // and make the old lower value into the new upper and bubble down
            bubbleDownUpper(interval, tmp);
          }
        }
      }
      else {
        // deleting the lower value

        if (v != container_[interval].lower) {
          // no need to do anything if the value is the same as the removed value
          if (v < container_[interval].lower) {
            // need to bubble up lower
            bubbleUpLower(interval, v);
          }
          else if (v <= container_[interval].upper) {
            // bubble the value downwards
            bubbleDownLower(interval, v);
          }
          else {
            // need to make v the nw upper and bubble up
            const _Ty tmp = container_[interval].upper;
            bubbleUpUpper(interval, v);
            // and make the old upper value into the new lower and bubble down
            bubbleDownLower(interval, tmp);
          }
        }
      }
    }
  }

  // replace the minimum value with the specified value
  inline void replaceMin(const _Ty v) { replace(0, v); }

  // replace the maximum value with the specified value
  inline void replaceMax(const _Ty v) { replace(1, v); }

 
  // print the underlying vector - for debugging - printF is a function for printing one of the base type
  void dump(void (*printF)(_Ty)) const {
    for (int i = 0; i < container_.size(); i++) {
      if (i == container_.size()-1 && isOdd_) {
        printF(container_[i].lower);
      }
      else {
        printF(container_[i].lower);
        printF(container_[i].upper);
      }
    }
  }

private:
  inline bool is_odd(unsigned int i) const { return i & 1; }

  class popFromEmpty : public std::exception {
    virtual const char* what() const throw()
    {
      return "Cannot pop an empty DEPQ";
    }
  };
  class removeIndexOutOfBounds : public std::exception {
    virtual const char* what() const throw()
    {
      return "Index out of bounds for remove";
    }
  };

  struct interval {
    interval() {}
    interval(_Ty l, _Ty u) : lower(l), upper(u) {}
  public:
    _Ty lower; 
    _Ty upper; 
  };

  inline unsigned int leftChild(const int parent) const {
    return (parent << 1) + 1; // 2 * parent + 1
  }

  inline unsigned int rightChild(const int parent) const {
    return (parent << 1) + 2; // 2 * parent + 2
  }

  inline unsigned int parent(const unsigned int node)  const {
    return (node-1) / 2;
  }

  // container_[n].lower is empty.
  // v is to be inserted into the interval or a higher interval
  void bubbleUpLower(unsigned int n, _Ty v) {
    assert (n >= 0);
    
    while (n > 0) {
      const unsigned int p = parent(n);

      if (v < container_[p].lower) {
        container_[n].lower = container_[p].lower;
        n = p;
      }
      else {
        break;
      }
    }

    // insert the value here
    container_[n].lower = v;
  }

  // container_[n].upper is empty.
  // v is to be inserted into the interval or a higher interval
  void bubbleUpUpper(unsigned int n, const _Ty v) {
    assert (n >= 0);

    while (n > 0) {
      const unsigned int p = parent(n);

      if (v > container_[p].upper) {
        container_[n].upper = container_[p].upper;
        n = p;
      }
      else {
        break;
      }
    }

    // insert the value here
    container_[n].upper = v;
  }

  // container_[n].upper is empty.
  // raise a value from below
  void bubbleDownUpper(unsigned int n) {
    while (true) {
      // loop until finished
      const unsigned int lc = leftChild(n);
      if (lc >= container_.size()) {
        // no children
        if (n == container_.size() - 1) {
          // the rightmost node
          assert(!isOdd_); // cannot delete the upper bound from a single value interval!

          // just invalidate the upper bound
          isOdd_ = true;
          return;
        }
        else {
          // no children but not the rightmost node, so take a value from the rightmost node
          _Ty v;  // the value that is taken

          if (isOdd_) {
            v = container_.back().lower;
            container_.pop_back();
            isOdd_ = false;
          }
          else {
            v = container_.back().upper;
            isOdd_ = true;
          }

          // insert the borrowed value into this branch of the heap
          if (v < container_[n].lower) {
            // the upper value becomes the value of lower and the borrowed value becomes the new upper
            // need to bubble this new upper value upwards
            container_[n].upper = container_[n].lower;
            bubbleUpLower(n, v);
          }
          else {
            bubbleUpUpper(n, v);
          }
          return;
        }
      }
      else {
        const unsigned int rc = rightChild(n);
        if (rc >= container_.size()) {
          // only the left child, which must be a leaf

          if (lc == container_.size()-1 && isOdd_) {
            // the child only has a lower bound so take it and delete the child node
            container_[n].upper = container_[lc].lower;
            container_.pop_back();
            isOdd_ = false;
            return;
          }
          else {
            // take the child's upper bound and continue down to borrow the value from the last node
            container_[n].upper = container_[lc].upper;
            n = lc;
          }
        }
        else {
          // two children

          // check whether the right child has only a lower bound
          if (rc == container_.size()-1 && isOdd_) {
            const _Ty rightLower = container_[rc].lower;

            // the heap shrinks by disposing of the right node
            container_.pop_back();
            isOdd_ = false;

            if (container_[lc].upper > rightLower) {
              container_[n].upper = container_[lc].upper; // take the higher value

              // insert the lower value into the left branch
              if (container_[lc].lower > rightLower) {
                // take value from right child and insert it as a lower bound in the left branch
                container_[lc].upper = container_[lc].lower;

                // insert the value
                bubbleUpLower(lc, rightLower);
                return;
              }
              else {
                container_[lc].upper = rightLower;
                return;
              }
            }
            else {
              container_[n].upper = rightLower;
              return;
            }
          }
          else {
            // take the upper child's upper bound and continue down
            if (container_[lc].upper > container_[rc].upper) {
              container_[n].upper = container_[lc].upper;
              n = lc;
            }
            else {
              container_[n].upper = container_[rc].upper;
              n = rc;
            }
          }
        }
      }
    }
  }

  // container_[n].lower is empty.
  // raise a value from below
  void bubbleDownLower(unsigned int n) {
    while (true) {
      // loop until finished
      
      const unsigned int lc = leftChild(n);
      if (lc >= container_.size()) {
        // no children
        if (n == container_.size() - 1) {
          // the rightmost node
          if (isOdd_) {
            // the only value in the node so simply delete it
            container_.pop_back();
            isOdd_ = false;
            return;
          }
          else {
            container_[n].lower = container_[n].upper;
            isOdd_ = true;
            return;
          }
        }
        else {
          // no children but not the rightmost node, so take a value from the rightmost node
          const _Ty v = container_.back().lower;

          // insert the borrowed value into this branch of the heap
          if (v > container_[n].upper) {
            // the upper value becomes the value of lower and the borrowed value becomes the new upper
            // need to bubble this new upper value upwards
            container_[n].lower = container_[n].upper;
            bubbleUpUpper(n, v);
          }
          else {
            bubbleUpLower(n, v);
          }

          // remove the borrowed value from the end of the heap
          if (isOdd_) {
            // there is only one value in the rightnode so simply delete the node
            container_.pop_back();
            isOdd_ = false;
          }
          else {
            container_.back().lower = container_.back().upper;
            isOdd_ = true;
          }

          return;
        }
      }
      else {
        const unsigned int rc = rightChild(n);
        if (rc >= container_.size()) {
          // only the left child

          // take the child's lower bound and continue down
          container_[n].lower = container_[lc].lower;
          n = lc;
        }
        else {
          // two children

          // take the lower child's lower bound and continue down
          if (container_[lc].lower < container_[rc].lower) {
            container_[n].lower = container_[lc].lower;
            n = lc;
          }
          else {
            container_[n].lower = container_[rc].lower;
            n = rc;
          }
        }
      }
    }
  }


  // container_[n].upper has been decreased to v, so bubble down if necessary.
  void bubbleDownUpper(unsigned int n, _Ty v) {
    if (n == container_.size()-1 && isOdd_) {
      // special case, we are inserting at a single value interval - just insert and we are done
      container_[n].lower = v;
      return;
    }

    interval* current;

    while (v > (current = &container_[n])->lower) {
      const unsigned int lc = leftChild(n);

      if (lc >= container_.size()) {
        // no children

        // just insert the value and we are done
        current->upper = v;
        return;
      }
      else {
        const unsigned int rc = rightChild(n);
        if (rc >= container_.size()) {
          // only the left child - insert in the correct place and we are done

          if (lc == container_.size()-1 && isOdd_) {
            // the child only has a lower bound
            // insert in the appropriate place and we are done
            _Ty* const leftChildLower = &container_[lc].lower;

            if (v < *leftChildLower) {
              current->upper = *leftChildLower;
              *leftChildLower = v;
            }
            else {
              current->upper = v;
            }
          }
          else {
            if (v < container_[lc].upper) {
              // take the child's upper bound and check whether to insert as upper or lower
              current->upper = container_[lc].upper;
              if (container_[lc].lower > v) {
                container_[lc].upper = container_[lc].lower;
                container_[lc].lower = v;
              }
              else {
                container_[lc].upper = v;
              }
            }
            else {
              // insert here and we are done
              current->upper = v;
            }
          }

          return;
        }
        else {
          // two children

          const _Ty leftUpper = container_[lc].upper;

          if (rc == container_.size()-1 && isOdd_) {
            // the right child has only a lower bound
            const _Ty rightLower = container_[rc].lower;

            // swap it if it is greater than v
            if (rightLower > v) {
              const _Ty tmp = rightLower;
              container_[rc].lower = v;
              v = tmp;
            }

            // then check the left branch
            if (leftUpper > v) {
              // swap the values - no need to recurse because there can't be any children
              current->upper = leftUpper;
              container_[lc].upper = v;
            }
            else {
              // insert here
              current->upper = v;
            }

            return;
          }
          else {
            // check against the greater child
            const _Ty rightUpper = container_[rc].upper;

            if (leftUpper > rightUpper) {
              if (leftUpper > v) {
                // lift the value and bubble down
                current->upper = leftUpper;
                n = lc;
              }
              else {
                // insert here and we are done
                current->upper = v;
                return;
              }
            }
            else {
              if (rightUpper > v) {
                // lift the value and bubble down
                current->upper = rightUpper;
                n = rc;
              }
              else {
                // insert here and we are done
                current->upper = v;
                return;
              }
            }
          }
        }
      }
    }

    // v is less than the lower bound so rotate the lower bounds down to a root and the upper bounds up to here then insert v as the lower bound here
    assert(v <= current->lower);

    cycleLowerToUpper(v, n);
  }

  // a new value is being inserted into current->lower and current->upper is empty
  // shuffle values of lower down and values of upper up accordingly
  inline void cycleLowerToUpper(_Ty v, unsigned int n) {
    while (true) {
      interval* const current = &container_[n];

      const unsigned int lc = leftChild(n);

      if (lc >= container_.size()) {
        // no children

        // just insert the value and we are done
        current->upper = current->lower;
        current->lower = v;
        return;
      }
      else {
        const unsigned int rc = rightChild(n);
        if (rc >= container_.size()) {
          // only the left child which must be a leaf so cycle through it and we are done

          if (lc == container_.size()-1 && isOdd_) {
            // the child only has a lower bound
            // cycle to the appropriate places and we are done
            _Ty* const leftChildLower = &container_[lc].lower;

            current->upper = *leftChildLower;
            *leftChildLower = current->lower;
            current->lower = v;
          }
          else {
            // cycle
            current->upper = container_[lc].upper;
            container_[lc].upper = container_[lc].lower;
            container_[lc].lower = current->lower;
            current->lower = v;
          }

          return;
        }
        else {
          // two children

          const _Ty leftUpper = container_[lc].upper;

          if (rc == container_.size()-1 && isOdd_) {
            // the right child has only a lower bound and both children are leaves
            const _Ty rightLower = container_[rc].lower;

            // cycle through the node with the greater value for upper and we are done
            if (rightLower >= leftUpper) {
              current->upper = rightLower;
              container_[rc].lower = current->lower;
            }
            else {
              current->upper = leftUpper;
              container_[lc].upper = container_[lc].lower;
              container_[lc].lower = current->lower;
            }

            current->lower = v;

            return;
          }
          else {
            // cycle through the child with the greater upper bound
            const _Ty rightUpper = container_[rc].upper;
            

            if (leftUpper > rightUpper) {
              current->upper = leftUpper;
              n = lc;
            }
            else {
              current->upper = rightUpper;
              n = rc;
            }

            const _Ty tmp = current->lower;
            current->lower = v;
            v = tmp;
          }
        }
      }
    }
  }

  // container_[n].lower has been increased to v, so bubble down if necessary.
  void bubbleDownLower(unsigned int n, _Ty v) {
    if (n == container_.size()-1 && isOdd_) {
      // special case, we are inserting at a single value interval - just insert and we are done
      container_[n].lower = v;
      return;
    }

    interval* current;

    while (v < (current = &container_[n])->upper) { // cannot be a single element node
      const unsigned int lc = leftChild(n);

      if (lc >= container_.size()) {
        // no children

        // just insert the value and we are done
        current->lower = v;
        return;
      }
      else {
        const unsigned int rc = rightChild(n);
        if (rc >= container_.size()) {
          // only the left child - insert in the correct place and we are done

          if (lc == container_.size()-1 && isOdd_) {
            // the child only has a lower bound
            // insert in the appropriate place and we are done
            _Ty* const leftChildLower = &container_[lc].lower;

            if (v > *leftChildLower) {
              current->lower = *leftChildLower;
              *leftChildLower = v;
            }
            else {
              current->lower = v;
            }
          }
          else {
            if (v > container_[lc].lower) {
              // take the child's upper bound and check whether to insert as upper or lower
              current->lower = container_[lc].lower;
              if (container_[lc].upper < v) {
                container_[lc].lower = container_[lc].upper;
                container_[lc].upper = v;
              }
              else {
                container_[lc].lower = v;
              }
            }
            else {
              // insert here and we are done
              current->lower = v;
            }
          }

          return;
        }
        else {
          // two children

          const _Ty leftLower = container_[lc].lower;

          if (rc == container_.size()-1 && isOdd_) {
            // the right child has only a lower bound
            const _Ty rightLower = container_[rc].lower;

            // swap it if it is less than v
            if (rightLower < v) {
              const _Ty tmp = rightLower;
              container_[rc].lower = v;
              v = tmp;
            }

            // then check the left branch
            if (leftLower < v) {
              // swap the values - no need to recurse because there can't be any children
              current->lower = leftLower;
              container_[lc].lower = v;
            }
            else {
              // insert here
              current->lower = v;
            }

            return;
          }
          else {
            // check against the lower child
            const _Ty rightLower = container_[rc].lower;

            if (leftLower < rightLower) {
              if (leftLower < v) {
                // lift the value and bubble down
                current->lower = leftLower;
                n = lc;
              }
              else {
                // insert here and we are done
                current->lower = v;
                return;
              }
            }
            else {
              if (rightLower < v) {
                // lift the value and bubble down
                current->lower = rightLower;
                n = rc;
              }
              else {
                // insert here and we are done
                current->lower = v;
                return;
              }
            }
          }
        }
      }
    }

    // v is greater than the upper bound so rotate the upper bound down to a leaf and the lower bound up to here then insert v as the auuper bound here
    assert(v >= current->upper);

    cycleUpperToLower(v, n);
  }

  // a new value is being inserted into current->lower and current->upper is empty
  // shuffle values of lower down and values of upper up accordingly
  inline void cycleUpperToLower(_Ty v, unsigned int n) {
    while (true) {
      interval* const current = &container_[n];

      const unsigned int lc = leftChild(n);

      if (lc >= container_.size()) {
        // no children

        // just insert the value and we are done
        current->lower = current->upper;
        current->upper = v;
        return;
      }
      else {
        const unsigned int rc = rightChild(n);
        if (rc >= container_.size()) {
          // only the left child which must be a leaf so cycle through it and we are done

          if (lc == container_.size()-1 && isOdd_) {
            // the child only has a lower bound
            // cycle to the appropriate places and we are done
            _Ty* const leftChildLower = &container_[lc].lower;

            current->lower = *leftChildLower;
            *leftChildLower = current->upper;
            current->upper = v;
          }
          else {
            // cycle
            current->lower = container_[lc].lower;
            container_[lc].lower = container_[lc].upper;
            container_[lc].upper = current->upper;
            current->upper = v;
          }

          return;
        }
        else {
          // two children

          const _Ty leftLower = container_[lc].lower;

          if (rc == container_.size()-1 && isOdd_) {
            // the right child has only a lower bound and both children are leaves
            const _Ty rightLower = container_[rc].lower;

            // cycle through the node with the lesser value for lower and we are done
            if (rightLower <= leftLower) {
              current->lower = rightLower;
              container_[rc].lower = current->upper;
            }
            else {
              current->lower = leftLower;
              container_[lc].lower = container_[lc].upper;
              container_[lc].upper = current->upper;
            }

            current->upper = v;

            return;
          }
          else {
            // cycle through the child with the lower lower bound
            const _Ty rightLower = container_[rc].lower;
            

            if (leftLower < rightLower) {
              current->lower = leftLower;
              n = lc;
            }
            else {
              current->lower = rightLower;
              n = rc;
            }

            const _Ty tmp = current->upper;
            current->upper = v;
            v = tmp;
          }
        }
      }
    }
  }

 
  // test whether the DEPQ is well stuctured
  bool test(interval bounds, unsigned int node) const {
    if (node >= container_.size()) return true;
    else if (node == container_.size()-1 &&
             isOdd_ &&
             (comp_(container_[node].lower, bounds.lower) ||
              comp_(bounds.upper, container_[node].lower))) {
                return false;  // special case for last interval containing only one element
    }
    if (comp_(container_[node].lower, bounds.lower)) {
      return false;          // lower must not not be lower than the containing interval
    }
    if (comp_(bounds.upper, container_[node].upper)) {
      return false;         // the containing interval must not not be lower than upper
    }
    if (!test(container_[node], leftChild(node))) {
      return false;              // the left child must be correct
    }
    return test(container_[node], rightChild(node));                         // the right child must be correct
  }

  std::vector<interval> container_; // the container
  bool isOdd_;                      // true if the last interval's value for upper is invalid
	_Pr comp_;	                      // the comparator functor
};
