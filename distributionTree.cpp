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

#include "distributionTree.h"
#include "smoothing.h"
#include "utils.h"
#include "AdaBoost.h"
#include <assert.h>

InstanceStream::MetaData const* dtNode::metaData_;



dtCatNode::dtCatNode() : att_(NOPARENT) {
}

dtCatNode::~dtCatNode() {
}


dtCatNode::dtCatNode(InstanceStream::MetaData const* meta, const CategoricalAttribute a) :xyCount(meta->getNoValues(a),meta->getNoClasses()), yCount(metaData_->getNoClasses(), 0) {
    att_=NOPARENT; 
    metaData_ = meta;
}

//The parameter const CategoricalAttribute a is useless but intentionally added (used in kdb-condDisc)
dtCatNode::dtCatNode(const CategoricalAttribute a, unsigned int noValues) : xyCount(noValues,metaData_->getNoClasses()), yCount(metaData_->getNoClasses(), 0)
#ifdef _kdbCDRAM
, discretised_(false), discUpdated_(false), valuesCount_(0)
#endif
{
    att_=NOPARENT; 
}

dtCatNode::dtCatNode(const CategoricalAttribute a) : xyCount(metaData_->getNoValues(a),metaData_->getNoClasses()), yCount(metaData_->getNoClasses(), 0)
#ifdef _kdbCDRAM
  , discretised_(false), discUpdated_(false), valuesCount_(0)
#endif
{
    att_=NOPARENT; 
}

void dtCatNode::init(InstanceStream::MetaData const* meta, const CategoricalAttribute a) {
  metaData_ = meta;
  att_ = NOPARENT;
  xyCount.assign(meta->getNoValues(a), meta->getNoClasses(), 0);
  yCount.assign(meta->getNoClasses(), 0);
  children_.clear();
#ifdef _kdbCDRAM
  valuesCount_ = 0;
  discretised_ = false;
  discUpdated_ = false;
#endif
}

void dtCatNode::clear() {
  xyCount.clear();
  yCount.clear();
  children_.clear();
  att_ = NOPARENT;
#ifdef _kdbCDRAM
  valuesCount_ = 0;
  discretised_ = false;
  discUpdated_ = false;
#endif
}

dtNumNode::dtNumNode() : att_(NOPARENT) {
}

dtNumNode::~dtNumNode() {
}

dtNumNode::dtNumNode(const bool aux) : att_(NOPARENT){
  xyGaussDist_.init(metaData_->getNoClasses());
}

dtNumNode::dtNumNode(InstanceStream::MetaData const* metaData) : att_(NOPARENT){
  xyGaussDist_.init(metaData->getNoClasses());
}

void dtNumNode::init(InstanceStream::MetaData const* meta) {
  metaData_ = meta;
  att_ = NOPARENT;
  xyGaussDist_.init(metaData_->getNoClasses());
  children_.clear();
}

void dtNumNode::clear() {
  xyGaussDist_.clear();
  children_.clear();
  att_ = NOPARENT;
}

distributionCatTree::distributionCatTree() 
{
}

distributionCatTree::distributionCatTree(InstanceStream::MetaData const* metaData, const CategoricalAttribute att) : dTree(metaData, att)
{
    metaData_ = metaData; 
}

distributionCatTree::~distributionCatTree(void)
{
}

void distributionCatTree::init(InstanceStream const& stream, const CategoricalAttribute att)
{
  metaData_ = stream.getMetaData();
  dTree.init(metaData_, att);
}

void distributionCatTree::clear()
{
  dTree.clear();
}

dtCatNode* distributionCatTree::getdTNode(){
  return &dTree;
}

void distributionCatTree::update(const instance &i, const CategoricalAttribute a, const std::vector<CategoricalAttribute> &parents) {
  const CatValue y = i.getClass();
  const CatValue v = i.getCatVal(a);

  dTree.ref(v, y)++;
  dTree.yCount[y]++;

  dtCatNode *currentNode = &dTree;

  for (unsigned int d = 0; d < parents.size(); d++) { 

    const CategoricalAttribute p = parents[d];

    if (currentNode->att_ == NOPARENT || currentNode->children_.empty()) {
      // children array has not yet been allocated
      currentNode->children_.assign(metaData_->getNoValues(p), NULL);
      currentNode->att_ = p;
    }

    assert(currentNode->att_ == p);
    
    dtCatNode *nextNode = currentNode->children_[i.getCatVal(p)];

    // the child has not yet been allocated, so allocate it
    if (nextNode == NULL) {
      currentNode = currentNode->children_[i.getCatVal(p)] = new dtCatNode(a);
    }
    else {
      currentNode = nextNode;
    }

    currentNode->ref(v, y)++;
    currentNode->yCount[y]++;
  }
}

  void distributionCatTree::update(const instance &i, const CategoricalAttribute a, const std::vector<CategoricalAttribute> &parents, NumValue attValue){
    const CatValue y = i.getClass();
    const CatValue v = i.getCatVal(a);

    dTree.ref(v, y)++;
    dTree.yCount[y]++;
#ifdef _kdbCDRAM
    dTree.valuesCount_++;
#endif

    dtCatNode *currentNode = &dTree;

    int d = 0;
    for (d = 0; d < static_cast<int>(parents.size())-1; d++) { 

      const CategoricalAttribute p = parents[d];

      if (currentNode->att_ == NOPARENT || currentNode->children_.empty()) {
        // children array has not yet been allocated
        currentNode->children_.assign(metaData_->getNoValues(p), NULL);
        currentNode->att_ = p;
      }

      assert(currentNode->att_ == p);

      dtCatNode *nextNode = currentNode->children_[i.getCatVal(p)];

      // the child has not yet been allocated, so allocate it
      if (nextNode == NULL) {
        currentNode = currentNode->children_[i.getCatVal(p)] = new dtCatNode(a);
      }
      else {
        currentNode = nextNode;
#ifdef _kdbCDRAM
        currentNode->valuesCount_++;
#endif
      }

      currentNode->ref(v, y)++;
      currentNode->yCount[y]++;
    }
    if(d < parents.size()){//store numeric values in the last leaf
      const CategoricalAttribute p = parents[d];

      if (currentNode->att_ == NOPARENT || currentNode->children_.empty()) {
        // children array has not yet been allocated
        currentNode->children_.assign(metaData_->getNoValues(p), NULL);
        currentNode->att_ = p;
      }

      assert(currentNode->att_ == p);

      dtCatNode *nextNode = currentNode->children_[i.getCatVal(p)];

      // the child has not yet been allocated, so allocate it
      if (nextNode == NULL) {
        currentNode = currentNode->children_[i.getCatVal(p)] = new dtCatNode(a,0);
#ifdef _kdbCDRAM
        currentNode->numValues_.resize(metaData_->getNoClasses());
        currentNode->valuesCount_ = 0;
        //currentNode->discretised_ = false; (not needed on the leaves)
#endif
      }
      else {
        currentNode = nextNode;
      }
#ifdef _kdbCDRAM
      currentNode->numValues_[y].push_back(attValue);
      currentNode->valuesCount_++;
#endif
    }
  }

  void distributionCatTree::update(std::vector<CatValue> &valsDisc,  
                                const CategoricalAttribute a, std::vector<CatValue> &classes, const std::vector<CategoricalAttribute>  &parents, 
                                std::vector<CatValue> &parentValues, std::vector<NumValue> &cuts, unsigned int noOrigCatAtts){
      
      dtCatNode *currentNode = &dTree;
      dtCatNode *nextNode;
      
      unsigned int d = 0;
      for (d = 0; d < parents.size()-1; d++) { 
        assert(currentNode->att_ == parents[d]);
        nextNode = currentNode->children_[parentValues[d]];
        currentNode = nextNode;
      }
      //for the last parent
      if(parents.size()>0){
        
        const CategoricalAttribute p = parents[d];

        //Useful only for kdb-condDisc
        if (currentNode->att_ == NOPARENT || currentNode->children_.empty()) {
          // children array has not yet been allocated
          currentNode->children_.assign(metaData_->getNoValues(p), NULL);
          currentNode->att_ = p;
        }

        assert(currentNode->att_ == p);

        nextNode = currentNode->children_[parentValues[d]];
        
        // the child has not yet been allocated, so allocate it
        if ((nextNode == NULL)) {//This for kdb-condDisc 
          currentNode = currentNode->children_[parentValues[d]] = new dtCatNode(a, cuts.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts));//this can be zero except for the last one
        }else if(nextNode->xyCount.getDim() == 0){//This for kdb-condDisc2
          nextNode->xyCount.resize(cuts.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts),metaData_->getNoClasses());
          nextNode->yCount.resize(metaData_->getNoClasses());
          currentNode = nextNode;
        }else {
          currentNode = nextNode;
        }
        //only update counts on the leaves
        int i=0;
        for (std::vector<CategoricalAttribute>::const_iterator it = valsDisc.begin(); it != valsDisc.end(); it++, i++){
          currentNode->ref(*it, classes[i])++;
          currentNode->yCount[classes[i]]++;
        }
      }
      //only store cuts on the leaves
      currentNode->cuts_ = cuts; //Only in the leaves. 
#ifdef _kdbCDRAM
      currentNode->discretised_ = true;
      currentNode->discUpdated_ = true;
#endif
  }

// update classDist using the evidence from the tree about i
void distributionCatTree::updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att_;
  }

  const unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
  }
}

// update classDist using the evidence from the tree about i
void distributionCatTree::updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, unsigned int k) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;

  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT) && (depth<k) ) { //We want to consider kdb k=k, we stop when the depth reached is equal to k
    depth++;
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att_;
  }

  const unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
  }
}

// update classDist using the evidence from the tree about i
// require that at least minCount values be used for proability estimation
void distributionCatTree::updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;
  const CatValue parentVal = i.getCatVal(a);
  const CatValue noOfClasses = metaData_->getNoClasses();

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < noOfClasses; y++) {
      cnt += next->getCount(parentVal, y);
      if (cnt >= minCount) goto next;
    }

    // break if the total count is < minCOunt
    break;

next:
    dt = next;
    att = dt->att_;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  const unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < noOfClasses; y++) {
    classDist[y] *= mEstimate(dt->getCount(parentVal, y), dt->yCount[y], noOfVals);
  }
}


void distributionCatTree::updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount, unsigned int k) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;
  const CatValue parentVal = i.getCatVal(a);
  const CatValue noOfClasses = metaData_->getNoClasses();

  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT) && (depth<k) ) { //We want to consider kdb k=k, we stop when the depth reached is equal to k
    depth++;
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < noOfClasses; y++) {
      cnt += next->getCount(parentVal, y);
      if (cnt >= minCount) goto next;
    }

    // break if the total count is < minCOunt
    break;

next:
    dt = next;
    att = dt->att_;
  }

  // sum over all values of the Attribute for the class to obtain count[y, parents]
  const unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < noOfClasses; y++) {
    classDist[y] *= mEstimate(dt->getCount(parentVal, y), dt->yCount[y], noOfVals);
  }
}



// update classDist using the evidence from the tree about i and deducting it at the same time (Pazzani's trick for loocv)
// require that at least 1 value (minCount = 1) be used for probability estimation
void distributionCatTree::updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if (next == NULL) break;

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses() && cnt < 2; y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2) 
        break;

    dt = next;
    att = dt->att_;
  }

  unsigned int noOfVals = metaData_->getNoValues(a);
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    if(y!=i.getClass())
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
    else
        classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, dt->yCount[y]-1, noOfVals);
  }
}

void distributionCatTree::updateClassDistributionloocv(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_){
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;
  const unsigned int noOfVals = metaData_->getNoValues(a);
  const CatValue targetV = i.getCatVal(a);
  
  // find the appropriate leaf
  unsigned int depth = 0;
  while ( (att != NOPARENT)) { //We want to consider kdb k=k
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      if(y!=i.getClass())
          classDist[depth][y] *= mEstimate(dt->getCount(targetV, y), dt->yCount[y], noOfVals);
      else
          classDist[depth][y] *= mEstimate(dt->getCount(targetV, y)-1, dt->yCount[y]-1, noOfVals);
    }

    dtCatNode * const next = dt->children_[i.getCatVal(att)];
    if (next == NULL) {
      break;
    }

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses() && cnt < 2; y++) {
      cnt += next->getCount(targetV, y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2){ 
      depth++;
      break;
    }

    dt = next;
    att = dt->att_; 
    depth++;
  }

  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    double mEst;
    if(y!=i.getClass())
      mEst = mEstimate(dt->getCount(targetV, y), dt->yCount[y], noOfVals);
    else
      mEst = mEstimate(dt->getCount(targetV, y)-1, dt->yCount[y]-1, noOfVals);

    for(int k=depth; k<=k_; k++){
      classDist[k][y] *= mEst;
    }
  }
}

void distributionCatTree::updateClassDistributionloocvWithNB(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_){
  const unsigned int noOfVals = metaData_->getNoValues(a);
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;
  
  // update the class distribution at each k
  unsigned int depth = 0;
  while ( (att != NOPARENT)) { //We want to consider kdb k=k
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      if(y!=i.getClass())
        classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
      else
        classDist[depth][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, dt->yCount[y]-1, noOfVals);
    }

    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if (next == NULL) {
      // No nodes down this path so apply the current node to all further classDists
      if (depth < k_) {
        for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
          double mEst;
          if(y!=i.getClass())
            mEst = mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
          else
            mEst = mEstimate(dt->getCount(i.getCatVal(a), y)-1, dt->yCount[y]-1, noOfVals);

          for(int k=depth+1; k<=k_; k++){
            classDist[k][y] *= mEst;
          }
        }
      }
      return;
    };

    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
    for (CatValue y = 0; y < metaData_->getNoClasses() && cnt < 2; y++) {
      cnt += next->getCount(i.getCatVal(a), y);
    }

    //In loocv, we consider minCount=1(+1), since we have to leave out i.
    if (cnt < 2){ 
      // Only the loo example down this path so apply the current node to all further classDists
      for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
        double mEst;
        if(y!=i.getClass())
           mEst = mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
        else
          mEst = mEstimate(dt->getCount(i.getCatVal(a), y)-1, dt->yCount[y]-1, noOfVals);

        for(int k=depth+1; k<=k_; k++){
          classDist[k][y] *= mEst;
        }
      }

      return;
    }

    dt = next;
    att = dt->att_; 
    depth++;
  }

  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    // no further nodes so apply the current node at all further levels
    double mEst;
    if(y!=i.getClass())
      mEst = mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
    else
      mEst = mEstimate(dt->getCount(i.getCatVal(a), y)-1, dt->yCount[y]-1, noOfVals);

    for(int k=depth; k<=k_; k++){
      classDist[k][y] *= mEst;
    }
  }
}


// Updates the class distribution for each of a specified list of different degrees of freedom using leave-one-out cross validation
void distributionCatTree::updateClassDistributionByDOFloocv(std::vector<std::vector<double> > &classDist, std::vector<unsigned int> &dofs, const CategoricalAttribute a, const instance &i){
  const CatValue childVal = i.getCatVal(a);
  const CatValue thisClass = i.getClass();
  const unsigned int noOfVals = metaData_->getNoValues(a);
  dtCatNode *dt = &dTree;
  CategoricalAttribute nextParentAtt = dTree.att_;
  CatValue nextParentVal;
  dtCatNode *next = &dTree;
  unsigned int dof = (noOfVals-1) * metaData_->getNoClasses(); // the degrees of freedom of the current node
  unsigned int dofI = 0;  // index into dofs
  unsigned int nextDOF = 0;
  std::vector<double> classUpdates(metaData_->getNoClasses());

  while (dofI < dofs.size()) {
    if (nextDOF <= dofs[dofI]) {
      // descend to the next node in the tree

      dt = next;
      nextParentAtt = dt->att_;

      if (nextParentAtt == NOPARENT) {
        nextDOF = std::numeric_limits<unsigned int>::max();
        next = NULL;
      }
      else {
        nextParentVal = i.getCatVal(nextParentAtt);

        // check that the next node has enough examples for this value;
        InstanceCount cnt = 0;
        for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
          cnt += next->getCount(childVal, y);
        }

        //In loocv, we consider minCount=1(+1), since we have to leave out i.
        if (cnt < 2){ 
          nextDOF = std::numeric_limits<unsigned int>::max();
          next = NULL;
        }
        else {
          nextDOF = dof * metaData_->getNoValues(nextParentAtt);
          next = dt->children_[nextParentVal];

          for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
            if(y!=i.getClass())
              classDist[dofI][y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
            else
              classDist[dofI][y] *= mEstimate(dt->getCount(i.getCatVal(a), y)-1, dt->yCount[y]-1, noOfVals);
          }
        }
      }
    }

    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      classDist[dofI][y] *= classUpdates[y];
    }

    dofI++;
  }
}

// update classDist using the evidence from the tree about i
void distributionCatTree::updateClassDistributionAndDiscAttValue(std::vector<double> &classDist, const CategoricalAttribute a, 
                                                              const instance &i, NumValue attValue, unsigned int noOrigCatAtts) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;
  

  bool smoothing = false;
  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if ((next == NULL) || (next->xyCount.getDim() == 0)){ //The second part required for kdb-condDisc2
      smoothing = true;
      break;
    }
    dt = next;
    att = dt->att_;
  }
  
  if(smoothing
#ifdef _kdbCDRAM
&& !dt->discUpdated_
#endif
     ){//Instead of P(x_i | x_p1, x_p2 ], x_p3, y) we use P(x_i | x_p1, x_p2, y) with the original discretisation (if it was NULL)
    const unsigned int noOfVals = metaData_->getNoValues(a);
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
    }
  }else{  //Normal cond disc (there is at least one num. sample or smoothing with cond. disc. in the upper parent.
    // sum over all values of the Attribute for the class to obtain count[y, parents]
    const unsigned int noOfVals = dt->cuts_.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts);
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      classDist[y] *= mEstimate(dt->getCount(discretise(attValue,dt->cuts_), y), dt->yCount[y], noOfVals);
    }
  }
}

// update classDist using the evidence from the tree about i
void distributionCatTree::updateClassDistributionAndDiscAttValue(std::vector<double> &classDist, const CategoricalAttribute a, 
                                                              const instance &i, NumValue attValue, unsigned int noOrigCatAtts, InstanceCount minCount) {
  dtCatNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;
  const CatValue noOfClasses = metaData_->getNoClasses();
  CatValue aVal = i.getCatVal(a);

  bool smoothing = false;
  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtCatNode *next = dt->children_[v];
    if ((next == NULL) || (next->xyCount.getDim() == 0)){ //The second part required for kdb-condDisc2 (not sure...)
      smoothing = true;
      break;
    }
    // check that the next node has enough examples for this value;
    InstanceCount cnt = 0;
#ifdef _kdbCDRAM
    if(next->discUpdated_)
       aVal =  discretise(attValue,next->cuts_);
#endif
    for (CatValue y = 0; y < noOfClasses; y++) {
      cnt += next->getCount(aVal, y);
      if (cnt >= minCount) goto next;
    }
    // break if the total count is < minCOunt
    smoothing = true;
    break;

next:
    dt = next;
    att = dt->att_;
  }
  
  if(smoothing
#ifdef _kdbCDRAM
&& !dt->discUpdated_
#endif
     ){//Instead of P(x_i | x_p1, x_p2 , x_p3, y) we use P(x_i | x_p1, x_p2, y), thus, we have to resort to the original discretization
    const unsigned int noOfVals = metaData_->getNoValues(a);
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      classDist[y] *= mEstimate(dt->getCount(i.getCatVal(a), y), dt->yCount[y], noOfVals);
    }
  }else{ //if smoothing==T but there is a discretization on this node, then P(x_i | x_p1, x_p2, y) is used (since dt is a pointer to that) with the tailored discretization. (Only can happen if options minCountDisc_ or levelOfDisc_ has been specified)
         //if smoothing==F then it is a leaf node, it sure has a tailored discretization.
    const unsigned int noOfVals = dt->cuts_.size() + 1 + metaData_->hasNumMissing(a - noOrigCatAtts);
    for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
      classDist[y] *= mEstimate(dt->getCount(discretise(attValue,dt->cuts_), y), dt->yCount[y], noOfVals);
    }
  }
}

void dtCatNode::updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned int depth, unsigned long long int &pc, double &apd, unsigned long long int &zc) {
  if (depth == parents.size()  || children_.empty()) {
    for (CatValue v = 0; v < metaData_->getNoValues(target); v++) {
      pc++;

      apd += (depth-apd) / static_cast<double>(pc);

      for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
        if (getCount(v, y) == 0) zc++;
      }
    }
  }
  else {
    for (CatValue v = 0; v < metaData_->getNoValues(parents[depth]); v++) {
      if (children_[v] == NULL) {
          unsigned long int pathsMissing = 1;
          
          for (unsigned int i = depth; i < parents.size(); i++) pathsMissing *= metaData_->getNoValues(parents[i]);

          pc += pathsMissing;

          apd += pathsMissing * ((depth-apd)/(pc-pathsMissing/2.0));

          for (CatValue tv = 0; tv < metaData_->getNoValues(target); tv++) {
            for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
              if (getCount(tv, y) == 0) zc++;
            }
          }
      }
      else {
        children_[v]->updateStats(target, parents, k, depth+1, pc, apd, zc);
      }
    }
  }
}


void distributionCatTree::updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned long long int &pc, double &apd, unsigned long long int &zc) {
  //dTree.updateStats(target, parents, parents.size(), 1, pc, apd, zc);
}


// prune the distribution tree to the specified number of degrees of freedom
void distributionCatTree::pruneToDOF(const CategoricalAttribute a, unsigned int dof) {
  dTree.pruneToDOF(a, metaData_, dof, (metaData_->getNoValues(a)-1) * metaData_->getNoClasses());
}

// prune the distribution tree to the specified number of degrees of freedom
void dtCatNode::pruneToDOF(const CategoricalAttribute a, InstanceStream::MetaData const* metaData, unsigned int dof, unsigned int dofSoFar) {
  if (att_ != NOPARENT) {
    // check there are children to prune

    const unsigned int nextDOF = dofSoFar * metaData->getNoValues(att_);

    if (nextDOF > dof) {
      att_ = NOPARENT;
      for (ptrVec<dtCatNode>::iterator child = children_.begin(); child != children_.end(); child++) {
        delete *child;
        *child = NULL;
      }
    }
    else {
      for (ptrVec<dtCatNode>::iterator child = children_.begin(); child != children_.end(); child++) {
        pruneToDOF(a, metaData, dof, nextDOF);
      }
    }
  }
}

distributionNumTree::distributionNumTree(){
    
}
distributionNumTree::distributionNumTree(InstanceStream::MetaData const* metaData) : dTree(metaData)
{
    metaData_ = metaData;
}

distributionNumTree::~distributionNumTree(void)
{
}

void distributionNumTree::init(InstanceStream const& stream)
{
  metaData_ = stream.getMetaData();
  dTree.init(metaData_);
}

void distributionNumTree::clear()
{
  dTree.clear();
}

void distributionNumTree::update(const instance &i, const std::vector<NumericAttribute> &parents, NumValue numValue){
  const CatValue y = i.getClass();
  //const CatValue v = i.getCatVal(a);

  //dTree.ref(v, y)++;
  dTree.update(numValue,y);

  dtNumNode *currentNode = &dTree;

  for (unsigned int d = 0; d < parents.size(); d++) { 

    const CategoricalAttribute p = parents[d];

    if (currentNode->att_ == NOPARENT || currentNode->children_.empty()) {
      // children array has not yet been allocated
      currentNode->children_.assign(metaData_->getNoValues(p), NULL);
      currentNode->att_ = p;
    }

    assert(currentNode->att_ == p);
    
    dtNumNode *nextNode = currentNode->children_[i.getCatVal(p)];

    // the child has not yet been allocated, so allocate it
    if (nextNode == NULL) {
      currentNode = currentNode->children_[i.getCatVal(p)] = new dtNumNode(true);
    }
    else {
      currentNode = nextNode;
    }

    //currentNode->ref(v, y)++;
    currentNode->update(numValue,y);
  } 
}

void distributionNumTree::updateClassDistribution(std::vector<double> &classDist, const instance &i, NumValue numValue){
  dtNumNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNumNode *next = dt->children_[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att_;
  }

  double temp, max = 0;
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    //This is to remove zeros
    temp = std::max(1e-75,dt->p(numValue,y));
    classDist[y] *= temp;
    if (classDist[y] > max) {
      max = classDist[y];
    }
  } 
  if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
    for (int j = 0; j < metaData_->getNoClasses(); j++) {
      classDist[j] *= 1e75;
    }
  }
}

void distributionNumTree::updateClassDistribution(std::vector<double> &classDist, const instance &i){
  dtNumNode *dt = &dTree;
  CategoricalAttribute att = dTree.att_;

  // find the appropriate leaf
  while (att != NOPARENT) {
    const CatValue v = i.getCatVal(att);
    dtNumNode *next = dt->children_[v];
    if (next == NULL)
      break;
    dt = next;
    att = dt->att_;
  }

  double temp, max = 0;
  for (CatValue y = 0; y < metaData_->getNoClasses(); y++) {
    //This is to remove zeros
    temp = std::max(1e-75,dt->p(y)); //gets the probability for the mean
    classDist[y] *= temp;
    if (classDist[y] > max) {
      max = classDist[y];
    }
  } 
  if ((max > 0) && (max < 1e-75)) { // Danger of probability underflow
    for (int j = 0; j < metaData_->getNoClasses(); j++) {
      classDist[j] *= 1e75;
    }
  }
}
