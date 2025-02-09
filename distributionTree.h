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
#pragma once
#include "instanceStream.h"
#include "utils.h"
#include "xyGaussDist.h"

const NumericAttribute NOPARENT = std::numeric_limits<NumericAttribute>::max();  // used because some compilers won't accept std::numeric_limits<NumericAttribute>::max() here

class dtNode {
public:
  virtual ~dtNode() {}
  virtual void init(InstanceStream::MetaData const* meta, const unsigned int a){};  // initialise a new uninitialised node
  virtual void clear(){};                          // reset a node to be empty

protected:
  static InstanceStream::MetaData const* metaData_; // save just one metadata pointer for the whole tree
};

class dtCatNode : public dtNode {
public:
  dtCatNode();
  virtual ~dtCatNode();
  dtCatNode(const CategoricalAttribute a);
  dtCatNode(const CategoricalAttribute a, unsigned int noValues); //For kdbCDdisc
  dtCatNode(InstanceStream::MetaData const* meta, const CategoricalAttribute a);
  void init(InstanceStream::MetaData const* meta, const CategoricalAttribute a); 
  void clear();
  
  // returns a reference to the X=v,Y=y counts for value v
  inline InstanceCount &ref(const CatValue v, const CatValue y) {
    return xyCount.ref(v, y);
  }

  // returns the count X=v,Y=y
  inline InstanceCount getCount(const CatValue v, const CatValue y) {
    return xyCount.ref(v, y);
  }

  void pruneToDOF(const CategoricalAttribute a, InstanceStream::MetaData const* metaData, unsigned int dof, unsigned int dofSoFar);    //< prune the distribution tree to the specified number of degrees of freedom

  void updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned int depthRemaining, unsigned long long int &pc, double &apd, unsigned long long int &zc);

  ptrVec<dtCatNode> children_;
  CategoricalAttribute att_;        // the Attribute whose values select the next child
  fdarray<InstanceCount> xyCount;  // joint count indexed by x val the y val
  std::vector<InstanceCount> yCount;  // the marginals - stored to save repeated recalculation
  std::vector<NumValue> cuts_;     // stores the cuts for the numeric attributes (only for kdbCDdisc and kdbCDRAM)
#ifdef _kdbCDRAM
  bool discretised_;               //indicates whether there is a node tailored discretisation for this node in cuts (required by kdbCDRAM because mdl discretisation can create zero cutpoints)
  bool discUpdated_;               //indicates whether the values for this node have been discretised with the "node-tailored" mdl discretisation
  std::vector<std::vector<NumValue> > numValues_;     // stores the numeric values for the different classes to be conditional discretised (only for kdbCDRAM)
  InstanceCount valuesCount_; //number of values  at each node (required for kdbCDRAM -d option, smoothing)
#endif
};

class dtNumNode : public dtNode {
public:
  dtNumNode();
  virtual ~dtNumNode();
  
  dtNumNode(const bool aux);
  dtNumNode(InstanceStream::MetaData const* metaData);
    
  void init(InstanceStream::MetaData const* meta);
  void clear();
  
  inline void update(const NumValue v, const CatValue y){
      xyGaussDist_.update(v,y);
  }
  
  // returns the prob. X=v|Y=y
  inline double p(const NumValue v, const CatValue y) {
    return xyGaussDist_.p(v,y);
  }
  
  // returns the prob. X=mu|Y=y
  inline double p(const CatValue y) {
    return xyGaussDist_.p(y);
  }
  
  
  ptrVec<dtNumNode> children_;
  CategoricalAttribute att_;        // the Attribute whose values select the next child
  xyGaussDist xyGaussDist_;         // p(x|y)
  //double precision_;
};


class distributionTree {
public:
  virtual ~distributionTree() {}
  virtual void clear(){};                            // reset a tree to be empty

protected:
   InstanceStream::MetaData const* metaData_;
};


class distributionCatTree : public distributionTree
{
public:
  distributionCatTree();   // default constructor - init must be called after construction
  distributionCatTree(InstanceStream::MetaData const* metaData, const CategoricalAttribute att);
  virtual ~distributionCatTree(void);

  void init(InstanceStream const& stream, const CategoricalAttribute att);
  void clear();                            // reset a tree to be empty

  void update(const instance &i, const CategoricalAttribute att, const std::vector<CategoricalAttribute> &parents);
  
  // update the counts as usual except for the leaf, where the numeric value is stored (needed in kdbCDRAM)
  void update(const instance &i, const CategoricalAttribute att, const std::vector<CategoricalAttribute> &parents, 
              NumValue attValue);
  
  // update a group of values for a combination of parents at a time (needed in kdbCDdisc) 
  void update(std::vector<CatValue> &valsDisc, const CategoricalAttribute att, std::vector<CatValue> &classes, 
              const std::vector<CategoricalAttribute>  &parents, std::vector<CatValue> &parentValues, 
              std::vector<NumValue> &cuts, unsigned int noOrigCatAtts);

  // update classDist using the evidence from the tree about i
  void updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i); 
  // update classDist using the evidence from the tree about i for kdb k=k (required for kdb selectiveK)
    void updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, unsigned int k); 
  void updateClassDistribution(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount);
  // update classDist using the evidence from the tree about i for kdb k=k and minCount (required for kdb selectiveK)
    void updateClassDistributionForK(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i, InstanceCount minCount, unsigned int k);
  //This method discounts i (Pazzani's trick for loocv)
  void updateClassDistributionloocv(std::vector<double> &classDist, const CategoricalAttribute a, const instance &i);  
  //This method discounts i (Pazzani's trick for loocv) using kdb k=k (the specified k value)
  void updateClassDistributionloocv(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_);  
  void updateClassDistributionloocvWithNB(std::vector<std::vector<double> > &classDist, const CategoricalAttribute a, const instance &i, unsigned int k_);
  // Updates the class distribution for each of a specified list of different degrees of freedom using leave-one-out cross validation
  void updateClassDistributionByDOFloocv(std::vector<std::vector<double> > &classDist, std::vector<unsigned int> &dofs, const CategoricalAttribute a, const instance &i);
  //This method updates classDist by using the discretised value of the attribute a, that is v, conditioned on its parents (kdbCDdisc methods)
  void updateClassDistributionAndDiscAttValue(std::vector<double> &classDist, const CategoricalAttribute a, 
                                              const instance &i, NumValue attValue, unsigned int noOrigCatAtts);
  void updateClassDistributionAndDiscAttValue(std::vector<double> &classDist, const CategoricalAttribute a, 
                                              const instance &i, NumValue attValue, unsigned int noOrigCatAtts, InstanceCount minCount);

  // get statistics on the kdb structure.
  // pc = the number of paths defined by the parent structure = sum over atts of product over parents of number of values for parent 
  // apd = the average depth to which those paths are instantiated
  // zc = the number of counts in leaf nodes that are zero
  void updateStats(CategoricalAttribute target, std::vector<CategoricalAttribute> &parents, unsigned int k, unsigned long long int &pc, double &apd, unsigned long long int &zc);

  void pruneToDOF(const CategoricalAttribute a, unsigned int dof);    //< prune the distribution tree to the specified number of degrees of freedom

  dtCatNode* getdTNode();
private:
  dtCatNode dTree;
};


class distributionNumTree : public distributionTree
{
public:
  distributionNumTree();   // default constructor - init must be called after construction
  distributionNumTree(InstanceStream::MetaData const* metaData);
  virtual ~distributionNumTree(void);

  void init(InstanceStream const& stream);
  void clear();                            // reset a tree to be empty

  void update(const instance &i, const std::vector<NumericAttribute> &parents, NumValue attValue); 
  
  void updateClassDistribution(std::vector<double> &classDist, const instance &i, NumValue numValue); 
  void updateClassDistribution(std::vector<double> &classDist, const instance &i); //considers numValue as the mean of the cond. Gaussian distribution on the node

private:
  dtNumNode dTree;
};
