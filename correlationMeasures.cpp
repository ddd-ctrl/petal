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

#include <math.h>
#include <assert.h>


#include "ALGLIB_specialfunctions.h"
#include "utils.h"
#include "correlationMeasures.h"

double getEntropy(std::vector<InstanceCount>& classDist) {
  double h = 0.0;

  double s = sum(classDist);

  if (s == 0.0) return 0.0;

  for (CatValue y = 0; y < classDist.size(); y++) {
    const double py = classDist[y]/s;

    if (py > 0.0) {
      h -= py * log2(py);
    }
  }

  return h;
}

double getInfoGain(xyDist& dist, CategoricalAttribute a) {
  const double s = sum(dist.classCounts);

  if (s == 0.0) return 0.0;

  double g = getEntropy(dist.classCounts);

  for (CatValue v = 0; v < dist.getNoValues(a); v++) {
    const double cnt = dist.getCount(a, v);
    if (cnt) {
      double ch = 0.0;  // H(y | a)

      for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        const double cp = dist.getCount(a, v, y) / cnt;

        if (cp > 0.0) {
          ch -= cp * log2(cp);
        }
      }

      g -= (cnt/s) * ch;
    }
  }

  return g;
}

double getInfoGain(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right) {
  std::vector<InstanceCount> jointDistribution;

  for (CatValue y = 0; y < left.size(); y++) {
    jointDistribution.push_back(left[y]+right[y]);
  }

  const double s = sum(jointDistribution);

  if (s == 0.0) return 0.0;

  double g = getEntropy(jointDistribution);

  const double cntl = sum(left);

  assert(cntl > 0);

  g -= (cntl/s) * getEntropy(left);

  const double cntr = sum(right);

  if (cntr > 0.0) {
    g -= (cntr/s) * getEntropy(right);
  }

  return g;
}

double getInfoGain(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right, std::vector<InstanceCount> &unknown) {
  std::vector<InstanceCount> jointDistribution;

  for (CatValue y = 0; y < left.size(); y++) {
    jointDistribution.push_back(left[y]+right[y]+unknown[y]);
  }

  const double s = sum(jointDistribution);

  if (s == 0.0) return 0.0;

  double g = getEntropy(jointDistribution);

  const double cntl = sum(left);

  assert(cntl > 0);

  g -= (cntl/s) * getEntropy(left);

  const double cntr = sum(right);

  if (cntr > 0.0) {
    g -= (cntr/s) * getEntropy(right);
  }

  const double cntu = sum(unknown);

  if (cntu > 0.0) {
    g -= (cntu/s) * getEntropy(unknown);
  }

  return g;
}

double getInformation(xyDist& dist, CategoricalAttribute a) {
  double i = 0.0;
  const double s = sum(dist.classCounts);

  if (s == 0.0) return 0.0;

  for (CatValue v = 0; v < dist.getNoValues(a); v++) {
    const double p = dist.getCount(a, v)/s;

    if (p > 0.0) {
      i -= p * log2(p);
    }
  }

  return i;
}

double getGainRatio(xyDist& dist, CategoricalAttribute a) {
  const double iv = getInformation(dist, a);

  if (iv == 0.0) return 0.0;
  else return getInfoGain(dist, a) / iv;
}

/**
 * Calculates the gain ratio for the split class distributions
 *
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right) {
  std::vector<InstanceCount> splitDistribution;

  splitDistribution.push_back(sum(left));
  splitDistribution.push_back(sum(right));

  const double iv = getEntropy(splitDistribution);

  if (iv == 0.0) return 0.0;
  else return getInfoGain(left, right) / iv;

}

/**
 * Calculates the gain ratio for the split class distributions
 *
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 * @param unknown the distribution for the unknown branch
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right, std::vector<InstanceCount> &unknown) {
  std::vector<InstanceCount> splitDistribution;

  splitDistribution.push_back(sum(left));
  splitDistribution.push_back(sum(right));
  splitDistribution.push_back(sum(unknown));

  const double iv = getEntropy(splitDistribution);

  if (iv == 0.0) return 0.0;
  else return getInfoGain(left, right, unknown) / iv;

}

/**
 *
 *             __
 *             \               P(x,y)
 *  MI(X,Y)=   /_  P(x,y)log------------
 *            x,y             P(x)P(y)
 *
 *
 */
void getMutualInformation(xyDist &dist, std::vector<float> &mi)
{
  mi.assign(dist.getNoCatAtts(), 0.0);

  const double totalCount = dist.count;

  for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
    double m = 0.0;

    for (CatValue v = 0; v < dist.getNoValues(a); v++) {
      for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        const InstanceCount avyCount = dist.getCount(a,v,y);
        if (avyCount) {
          m += (avyCount / totalCount) * log2(avyCount/((dist.getCount(a, v)/totalCount)
                                       * dist.getClassCount(y)));
        }
      }
    }

    mi[a] = m;
  }
}


/**
 *
 *             __
 *             \               P(x,y)
 *  MI(X,Y)=   /_  P(x,y)log------------
 *            x,y             P(x)P(y)
 *
 *
 */
void getMutualInformation(xyDist &dist, std::vector<double> &mi)
{
  mi.assign(dist.getNoCatAtts(), 0.0);

  const double totalCount = dist.count;

  for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
    double m = 0.0;

    for (CatValue v = 0; v < dist.getNoValues(a); v++) {
      for (CatValue y = 0; y < dist.getNoClasses(); y++) {
        const InstanceCount avyCount = dist.getCount(a,v,y);
        if (avyCount) {
          m += (avyCount / totalCount) * log2(avyCount/((dist.getCount(a, v)/totalCount)
                                       * dist.getClassCount(y)));
        }
      }
    }

    mi[a] = m;
  }
}



void getSymmetricalUncert(xyDist &dist, std::vector<float> &su) {

  su.assign(dist.getNoCatAtts(), 0.0);

  const double totalCount = dist.count;

  for (CategoricalAttribute a = 0; a < dist.getNoCatAtts(); a++) {
		double m = 0.0;
		double xEnt = 0.0;
		double yEnt = 0.0;

		for (CatValue y = 0; y < dist.getNoClasses(); y++) {
			double pvy = dist.getClassCount(y) / totalCount;
			if (pvy)
				yEnt += pvy * log2(pvy);
		}

		for (CatValue v = 0; v < dist.getNoValues(a); v++) {

			double pvx = dist.getCount(a, v) / totalCount;
			if (pvx)
				xEnt += pvx * log2(pvx);

			for (CatValue y = 0; y < dist.getNoClasses(); y++) {
				const InstanceCount avyCount = dist.getCount(a, v, y);

				if (avyCount) {
					m += (avyCount / totalCount)
							* log2(avyCount / (pvx * dist.getClassCount(y)));
				}
			}
		}

		su[a] = 2 * (m / (-xEnt - yEnt));
	}
}



/*
 *                 __
 *                 \                    P(x1,x2|y)
 * CMI(X1,X2|Y)= = /_   P(x1,x2,y) log-------------
 *               x1,x2,y              P(x1|y)P(x2|y)
 *
 */
void getCondMutualInf(xxyDist &dist, crosstab<float> &cmi)
{
  const double totalCount = dist.xyCounts.count;

  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float m = 0.0;
        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
                m += (x1x2y/totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                     (static_cast<double>(dist.xyCounts.getCount(x1, v1, y)) *
                      dist.xyCounts.getCount(x2, v2, y)));
              }
            }
          }
        }

        assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

      cmi[x1][x2] = m;
      cmi[x2][x1] = m;
    }
  }
}


/*
 *               __
 *               \                  P(x1,x2)
 * AMI(X1,X2)= = /_   P(x1,x2) log-------------
 *               x1,x2              P(x1)P(x2)
 *
 */
void getAttMutualInf(xxyDist &dist, crosstab<float> &ami)
{
  const double totalCount = dist.xyCounts.count;

  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float m = 0.0;
        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {

			  double x1x2=dist.getCount(x1, v1, x2, v2);

              if (x1x2) {
                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
                m += (x1x2/totalCount) *
					log2( x1x2 /
                     (static_cast<double>(dist.xyCounts.getCount(x1, v1))/totalCount
					 * dist.xyCounts.getCount(x2, v2)));
              }
            }
          }
      assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

      ami[x1][x2] = m;
      ami[x2][x1] = m;
    }
  }
}
void getErrorDiff(xxyDist &dist, crosstab<float> &cm)
{
  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float m = 0.0;
        std::vector<double> classDistIndep(dist.getNoClasses());
        std::vector<double> classDist(dist.getNoClasses());

        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                classDistIndep[y] = dist.xyCounts.p(y) * dist.xyCounts.p(x1, v1, y)
                                                       * dist.xyCounts.p(x2, v2, y);
                classDist[y] = dist.xyCounts.p(y) * dist.xyCounts.p(x1, v1, y)
                                                  * dist.p(x2, v2, x1, v1, y);
              }
              else {
                classDistIndep[y] = 0.0;
                classDist[y] = 0.0;
              }
            }
            normalise(classDistIndep);
            normalise(classDist);
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              m += fabs(classDistIndep[y] - classDist[y]);
            }
          }
        }

      cm[x1][x2] = m;
      cm[x2][x1] = m;
    }
  }
}
double getSymmetricalUncert(const xxyDist &dist, CategoricalAttribute x1,CategoricalAttribute x2)
{

	const double totalCount = dist.xyCounts.count;

	double x1Ent = 0.0;
	double x2Ent = 0.0;

	double m = 0.0;

	for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
		double px2 = dist.xyCounts.getCount(x2, v2) / totalCount;
		if (px2)
			x2Ent += px2 * log2(px2);
	}
	for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {

		double px1 = dist.xyCounts.getCount(x1, v1) / totalCount;
		if (px1)
			x1Ent += px1 * log2(px1);

		for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
			const InstanceCount x2Count = dist.xyCounts.getCount(x2, v2);
			const InstanceCount x1x2Count = dist.getCount(x1, v1, x2, v2);

			if (x1x2Count) {
				m += (x1x2Count / totalCount)
						* log2(x1x2Count / (px1 * x2Count));
			}

		}

	}
	return 2 * (m / (-x1Ent - x2Ent));

}

void getCondSymmUncert(xxyDist &dist, crosstab<float> &csu)
{
  const double totalCount = dist.xyCounts.count;


  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
    for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
      float m = 0.0;
      double x1yEnt = 0.0;
      double x2yEnt = 0.0;
      for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
        for (CatValue y = 0; y < dist.getNoClasses(); y++) {
          const double x1y = dist.xyCounts.getCount(x1, v1, y);
          if(x1y)
            x1yEnt += (x1y/totalCount) *
                       log2(x1y/dist.xyCounts.getClassCount(y));
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            if(v1==0){
              const double x2y = dist.xyCounts.getCount(x2, v2, y);
              if(x2y)
                x2yEnt += (x2y/totalCount) *
                           log2(x2y/dist.xyCounts.getClassCount(y));
            }
            const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
            if (x1x2y) {
              //const unsigned int yCount = dist->xyCounts.getClassCount(y);
              //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
              //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
              m += (x1x2y/totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                 (static_cast<double>(dist.xyCounts.getCount(x1, v1, y)) *
                  dist.xyCounts.getCount(x2, v2, y)));
            }
          }
        }
      }
      assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

      //ocurring when x1 and x2 are useless attributes, it is replaced by -1.
      if((-x1yEnt-x2yEnt)==0)
        m= -1;
      m = 2 * (m / (-x1yEnt-x2yEnt));

      csu[x1][x2] = m;
      csu[x2][x1] = m;
    }
  }
}

double chiSquare(const InstanceCount *cells, const unsigned int rows,
                                             const unsigned int cols) {
  std::vector<unsigned int> rowSums(rows, 0);
  std::vector<unsigned int> colSums(cols, 0);
  unsigned int n = 0;
  int degreesOfFreedom = (rows-1) * (cols - 1);

  if (degreesOfFreedom == 0) return 1.0;

  for (unsigned int r = 0; r < rows; r++) {
    for (unsigned int c = 0; c < cols; c++) {
      rowSums[r] += cells[r*cols + c];
      colSums[c] += cells[r*cols + c];
      n += cells[r*cols + c];
    }
  }

  double chisq = 0.0;

  for (unsigned int r = 0; r < rows; r++) {
    if (rowSums[r] != 0) {
      for (unsigned int c = 0; c < cols; c++) {
        if (colSums[c] != 0) {
          double expect = rowSums[r]*(colSums[c]/static_cast<double>(n));
          const double diff = cells[r*cols+c] - expect;
          chisq += (diff * diff) / expect;
        }
      }
    }
  }

  return alglib::chisquarecdistribution(degreesOfFreedom, chisq);
}

/*
 *
 *                __
 *                \                     P(x1,y|x2)
    CMI(X1,Y|X2)= /_   P(x1,y,x2) log ---------------
*               x1,x2,y               P(x1|x2)P(y|x2)
*
*/

void getAttClassCondMutualInf(xxyDist &dist, crosstab<float> &acmi, bool transpose){
    const double totalCount = dist.xyCounts.count;

  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float m1 = 0.0, m2 = 0.0;
        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                m1 += (x1x2y/totalCount) * log2(dist.xyCounts.getCount(x2,v2) * x1x2y /
                     (static_cast<double>(dist.getCount(x1, v1, x2, v2)) *
                      dist.xyCounts.getCount(x2, v2, y)));
                m2 += (x1x2y/totalCount) * log2(dist.xyCounts.getCount(x1,v1) * x1x2y /
                     (static_cast<double>(dist.getCount(x2, v2, x1, v1)) *
                      dist.xyCounts.getCount(x1, v1, y)));
              }
            }
          }
        }

		assert(m1 >= -0.00000001);
		assert(m2 >= -0.00000001);
		if (transpose == false) {
			//MI(x1;c|x2)
			acmi[x1][x2] = m1;
			//MI(x2;c|x1)
			acmi[x2][x1] = m2;
		} else {
			//MI(x1;c|x2)
			acmi[x2][x1] = m1;
			//MI(x2;c|x1)
			acmi[x1][x2] = m2;
		}
	}
  }
}
/*
 *
 *                   __
 *                   \                       P(x1,x2|x3,y)
 * MCMI(X1,X2|X3,Y)= /_ P(x1,x2,x3,y)log------------------------
 *                x1,x2,x3,y               P(x1|x3,y)P(x2|x3,y)
 *
 *
 */
void getMultCondMutualInf(xxxyDist &dist, std::vector<crosstab<float> > &mcmi){
  const double totalCount = dist.xxyCounts.xyCounts.count;

  for (CategoricalAttribute x1 = 2; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
        for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
            float m1 = 0.0, m2 = 0.0, m3 = 0.0;
            for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
              for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
                for (CatValue v3 = 0; v3 < dist.getNoValues(x3); v3++) {
                  for (CatValue y = 0; y < dist.getNoClasses(); y++) {
                    const double x1x2x3y = dist.getCount(x1, v1, x2, v2, x3, v3, y);
                    if (x1x2x3y) {
                      const unsigned int  x1x2y = dist.xxyCounts.getCount(x1, v1, x2, v2, y);
                      const unsigned int  x1x3y = dist.xxyCounts.getCount(x1, v1, x3, v3, y);
                      const unsigned int  x2x3y = dist.xxyCounts.getCount(x2, v2, x3, v3, y);
                      m1 += (x1x2x3y/totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x3, v3, y) * x1x2x3y /
                           (static_cast<double>(x1x3y) * x2x3y));
                      m2 += (x1x2x3y/totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x2, v2, y) * x1x2x3y /
                           (static_cast<double>(x1x2y) * x2x3y));
                      m3 += (x1x2x3y/totalCount) * log2(dist.xxyCounts.xyCounts.getCount(x1, v1, y) * x1x2x3y /
                           (static_cast<double>(x1x3y) * x1x2y));
                    }
                  }
                }
              }
            }

          assert(m1 >= -0.00000001);
          assert(m2 >= -0.00000001);
          assert(m3 >= -0.00000001);

          mcmi[x1][x2][x3] = m1;
          mcmi[x2][x1][x3] = m1;
          mcmi[x1][x3][x2] = m2;
          mcmi[x3][x1][x2] = m2;
          mcmi[x2][x3][x1] = m3;
          mcmi[x3][x2][x1] = m3;
        }
    }
  }
}
/*                  __
 *                  \                   P(x1,x2,y)
 * PMI(<X1,X2>,Y)=  /_   P(x1,x2,y)log---------------
 *                x1,x2,y              P(x1,x2)P(y)
 *
 */
void getPairMutualInf(xxyDist &dist,crosstab<float> &pmi)
{

	  const double totalCount = dist.xyCounts.count;

	  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
	      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
	        float m = 0.0;
	        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
	          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
	            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
	              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
	              if (x1x2y) {
	                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
	                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
	                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
	                m += (x1x2y/totalCount) * log2(totalCount * x1x2y /
	                     (static_cast<double>(dist.getCount(x1, v1, x2, v2)) *
	                      dist.xyCounts.getClassCount(y)));
	              }
	            }
	          }
	        }

	      assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision

	      pmi[x1][x2] = m;
	      pmi[x2][x1] = m;
	    }
	  }
}

/*                      __
 *                      \                      P(x1,x2,x3,y)
 * TMI(<X1,X2,X3>,Y)=   /_   P(x1,x2,x3,y)log-----------------
 *                  x1,x2,x3,y                P(x1,x2,x3)P(y)
 *
 */
void getTripleMutualInf(xxxyDist &dist,crosstab3D<float> &tmi)
{

	  const double totalCount = dist.xxyCounts.xyCounts.count;

	  for (CategoricalAttribute x1 = 2; x1 < dist.getNoCatAtts(); x1++) {
	      for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
		      for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {

				float m = 0.0;
				for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
				  for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
					  for (CatValue v3 = 0; v3 < dist.getNoValues(x3); v3++) {

						for (CatValue y = 0; y < dist.getNoClasses(); y++) {
						  const double x1x2x3y = dist.getCount(x1, v1, x2, v2,x3,v3, y);
						  if (x1x2x3y) {
							//const unsigned int yCount = dist->xyCounts.getClassCount(y);
							//const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
							//const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
							m += (x1x2x3y/totalCount) * log2(totalCount * x1x2x3y /
								 (static_cast<double>(dist.getCount(x1, v1, x2, v2,x3,v3)) *
								  dist.xxyCounts.xyCounts.getClassCount(y)));
								}
							}
						}
					}
				}

				assert(m >= -0.00000001);
				// CMI is always positive, but allow for some imprecision

				tmi[x1][x2][x3] = m;
				tmi[x1][x3][x2] = m;
				tmi[x2][x1][x3] = m;
				tmi[x2][x3][x1] = m;
				tmi[x3][x2][x1] = m;
				tmi[x3][x1][x2] = m;
				//pmi[x2][x1] = m;
			}
		}
	}


}


void getBothCondMutualInf(xxyDist &dist, crosstab<float> &cmi,
                             crosstab<float> &acmi){
  const double totalCount = dist.xyCounts.count;

  for (CategoricalAttribute x1 = 1; x1 < dist.getNoCatAtts(); x1++) {
      for (CategoricalAttribute x2 = 0; x2 < x1; x2++) {
        float acmi_m1 = 0.0, acmi_m2 = 0.0, m = 0.0;
        for (CatValue v1 = 0; v1 < dist.getNoValues(x1); v1++) {
          for (CatValue v2 = 0; v2 < dist.getNoValues(x2); v2++) {
            for (CatValue y = 0; y < dist.getNoClasses(); y++) {
              const double x1x2y = dist.getCount(x1, v1, x2, v2, y);
              if (x1x2y) {
                acmi_m1 += (x1x2y/totalCount) * log2(dist.xyCounts.getCount(x2,v2) * x1x2y /
                     (static_cast<double>(dist.getCount(x1, v1, x2, v2)) *
                      dist.xyCounts.getCount(x2, v2, y)));
                acmi_m2 += (x1x2y/totalCount) * log2(dist.xyCounts.getCount(x1,v1) * x1x2y /
                     (static_cast<double>(dist.getCount(x2, v2, x1, v1)) *
                      dist.xyCounts.getCount(x1, v1, y)));
                m += (x1x2y/totalCount) * log2(dist.xyCounts.getClassCount(y) * x1x2y /
                     (static_cast<double>(dist.xyCounts.getCount(x1, v1, y)) *
                      dist.xyCounts.getCount(x2, v2, y)));
              }
            }
          }
        }

      assert(m >= -0.00000001);
      assert(acmi_m1 >= -0.00000001);
      assert(acmi_m2 >= -0.00000001);
      cmi[x1][x2] = m;
      cmi[x2][x1] = m;
      //MI(x1;c|x2)
      acmi[x1][x2] = acmi_m1;
      //MI(x2;c|x1)
      acmi[x2][x1] = acmi_m2;
    }
  }

}

void getrow(crosstab<InstanceCount> &xtab, unsigned int noClasses, unsigned int trow, std::vector<InstanceCount> &Crow){
  for (unsigned int k = 0; k < noClasses; k++) {
    Crow[k] = xtab[trow][k];
  }
}

void getcol(crosstab<InstanceCount> &xtab, unsigned int noClasses, unsigned int tcol, std::vector<InstanceCount> &Ccol){
  for (unsigned int k = 0; k < noClasses; k++) {
    Ccol[k] = xtab[k][tcol];
  }
}

unsigned long long int dotproduct(std::vector<InstanceCount> &Crow, std::vector<InstanceCount> &Ccol, unsigned int noClasses){
  unsigned long long int val = 0;
  fflush(stdout);
  for (unsigned int k = 0; k < noClasses; k++) {
    val += static_cast<unsigned long long>(Crow[k])*static_cast<unsigned long long>(Ccol[k]);
  }
  return val;
}

double calcMCC(crosstab<InstanceCount> &xtab){
  // Compute MCC for multi-class problems as in http://rk.kvl.dk/
    unsigned int noClasses = xtab[0].size();
    double MCC = 0.0;

    //Compute N, sum of all values
    double N = 0.0;
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        N += xtab[k][l];
      }
    }

    //compute correlation coefficient
    double trace = 0.0;
    for (unsigned int k = 0; k < noClasses; k++) {
      trace += xtab[k][k];
    }
    //sum row col dot product
    unsigned long long int rowcol_sumprod=0;
    std::vector<InstanceCount> Crow (noClasses);
    std::vector<InstanceCount> Ccol (noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        getrow(xtab,noClasses,k,Crow);
        getcol(xtab,noClasses,l,Ccol);
        rowcol_sumprod += dotproduct(Crow, Ccol, noClasses);
      }
    }

    //sum over row dot products
    unsigned long long int rowrow_sumprod=0;
    std::vector<InstanceCount> Crowk (noClasses);
    std::vector<InstanceCount> Crowl (noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        getrow(xtab,noClasses,k,Crowk);
        getrow(xtab,noClasses,l,Crowl);
        rowrow_sumprod += dotproduct(Crowk, Crowl, noClasses);
      }
    }

    //sum over col dot products
    unsigned long long int colcol_sumprod=0;
    std::vector<InstanceCount> Ccolk (noClasses);
    std::vector<InstanceCount> Ccoll (noClasses);
    for (unsigned int k = 0; k < noClasses; k++) {
      for (unsigned int l = 0; l < noClasses; l++) {
        getcol(xtab,noClasses,k,Ccolk);
        getcol(xtab,noClasses,l,Ccoll);
        colcol_sumprod += dotproduct(Ccolk, Ccoll, noClasses);
      }
    }

    double cov_XY = N*trace - rowcol_sumprod;
    double cov_XX = N*N - rowrow_sumprod;
    double cov_YY = N*N - colcol_sumprod;
    double denominator = sqrt(cov_XX*cov_YY);

    if(denominator > 0){
      MCC = cov_XY / denominator;
    }
    else if (denominator == 0){
      MCC = 0;
    }
    else{
      printf("Error when calculating MCC2");
    }
    return MCC;
}
