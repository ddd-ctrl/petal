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

#include "crosstab3d.h"
#include "crosstab.h"
#include "xyDist.h"
#include "xxyDist.h"
#include "xxxyDist.h"

#include "math.h"
/**
<!-- globalinfo-start -->
 * File that includes different correlation measures among variables.<br/>
 <!-- globalinfo-end -->
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 * @author Ana M. Martinez (anam.martinez@monash.edu)
 */



/**
 * Calculates the information gain between an attribute in dist and the class
 *
 * @param dist  counts for the xy distributions.
 * @param a the attribute.
 */
double getInfoGain(xyDist& dist, CategoricalAttribute a);

/**
 * Calculates the information of an attribute
 *
 * @param dist  counts for the xy distributions.
 * @param a the attribute.
 */
double getInformation(xyDist& dist, CategoricalAttribute a);

/**
 * Calculates the gain ratio between the attribute and the
 * class
 *
 * @param dist  counts for the xy distributions.
 * @param a the attribute.
 */
double getGainRatio(xyDist& dist, CategoricalAttribute a);

/**
 * Calculates the gain ratio for a potential split class distribution
 *
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right);

/**
 * Calculates the gain ratio for a potential split class distribution
 *
 * @param left the distribution for the first branch
 * @param right the distribution for the second branch
 * @param unknown the distribution for instances with missing values
 */
double getGainRatio(std::vector<InstanceCount> &left, std::vector<InstanceCount> &right, std::vector<InstanceCount> &unknown);

/**
 * Calculates the mutual information between the attributes in dist and the
 * class
 *
 * MI(X;C) = H(X) - H(X|C)
 *
 * @param dist  counts for the xy distributions.
 * @param[out] mi mutual information between the attributes and the class.
 */
void getMutualInformation(xyDist &dist, std::vector<float> &mi);

void getMutualInformation(xyDist &dist, std::vector<double> &mi);

/**
 * Calculates the symmetrical uncertainty between the attributes in dist and the
 * class
 *
 * SU(X;Y) = 2 . MI(X,C) / ( H(X)+H(C))
 *
 * @param dist  counts for the xy distributions.
 * @param[out] su symmetrical uncertainty between the attributes and the class.
 */
void getSymmetricalUncert(xyDist &dist, std::vector<float> &su);


/**
 * Calculates the class conditional mutual information between the attributes in
 * dist conditioned on the class
 *
 * CMI(X;Y|C) = H(X|C) - H(X|Y,C)
 *
 * @param dist  counts for the xxy distributions.
 * @param[out] cmi class conditional mutual information between the attributes.
 */
void getCondMutualInf(xxyDist &dist, crosstab<float> &cmi);


/**
 * Calculates the mutual information between the attributes
 * MI(X;Y) = H(X) - H(X|Y)
 * @param dist  counts for the xxy distributions.
 * @param[out] ami pair attributes mutual information.
*/
void getAttMutualInf(xxyDist &dist, crosstab<float> &ami);


/**
 * Calculate the difference between the probabilities considering independency
 * over dependency between the attributes X and Y: abs ( p(X|C) - P(X|Y,C) ).
 *
 * @param dist  counts for the xy distributions.
 * @param[out] cm class conditional correlation measure between the attributes.
 */
void getErrorDiff(xxyDist &dist, crosstab<float> &cm);


/**
 * Calculates the class conditional symmetrical uncertainty between the
 * attributes in dist conditioned on the class
 *
 * CSU(X;Y|C) = 2 . CMI(X;Y|C)/( H(X|C) + H(Y|C) )
 *
 * @param dist  counts for the xy distributions.
 * @param[out] csu class conditional symmetrical uncertainty between the attributes.
 */
void getCondSymmUncert(xxyDist &dist, crosstab<float> &csu);


/**
 * Chi-squared test for independence, one attribute against the class
 *
 * @param cells counts for the attribute's values and the class
 * @param rows number of values for the attribute
 * @param cols number of classes
 *
 * @return The complemented Chi-square distribution
 */
double chiSquare(const InstanceCount *cells, const unsigned int rows,
                                             const unsigned int cols);



/**
 * Calculates the conditional mutual information between the class and the attributes in
 * dist conditioned on another attribute in dist
 *
 * ACMI(X;C|Y) = H(X|Y) - H(X|Y,C)
 *
 * @param dist  counts for the xy distributions.
 * @param[out] atcmi conditional mutual information between the attributes and the class
 * given another attribute.
 */
void getAttClassCondMutualInf(xxyDist &dist, crosstab<float> &acmi, bool transpose=false);

/**
 * Calculates the class conditional mutual information between the attributes in
 * dist conditioned on the class and another attribute
 *
 * MCMI(X;Z|C,Y) = H(X|C,Y) - H(X|Z,C,Y)
 *
 * @param dist  counts for the xy distributions.
 * @param[out] mcmi class conditional mutual information between the attributes in
 * dist conditioned on the class and another attribute
 */
void getMultCondMutualInf(xxxyDist &dist, std::vector<crosstab<float> > &mcmi);



/*
 * Calculate the mutual information between one pair of attributes and class
 *
 * PMI(<X1,X2>,Y)=H(<X1,X2>)-H(<X1,X2>|Y)
 *
 * @param dist  counts for the xxy distributions.
 * @param[out] pmi  mutual information between pair of attributes and class
 *
 */
void getPairMutualInf(xxyDist &dist,crosstab<float> &pmi);



/*
 * Calculate the mutual information between triple of attributes and class
 *
 * PMI(<X1,X2,X3>,Y)=H(<X1,X2,X3>)-H(<X1,X2,X3>|Y)
 *
 * @param dist  counts for the xxxy distributions.
 * @param[out] pmi  mutual information between pair of attributes and class
 *
 */
void getTripleMutualInf(xxxyDist &dist,crosstab3D<float> &tmi);



/**
 * Calculates the following measures at the same time:
 * CMI(X;Y|C) = H(X|C) - H(X|Y,C)
 * ACMI(X;C|Y) = H(X|Y) - H(X|Y,C)
 *
 * @param dist  counts for the xy distributions.
 * @param[out] cmi class conditional mutual information between the attributes.
 * @param[out] acmi conditional mutual information between the attributes and the class
 * given another attribute.
 */
void getBothCondMutualInf(xxyDist &dist, crosstab<float> &cmi,
                             crosstab<float> &acmi);

/// Calculates the Matthew's Correlation Coefficient from a set of TP, FP, TN, FN counts
inline double calcBinaryMCC(const InstanceCount TP, const InstanceCount FP, const InstanceCount TN, const InstanceCount FN) {
  if (TP+TN == 0) return -1.0;
  if (FP+FN == 0) return 1.0;
  if ((TP+FN == 0) || (TN+FP == 0) || (TP+FP == 0) || (TN+FN == 0)) return 0.0;
  else return (static_cast<double>(TP)*TN-static_cast<double>(FP)*FN)
                    / sqrt(static_cast<double>(TP+FP)
                            * static_cast<double>(TP+FN)
                            * static_cast<double>(TN+FP)
                            * static_cast<double>(TN+FN));

}

/**
 * Calculates the Matthew's Correlation Coefficient given a confusion matrix (any number of classes) as in http://rk.kvl.dk/
 *
 * @param xtab  confusion matrix.
 * @return the MCC
 */

double calcMCC(crosstab<InstanceCount> &xtab);


/**
 * Calculate the symmetrical uncertainty between two attributes
 */
double getSymmetricalUncert(const xxyDist &dist, CategoricalAttribute x1,CategoricalAttribute x2);

