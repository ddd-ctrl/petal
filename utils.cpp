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

#include <functional>
#include "utils.h"
#include "ALGLIB_specialfunctions.h"
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux__
#include <sys/time.h>
#include <sys/resource.h>
#endif

// get the number of instances in a data file
InstanceCount getFileCount(const char *fname, const bool update) {
  char *cntfname;
  safeAlloc(cntfname, strlen(fname) + 5);

  sprintf(cntfname, "%s%s", fname, ".cnt");

  struct stat buf;

  int result;

  // Get data associated with cnt file:
  result = stat( cntfname, &buf );

  // Check if the cnt file exists
  if( result == 0 ) {
      struct stat buf2;

    // Get data associated with fname:
    result = stat( fname, &buf2 );

    // Check if the file exists
    if( result == 0 ) {
      // check whether the cnt file is more recent
      if (buf.st_mtime > buf2.st_mtime) {
        InstanceCount cnt = 0;

        FILE *f = fopen(cntfname, "r");

        if (f != NULL) {
          (void)fscanf(f, "%" ICFMT, &cnt);
          fclose(f);
          delete []cntfname;
          return cnt;
        }
      }
    }
  }

  FILE *f = fopen(fname, "r");

  int c = getc(f);
  InstanceCount cnt = 0;

  while (c != EOF) {
    // skip lines containing only non-printable characters
    while (c <= ' ') {
      if (c == EOF) goto exitloop;
      c = getc(f);
    }

    cnt++;
    while (c != '\n' && c != '\r' && c != EOF) {
      c = getc(f);
    }
exitloop:;
  }

  fclose(f);

  if (update) {
    FILE *f = fopen(cntfname, "w");
    fprintf(f, "%" ICFMT "\n", cnt);
  }

  delete []cntfname;

  return cnt;
}

unsigned int getUIntListFromStr(char *s, std::vector<unsigned int*> &vals, char const *context) {
  if (s == NULL) return 0;

  unsigned int length=0;
  for (unsigned int i = 0; i < vals.size(); i++) {
    if (*s != '\0') {
      if (*s < '0' || *s > '9') {
        error("Encountered '%s' when expecting a list of unsigned integers for %s", s, context);
      }
      unsigned int v = 0;
      while (*s >= '0' && *s <= '9') {
        v *= 10;
        v += *s++ - '0';
      }
      *vals[i] = v;
      if (*s != ',' && *s != '\0') {
        error("Encountered '%s' when expecting a list of unsigned integers for %s", s, context);
      }
      if (*s == ',') s++;
      length++;
    }
  }
  return length;
}

InstanceCount countCases(FILEtype *f) {
  InstanceCount count = 0;

  rewind(f);

  int c = getc(f);
  bool lineHasContent = false;

  while (c != EOF) {
    if (c == '\n') {
      if (lineHasContent) count++;
      lineHasContent = false;
    }
    else if (!isspace(c)) {
      lineHasContent = true;
    }
    c = getc(f);
  }

  if (lineHasContent) {
    // allow for a case that is terminated by EOF
    count++;
  }

  rewind(f);

  return count;
}

bool streq(char const *s1, char const *s2, const bool caseSensitive) {
  if (caseSensitive) {
    while (*s1 != '\0' && *s1 == *s2) {
      s1++;
      s2++;
    }
  }
  else {
    while (*s1 != '\0' && tolower(*s1) == tolower(*s2)) {
      s1++;
      s2++;
    }
  }

  return *s1 == '\0' && *s2 == '\0';
}

void printResults(crosstab<InstanceCount> &xtab, const InstanceStream &is) {
  // find the maximum value to determine how wide the output fields need to be
  InstanceCount maxval = 0;
  for (CatValue predicted = 0; predicted < is.getNoClasses(); predicted++) {
    for (CatValue y = 0; y < is.getNoClasses(); y++) {
      if (xtab[y][predicted] > maxval) maxval = xtab[y][predicted];
    }
  }

  // calculate how wide the output fields should be
  const int printwidth = max(4, printWidth(maxval));

  // print the heading line of class names
  printf("\n");
  for (CatValue y = 0; y < is.getNoClasses(); y++) {
    printf(" %*.*s", printwidth, printwidth, is.getClassName(y));
  }
  printf(" <- Actual class\n");

  // print the counts of instances classified as each class
  for (CatValue predicted = 0; predicted < is.getNoClasses(); predicted++) {
    for (CatValue y = 0; y < is.getNoClasses(); y++) {
      printf(" %*" ICFMT, printwidth, xtab[y][predicted]);
    }
    printf(" <- %s predicted\n", is.getClassName(predicted));
  }
}

// print an error mesage and exit.  Supports printf style format and arguments
void error(const char *fmt, ...)
{ va_list v_args;
  va_start(v_args, fmt);
  vfprintf(stderr, fmt, v_args);
  va_end(v_args);
  putc('\n', stderr);
  exit(0);
}

// print an error mesage without exiting.  Supports printf style format and arguments
void errorMsg(const char *fmt, ...)
{ va_list v_args;
  va_start(v_args, fmt);
  vfprintf(stderr, fmt, v_args);
  va_end(v_args);
  putc('\n', stderr);
}

void print(const std::vector<double> &vals) {
  const char *sep = "";
  for (std::vector<double>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%0.4f", sep, *it);
    sep = ", ";
  }
}

void print(const std::vector<float> &vals) {
  const char *sep = "";
  for (std::vector<float>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%0.4f", sep, *it);
    sep = ", ";
  }
}
void print(const std::vector<unsigned int> &vals) {
  const char *sep = "";
  for (std::vector<unsigned int>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%d", sep, *it);
    sep = ", ";
  }
}

void print(const std::vector<long int> &vals) {
  const char *sep = "";
  for (std::vector<long int>::const_iterator it = vals.begin(); it != vals.end(); it++) {
    printf("%s%ld", sep, *it);
    sep = ", ";
  }
}

void summariseUsage() {
#ifdef __linux__
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  printf("total time: %ld seconds\nmaximum memory: %ld Kb\n", usage.ru_utime.tv_sec+usage.ru_stime.tv_sec, usage.ru_maxrss);
#endif
}

CatValue discretise(const NumValue val, std::vector<NumValue> &cuts) {
  if (val == MISSINGNUM) {
    return cuts.size()+1;
  }
  else if (cuts.size() == 0) {
    return 0;
  }
  else if (val > cuts.back()) {
    return cuts.size();
  }
  else {
    unsigned int upper = cuts.size()-1;
    unsigned int lower = 0;

    while (upper > lower) {
      const unsigned int mid = lower + (upper-lower) / 2;

      if (val <= cuts[mid]) {
        upper = mid;
      }
      else {
        lower = mid+1;
      }
    }

    assert(upper == lower);
    return upper;
  }
}

/// Calculate the Area Under the ROC Curve

//Hand D J , Till R J . A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems[J].
//Machine Learning, 2001, 45(2):171-186.


double calcBinaryAUC(
        std::vector<std::vector<double> >& probs, //< the sequence of predicted probabilities for each class
        const std::vector<CatValue>& trueClasses //< the sequence of true classes
        ) {

        const CatValue classCount= probs.size();

        std::vector<unsigned int> order1;
        std::vector<unsigned int> order2;
        //< indexes of classifications to be sorted in ascending order on predicted probability of positive class

        for (unsigned int i = 0; i < trueClasses.size(); i++) {
            order1.push_back(i);
            order2.push_back(i);
        }


        unsigned long long int sumRank;
        double A12,A21,auc,sumA=0;
        unsigned long long int n1,n2;

        for (CatValue y1 = 0; y1 < classCount-1; y1++) {


			n1=0;
			for(unsigned int i=0;i<trueClasses.size();i++)
			{
				//printf("%d,",trueClasses[i]);
				if(trueClasses[i]==y1)
					n1++;
			}



           for (CatValue y2 = y1+1; y2 < classCount; y2++) {

              n2=0;
              for(unsigned int i=0;i<trueClasses.size();i++)
              {
                if(trueClasses[i]==y2)
                  n2++;
              }


              sumRank=0;
              IndirectCmpClassAscending<double> cmp1(&probs[y1][0]);
              std::sort(order1.begin(), order1.end(), cmp1);


              for(unsigned int i=0;i<trueClasses.size();i++)
              {
               // printf("%d,",order1[i]);
                if(trueClasses[order1[i]]==y1)
                  sumRank+=i+1;
              }
             // printf("\n" );
              A12= (sumRank-n1*(n1+1)/2.0)/(n1*n2);

              sumRank=0;
              IndirectCmpClassAscending<double> cmp2(&probs[y2][0]);
              std::sort(order2.begin(), order2.end(), cmp2);
              for(unsigned int i=0;i<trueClasses.size();i++)
              {
                //printf("%d,",order2[i]);
                if(trueClasses[order2[i]]==y2)
                  sumRank+=i+1;
              }
              //printf("\n" );
              A21= (sumRank-n2*(n2+1)/2.0)/(n1*n2);

              sumA+=(A12+A21)/2;
           }
        }
        auc=2*sumA/(classCount*(classCount-1));
        return auc;
}


/// Calculate the Area Under the ROC Curve

//Hand D J , Till R J . A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems[J].
//Machine Learning, 2001, 45(2):171-186.


double calcMultiAUC(
        std::vector<std::vector<double> >& probs, //< the sequence of predicted probabilities for each class
        const std::vector<CatValue>& trueClasses //< the sequence of true classes
        ) {

        CatValue classCount=probs.size();
        CatValue combinationCount=classCount*(classCount-1)/2;

		//for very large data sets such as poker-hand, the unsigned int type for sumRank is not enough
		//so we use unsigned long long type.
        unsigned long long sumRank;
        double A12,A21,auc,sumA=0;
        unsigned long long n1,n2;
		unsigned int k;

		     //   unsigned long long sumRanktest;
			//	double A13;

        for (CatValue y1 = 0; y1 < classCount-1; y1++) {

           for (CatValue y2 = y1+1; y2 < classCount; y2++) {

              std::vector<unsigned int> subTrueClasses;
              std::vector<std::vector<double> > subProbs(2);
              std::vector<unsigned int> order1;
              std::vector<unsigned int> order2;

              n1=n2=0;
              k=0;

              for(unsigned int i=0;i<trueClasses.size();i++)
              {
                //printf("%d,",trueClasses[i]);

                if(trueClasses[i]==y1||trueClasses[i]==y2)
                {
                  subProbs[0].push_back(probs[y1][i]);
                  subProbs[1].push_back(probs[y2][i]);
                  subTrueClasses.push_back(trueClasses[i]);

                  if(trueClasses[i]==y1)
                  {
                    n1++;
                  }
                  else
                  {
                    n2++;
                  }
                  order1.push_back(k);
                  order2.push_back(k);
                  k++;
                }
              }

              if(n1==0||n2==0)
              {
                combinationCount--;
                continue;
              }



			  
			  /*
			  sumRanktest=0;
			  IndirectCmpClassAscending<double> cmp3(&subProbs[0][0]);
              std::sort(subTrueClasses.begin(), subTrueClasses.end(),cmp3);


              for(unsigned int i=0;i<subTrueClasses.size();i++)
              {
               // printf("%d,",order1[i]);
                if(subTrueClasses[i]==y1)
                  sumRanktest+=i+1;
              }
             // printf("\n" );
              A13= (sumRanktest-n1*(n1+1)/2.0)/(n1*n2);

			  */

              sumRank=0;
              IndirectCmpClassAscending<double> cmp1(&subProbs[0][0]);
              std::sort(order1.begin(), order1.end(),cmp1);


              for(unsigned int i=0;i<subTrueClasses.size();i++)
              {
				//  if(y1==0&&y2==1)
               // printf("%.8f,",subProbs[0][i]);
                if(subTrueClasses[order1[i]]==y1)
                  sumRank+=i+1;
              }
            // if(y1==0&&y2==1)  printf("\n" );
              A12= (sumRank-n1*(n1+1)/2.0)/(n1*n2);

              sumRank=0;
              IndirectCmpClassAscending<double> cmp2(&subProbs[1][0]);
              std::sort(order2.begin(), order2.end(), cmp2);
              for(unsigned int i=0;i<subTrueClasses.size();i++)
              {
                //printf("%d,",order2[i]);
                if(subTrueClasses[order2[i]]==y2)
                  sumRank+=i+1;
              }
              //printf("\n" );
              A21= (sumRank-n2*(n2+1)/2.0)/(n1*n2);

			  

              sumA+=(A12+A21)/2;

           }
        }
		
        auc=sumA/combinationCount;
        return auc;
}

/// Calculate the Area Under the Precision Recall Curve
void calcAUPRC(
        std::vector<std::vector<float> >& probs, //< the sequence of predicted probabilities for each class
        const std::vector<CatValue>& trueClasses, //< the sequence of true classes
        InstanceStream::MetaData& metadata
        ) {
  std::vector<unsigned int> order;  //< indexes of classifications to be sorted in ascending order on predicted probability of positive class

  for (unsigned int i = 0; i < trueClasses.size(); i++) {
    order.push_back(i);
  }

  for (CatValue y = 0; y < metadata.getNoClasses(); y++) {
    InstanceCount POS = 0;
    InstanceCount NEG = 0;
    InstanceCount TP = 0;
    InstanceCount PREDPOS = 0;

    std::vector<CatValue>::const_iterator it = trueClasses.begin();

    while (it != trueClasses.end()) {
      if (*it == y) ++POS;
      else ++NEG;
      ++it;
    }

    IndirectCmpClassAscending<float> cmp(&probs[y][0]);

    std::sort(order.begin(), order.end(), cmp);

    float lastProb = 0.0;

    // initially everything is classified as positive
    TP = POS;
    PREDPOS = POS + NEG;

    double lastRecall = 1.0;  // up until the lowest prob prediction, recall is 1.0.
    double lastPrecision = TP / static_cast<double>(PREDPOS); // precision when everything classified pos

    double auc = 0.0;

    for (unsigned int i = 0; i < trueClasses.size(); ++i) {
      unsigned int idx = order[i];

      --PREDPOS;

      if (trueClasses[idx] == y) {
        --TP;
      }

      const float thisProb = probs[y][idx];

      if (thisProb != lastProb) {
        const double recall = TP / static_cast<double>(POS);

        if (recall != lastRecall) {
          const double delta = lastRecall - recall;

          auc += delta * lastPrecision;

          lastRecall = recall;
          lastPrecision = TP / static_cast<double>(PREDPOS); // precision before recall next changes
        }

        lastProb = thisProb;
      }
    }

    auc += lastRecall * lastPrecision;

    printf("\nArea under precision-recall curve %s: %f", metadata.getClassName(y), auc);
  }

  putchar('\n');
}

