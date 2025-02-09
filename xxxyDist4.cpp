#include "xxxyDist4.h"
#include "xxxyDist.h"
#include "globals.h"
#include "utils.h"
//The distribution class  xxxyDist4 stores selected numbers of children and  parents.


xxxyDist4::xxxyDist4() {
}

xxxyDist4::xxxyDist4(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxyDist4::setOrder(std::vector<CategoricalAttribute> &order) {
	order_ = order;
}
void xxxyDist4::setNoSelectedCatAtts(unsigned int noSelectedCatAtts)
{
	noSelectedCatAtts_=noSelectedCatAtts;
}

void xxxyDist4::reset(InstanceStream& stream) {

	metaData_ = stream.getMetaData();

	xxyCounts_.reset(stream);

//	noCatAtts_ = metaData_->getNoCatAtts();
	noClasses_ = metaData_->getNoClasses();



	//out vector
	count_.resize(noSelectedCatAtts_);
	for (CategoricalAttribute x1 = 2; x1 < noSelectedCatAtts_; x1++) {

		//second vector
		count_[x1].resize(getNoValues(x1) * x1);

		for (CatValue v1 = 0; v1 < getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {

				//third vector
				count_[x1][v1 * x1 + x2].resize(getNoValues(x2) * x2);
				for (CatValue v2 = 0; v2 < getNoValues(x2); v2++) {
					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {

						//inner vector
						count_[x1][v1 * x1 + x2][v2 * x2 + x3].assign(
								getNoValues(x3) * noClasses_, 0);
					}
				}
			}
		}
	}
}

xxxyDist4::~xxxyDist4(void) {

}

void xxxyDist4::update(const instance &i) {
	xxyCounts_.update(i);

	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 =2; x1 < noSelectedCatAtts_; x1++) {
		CatValue v1 = i.getCatVal(order_[x1]);

		constXXYSubDist4 xxySubDist(getXXYSubDist(x1, v1), noClasses_);

		for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
			CatValue v2 = i.getCatVal(order_[x2]);

 		    XYSubDist xySubDist(xxySubDist.getXYSubDist(x2,v2), noClasses_);

			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				CatValue v3 = i.getCatVal(order_[x3]);

 				xySubDist.incCount(x3,v3,theClass);

//				if(!(*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count))
//					printf("error!\n");

				assert(
						*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts_.xyCounts.count);
			}
		}
	}

}

void xxxyDist4::clear() {
	count_.clear();
	xxyCounts_.clear();
}
