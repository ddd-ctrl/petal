#include "xxxyDist.h"
#include "utils.h"

xxxyDist::xxxyDist() {

}
xxxyDist::xxxyDist(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxyDist::reset(InstanceStream& stream) {

	metaData_ = stream.getMetaData();

	xxyCounts.reset(stream);

	noCatAtts_ = metaData_->getNoCatAtts();
	noClasses_ = metaData_->getNoClasses();

	//out vector
	count.resize(noCatAtts_);
	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {

		//second vector
		count[x1].resize(metaData_->getNoValues(x1) * x1);
		for (CatValue v1 = 0; v1 < metaData_->getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {

				//third vector
				count[x1][v1 * x1 + x2].resize(
						metaData_->getNoValues(x2) * x2);
				for (CatValue v2 = 0; v2 < metaData_->getNoValues(x2);
						v2++) {
					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {

						//inner vector
						count[x1][v1 * x1 + x2][v2 * x2 + x3].assign(
								metaData_->getNoValues(x3) * noClasses_,
								0);
					}
				}
			}
		}
	}
}

xxxyDist::~xxxyDist(void) {

}

void xxxyDist::update(const instance &i) {
	xxyCounts.update(i);

	const CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
		const CatValue v1 = i.getCatVal(x1);

		XXYSubDist xxySubDist(getXXYSubDist(x1, v1), noClasses_ );

		for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
			const CatValue v2 = i.getCatVal(x2);

//		    XYSubDist xySubDist(getXYSubDist(x1, v1,x2,v2), noClasses_);

			for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
				const CatValue v3 = i.getCatVal(x3);

//				xySubDist.incCount(x3,v3,theClass);
				 xxySubDist.incCount(x2,v2,x3, v3, theClass);

				 assert(*ref(x1,v1,x2,v2,x3,v3,theClass) <= xxyCounts.xyCounts.count);
			}
		}
	}

}


void xxxyDist::outputDist()
{

	//0: zero instance
	//1: one instance
	//2:two instances
	//3: [3,254] instances
	//4: [0xFF,0xFFFFFFFE] instances
	//5:  [0xFFFFFFFF,0xFFFFFFFFFFFFFFFF]
	//6:  else
	std::vector<InstanceCount> dist(7);
	InstanceCount count;

	for(unsigned short i=0;i<7;i++ )
		dist[i]=0;

	for (CategoricalAttribute x1 = 2; x1 < noCatAtts_; x1++) {
		for (CatValue v1 = 0; v1 < metaData_->getNoValues(x1); ++v1) {

			for (CategoricalAttribute x2 = 1; x2 < x1; x2++) {
				for (CatValue v2 = 0; v2 < metaData_->getNoValues(x2); ++v2) {

					for (CategoricalAttribute x3 = 0; x3 < x2; x3++) {
						for (CatValue v3 = 0; v3 < metaData_->getNoValues(x3);
								++v3) {

							for (CatValue y = 0; y < noClasses_; y++) {

								count = getCount(x1, v1, x2, v2, x3, v3, y);

								if (count == 0)
									dist[0]++;
								else if (count == 1)
									dist[1]++;
								else if (count == 2)
									dist[2]++;
								else if (count < 255)
									dist[3]++;
								else if (count < 0xFFFFFFFF)
									dist[4]++;
								else if (count < 0xFFFFFFFFFFFFFFFF)
									dist[5]++;
								else
									dist[6]++;

							}
						}
					}
				}
			}
		}
	}

	printf("0:\t%u\t%s\n",dist[0],"0");
	printf("1:\t%u\t%s\n",dist[1],"1");
	printf("2:\t%u\t%s\n",dist[2],"2");
	printf("3:\t%u\t%s\n",dist[3],"[3,254]");
	printf("4:\t%u\t%s\n",dist[4],"[0xFF,0xFFFFFFFE]");
	printf("5:\t%u\t%s\n",dist[5],"[0xFFFFFFFF,0xFFFFFFFFFFFFFFFF]");
	printf("6:\t%u\t%s\n",dist[6],"else");

}

void xxxyDist::clear(){
  count.clear();
  xxyCounts.clear();
}
