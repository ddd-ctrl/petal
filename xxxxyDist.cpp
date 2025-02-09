#include "xxxxyDist.h"
#include "utils.h"

xxxxyDist::xxxxyDist() {

}
xxxxyDist::xxxxyDist(InstanceStream& stream) {
	reset(stream);

	// pass through the stream updating the counts incrementally
	stream.rewind();

	instance i;

	while (stream.advance(i)) {
		update(i);
	}

}

void xxxxyDist::reset(InstanceStream& stream) {
	
	metaData_ = stream.getMetaData();

	xxxyCounts.reset(stream);

	noCatAtts_ = metaData_->getNoCatAtts();
	noClasses_ = metaData_->getNoClasses();

	//out vector
	count.resize(noCatAtts_);
	for (CategoricalAttribute x1 = 3; x1 < noCatAtts_; x1++) {

		//second vector
		count[x1].resize(metaData_->getNoValues(x1) * x1);
		for (CatValue v1 = 0; v1 < metaData_->getNoValues(x1); v1++) {
			for (CategoricalAttribute x2 = 2; x2 < x1; x2++) {

				//third vector
				count[x1][v1 * x1 + x2].resize(
						metaData_->getNoValues(x2) * x2);
				for (CatValue v2 = 0; v2 < metaData_->getNoValues(x2);
						v2++) {
					for (CategoricalAttribute x3 = 1; x3 < x2; x3++) {

						//fourth vector
						count[x1][v1 * x1 + x2][v2 * x2 + x3].resize(
								metaData_->getNoValues(x3) * x3);
						for (CatValue v3 = 0;
								v3 < metaData_->getNoValues(x3); v3++)
							for (CategoricalAttribute x4 = 0; x4 < x3; x4++)

								//inner vector
								count[x1][v1 * x1 + x2][v2 * x2 + x3][v3 * x3
										+ x4].assign(
										metaData_->getNoValues(x4)
												* noClasses_, 0);
					}
				}
			}
		}
	}
}

xxxxyDist::~xxxxyDist(void) {

}

void xxxxyDist::update(const instance &i) {
	xxxyCounts.update(i);

	CatValue theClass = i.getClass();

	for (CategoricalAttribute x1 = 3; x1 < noCatAtts_; x1++) {
		CatValue v1 = i.getCatVal(x1);

		for (CategoricalAttribute x2 = 2; x2 < x1; x2++) {
			CatValue v2 = i.getCatVal(x2);

			for (CategoricalAttribute x3 = 1; x3 < x2; x3++) {
				CatValue v3 = i.getCatVal(x3);

				for (CategoricalAttribute x4 = 0; x4 < x3; x4++) {
					CatValue v4 = i.getCatVal(x4);

					(*ref(x1, v1, x2, v2, x3, v3, x4, v4, theClass))++;
				}
			}
		}

	}
}



void xxxxyDist::outputDist()
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

	for (CategoricalAttribute x1 = 3; x1 < noCatAtts_; x1++) {
		for (CatValue v1 = 0; v1 < metaData_->getNoValues(x1); ++v1) {

			for (CategoricalAttribute x2 = 2; x2 < x1; x2++) {
				for (CatValue v2 = 0; v2 < metaData_->getNoValues(x2); ++v2) {

					for (CategoricalAttribute x3 = 1; x3 < x2; x3++) {
						for (CatValue v3 = 0; v3 < metaData_->getNoValues(x3);
								++v3) {

							for (CategoricalAttribute x4 = 0; x4 < x3; x4++) {
								for (CatValue v4 = 0;
										v4 < metaData_->getNoValues(x4); ++v4) {
									for (CatValue y = 0; y < noClasses_; y++) {

										count = getCount(x1, v1, x2, v2, x3, v3,
												x4, v4, y);

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


void xxxxyDist::clear() {
	//count.swap(std::vector<int>());
	//std::vector<std::vector<std::vector<std::vector<InstanceCount> > > >( count.begin(), count.end() ).swap ( count );
	count.clear();
	xxxyCounts.clear();
}
