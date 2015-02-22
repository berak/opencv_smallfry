/*
 * BOWKmajorityTrainer.h
 *
 *  Created on: Sep 26, 2013
 *      Author: andresf
 */

#ifndef BOWKMAJORITYTRAINER_H_
#define BOWKMAJORITYTRAINER_H_

#include <opencv2/features2d.hpp>

namespace cv {

class BOWKmajorityTrainer: public BOWTrainer {

public:
    BOWKmajorityTrainer(int clusterCount, const TermCriteria& termcrit=TermCriteria());
    virtual ~BOWKmajorityTrainer();
    virtual Mat cluster() const;
    virtual Mat cluster(const Mat& descriptors) const;

protected:
    int numClusters;
    int maxIterations;
};

} /* namespace cv */
#endif /* BOWKMAJORITYTRAINER_H_ */
