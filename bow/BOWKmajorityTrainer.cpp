//
// based on code taken from:
//    https://github.com/gantzer89/VLRPipeline
//

/*
 * BOWKmajorityTrainer.cpp
 *
 *  Created on: Sep 26, 2013
 *      Author: andresf
 */



#include "BOWKmajorityTrainer.h"
#include <opencv2/flann/linear_index.h>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>

typedef cvflann::Hamming<uchar> HammingDistance;
typedef cvflann::LinearIndex<HammingDistance> HammingIndex;


namespace cv {

struct KMajority 
{
    /**
     * Initializes cluster centers choosing among the data points indicated by indices.
     */
    static cv::Mat initCentroids(const cv::Mat &trainData, int numClusters) 
    {
        // Initializing variables useful for obtaining indexes of random chosen center
        std::vector<int> centers_idx(numClusters);
        randu(centers_idx, Scalar(0), Scalar(numClusters));
        //randu(centers_idx, Scalar(0), Scalar(trainData.rows));
        std::sort(centers_idx.begin(), centers_idx.end());

        // Assign centers based on the chosen indexes
        cv::Mat centroids(centers_idx.size(), trainData.cols, trainData.type());
        for (int i = 0; i < numClusters; ++i) 
        {
            trainData.row(centers_idx[i]).copyTo(centroids(cv::Range(i, i + 1), cv::Range(0, trainData.cols)));
        }
        return centroids;
    }

    /**
     * Implements majority voting scheme for cluster centers computation
     * based on component wise majority of bits from data matrix
     * as proposed by Grana2013.
     */
    static void computeCentroids(const Mat &features, Mat &centroids,
        std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo) 
    {
        // Warning: using matrix of integers, there might be an overflow when summing too much descriptors
        cv::Mat bitwiseCount(centroids.rows, features.cols * 8, CV_32S);
        // Zeroing matrix of cumulative bits
        bitwiseCount = cv::Scalar::all(0);
        // Zeroing all cluster centers dimensions
        centroids = cv::Scalar::all(0);

        // Bitwise summing the data into each center
        for (int i=0; i<features.cols; ++i) 
        {
            cv::Mat b = bitwiseCount.row(belongsTo[i]);
            KMajority::cumBitSum(features.row(i), b);
        }

        // Bitwise majority voting
        for (int j=0; j<centroids.rows; j++) 
        {
            cv::Mat centroid = centroids.row(j);
            KMajority::majorityVoting(bitwiseCount.row(j), centroid, clusterCounts[j]);
        }
    }
    /**
     * Decomposes data into bits and accumulates them into cumResult.
     *
     * @param feature- Row vector containing the data to accumulate
     * @param accVector - Row oriented accumulator vector
     */
    static void cumBitSum(const cv::Mat& feature, cv::Mat& accVector) 
    {
        // cumResult and data must be row vectors
        CV_Assert(feature.rows == 1 && accVector.rows == 1);
        // cumResult and data must be same length
        CV_Assert(feature.cols * 8 == accVector.cols);

        uchar byte = 0;
        for (int l = 0; l < accVector.cols; l++) 
        {
            // bit: 7-(l%8) col: (int)l/8 descriptor: i
            // Load byte every 8 bits
            if ((l % 8) == 0) 
            {
                byte = *(feature.col((int) l / 8).data);
            }
            // Note: ignore maybe-uninitialized warning because loop starts with l=0 that means byte gets a value as soon as the loop start
            // bit at ith position is mod(bitleftshift(byte,i),2) where ith position is 7-mod(l,8) i.e 7, 6, 5, 4, 3, 2, 1, 0
            accVector.at<int>(0, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
        }
    }
   
    static void majorityVoting(const cv::Mat& accVector, cv::Mat& result, const int& threshold) 
    {
        // cumResult and data must be row vectors
        CV_Assert(result.rows == 1 && accVector.rows == 1);
        // cumResult and data must be same length
        CV_Assert(result.cols * 8 == accVector.cols);

        // In this point I already have stored in bitwiseCount the bitwise sum of all data assigned to jth cluster
        for (int l = 0; l < accVector.cols; ++l) 
        {
            // If the bitcount for jth cluster at dimension l is greater than half of the data assigned to it
            // then set lth centroid bit to 1 otherwise set it to 0 (break ties randomly)
            bool bit;
            // There is a tie if the number of data assigned to jth cluster is even
            // AND the number of bits set to 1 in lth dimension is the half of the data assigned to jth cluster
            if ((threshold % 2 == 1) && (2 * accVector.at<int>(0, l) == (int) threshold))
            {
                bit = (bool)(rand() % 2);
            } 
            else 
            {
                bit = 2 * accVector.at<int>(0, l) > (int) (threshold);
            }
            // Stores the majority voting result from the LSB to the MSB
            result.at<unsigned char>(0, (int) (accVector.cols - 1 - l) / 8) += (bit) << ((accVector.cols - 1 - l) % 8);
        }
    }
    
    /**
     * Assigns data to clusters by means of Hamming distance.
     *
     * @return true if convergence was achieved (cluster assignment didn't changed), false otherwise
     */
    static bool quantize(cv::Ptr<HammingIndex> index, const Mat &descriptors,
        std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo, int numClusters) 
    {
        bool converged = true;

        // Number of nearest neighbors
        int knn = 1;

        // The indices of the nearest neighbors found (numQueries X numNeighbors)
        cvflann::Matrix<int> indices(new int[1 * knn], 1, knn);

        // Distances to the nearest neighbors found (numQueries X numNeighbors)
        cvflann::Matrix<int> distances(new int[1 * knn], 1, knn);

        for (int i=0; i<descriptors.rows; ++i) 
        {
            std::fill(indices.data, indices.data + indices.rows * indices.cols, 0);
            std::fill(distances.data, distances.data + distances.rows * distances.cols, 0);

            cvflann::Matrix<uchar> descriptor(descriptors.row(i).data, 1, descriptors.cols);

            /* Get new cluster it belongs to */
            index->knnSearch(descriptor, indices, distances, knn, cvflann::SearchParams());

            /* Check if cluster assignment changed */
            // If it did then algorithm hasn't converged yet
            if (belongsTo[i] != indices[0][0]) {
                converged = false;
            }

            /* Update cluster assignment and cluster counts */
            // Decrease cluster count in case it was assigned to some valid cluster before.
            // Recall that initially all transaction are assigned to kth cluster which
            // is not valid since valid clusters run from 0 to k-1 both inclusive.
            if (belongsTo[i] != numClusters) {
                --clusterCounts[belongsTo[i]];
            }
            belongsTo[i] = indices[0][0];
            ++clusterCounts[indices[0][0]];
            distanceTo[i] = distances[0][0];
        }

        delete[] indices.data;
        delete[] distances.data;

        return converged;
    }

    /**
     * Fills empty clusters using data assigned to the most populated ones.
     */
    static void handleEmptyClusters(std::vector<int> &belongsTo, std::vector<int> &clusterCounts, std::vector<int> &distanceTo, int numClusters, int numDatapoints)
    {
        // If some cluster appeared to be empty then:
        // 1. Find the biggest cluster.
        // 2. Find farthest point in the biggest cluster
        // 3. Exclude the farthest point from the biggest cluster and form a new 1-point cluster.

        for (int k = 0; k < numClusters; ++k) {
            if (clusterCounts[k] != 0) {
                continue;
            }

            // 1. Find the biggest cluster
            int max_k = 0;
            for (int k1 = 1; k1 < numClusters; ++k1) {
                if (clusterCounts[max_k] < clusterCounts[k1])
                    max_k = k1;
            }

            // 2. Find farthest point in the biggest cluster
            int maxDist(-1);
            int idxFarthestPt = -1;
            for (int i = 0; i < numDatapoints; ++i) {
                if (belongsTo[i] == max_k) {
                    if (maxDist < distanceTo[i]) {
                        maxDist = distanceTo[i];
                        idxFarthestPt = i;
                    }
                }
            }

            // 3. Exclude the farthest point from the biggest cluster and form a new 1-point cluster
            --clusterCounts[max_k];
            ++clusterCounts[k];
            belongsTo[idxFarthestPt] = k;
        }
    }
};


BOWKmajorityTrainer::BOWKmajorityTrainer(int clusterCount, const TermCriteria &tc) 
    : numClusters(clusterCount)
    , maxIterations(tc.maxCount)
{
}

BOWKmajorityTrainer::~BOWKmajorityTrainer() 
{
}

Mat BOWKmajorityTrainer::cluster() const 
{
    CV_Assert(descriptors.empty() == false);

    // Compute number of rows of matrix containing all training descriptors,
    // that is matrix resulting from the concatenation of the images descriptors
    int descriptorsCount = 0;
    for (size_t i = 0; i < descriptors.size(); i++) 
    {
        descriptorsCount += descriptors[i].rows;
    }

    // Concatenating the images descriptors into a single big matrix
    Mat trainingDescriptors(descriptorsCount, descriptors[0].cols, descriptors[0].type());

    for (size_t i = 0, start = 0; i < descriptors.size(); i++) 
    {
        Mat submut = trainingDescriptors.rowRange((int) start, (int) (start + descriptors[i].rows));
        descriptors[i].copyTo(submut);
        start += descriptors[i].rows;
    }

    return cluster(trainingDescriptors);
}

Mat BOWKmajorityTrainer::cluster(const Mat& descriptors) const 
{
    // Trivial case: less data than clusters, assign one data point per cluster
    if (descriptors.rows <= numClusters) 
    {
        Mat centroids;
        for (int i=0; i<numClusters; i++)
            centroids.push_back(descriptors.row(i % numClusters));
        return centroids;
    }

    Mat centroids = KMajority::initCentroids(descriptors, numClusters);

    cvflann::Matrix<uchar> inputData(centroids.data, centroids.rows, centroids.cols);
    cvflann::IndexParams params = cvflann::LinearIndexParams();
    cv::Ptr<HammingIndex> index = makePtr<HammingIndex>(inputData, params, HammingDistance());

    // Initially all transactions belong to any cluster
    std::vector<int> belongsTo(descriptors.rows, numClusters);
    // List of distance from each data point to the cluster it belongs to
    //  Initially all transactions are at the farthest possible distance
    //  i.e. m_dim*8 the max Hamming distance
    std::vector<int> distanceTo(descriptors.rows, descriptors.cols * 8);
    // Number of data points assigned to each cluster
    //  Initially no transaction is assigned to any cluster
    std::vector<int> clusterCounts(numClusters, 0);
    KMajority::quantize(index, descriptors, belongsTo, clusterCounts, distanceTo, numClusters);

    for (int iteration=0; iteration<maxIterations; ++iteration)
    {
        KMajority::computeCentroids(descriptors,centroids,belongsTo,clusterCounts,distanceTo);

        index = makePtr<HammingIndex>(inputData, params, HammingDistance());

        bool converged = KMajority::quantize(index, descriptors, belongsTo, clusterCounts, distanceTo, numClusters);
        KMajority::handleEmptyClusters(belongsTo, clusterCounts, distanceTo, numClusters, descriptors.rows);

        if (converged)
            break;
    }    
    
    return centroids;
}

} /* namespace cv */
