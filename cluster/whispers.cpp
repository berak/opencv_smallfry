#include "opencv2/opencv.hpp"
#include "profile.h"
#include "cluster.h"

namespace whispers {

	typedef pair<int,int> int_pair;

	struct sample_pair {
		int first,second;
		double dist;
		sample_pair() {}
		sample_pair(int a, int b, double d=1) : first(a), second(b), dist(d) {}
	};

    int max_index_plus_one (const vector<sample_pair>& pairs)
    {
        if (pairs.size() == 0)
            return 0;
        int max_idx = 0;
        for (unsigned long i = 0; i < pairs.size(); ++i)
        {
            if (pairs[i].first > max_idx)
                max_idx = pairs[i].first;
            if (pairs[i].second > max_idx)
                max_idx = pairs[i].second;
        }
        return max_idx + 1;
    }



    void find_neighbor_ranges (
        const std::vector<sample_pair>& edges,
        std::vector<int_pair>& neighbors
    )
    {
        // setup neighbors so that [neighbors[i].first, neighbors[i].second) is the range
        // within edges that contains all node i's edges.
        const int num_nodes = max_index_plus_one(edges);
        neighbors.assign(num_nodes, std::make_pair(0,0));
        int cur_node = 0;
        int start_idx = 0;
        for (int i = 0; i < edges.size(); ++i)
        {
            if (edges[i].first != cur_node)
            {
                neighbors[cur_node] = std::make_pair(start_idx, i);
                start_idx = i;
                cur_node = edges[i].first;
            }
        }
        if (neighbors.size() != 0)
            neighbors[cur_node] = std::make_pair(start_idx, (int)edges.size());
    }

    inline bool order_by_index (const sample_pair& a, const sample_pair& b)
    {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    }

    void convert_unordered_to_ordered (
        const std::vector<sample_pair>& edges,
        std::vector<sample_pair>& out_edges
    )
    {
        out_edges.clear();
        out_edges.reserve(edges.size()*2);
        for (unsigned long i = 0; i < edges.size(); ++i)
        {
            out_edges.push_back(sample_pair(edges[i].first, edges[i].second, edges[i].dist));
            if (edges[i].first != edges[i].second)
                out_edges.push_back(sample_pair(edges[i].second, edges[i].first, edges[i].dist));
        }
    }




    inline int chinese_whispers (
        const std::vector<sample_pair>& in_edges,
        std::vector<int>& labels,
        const int num_iterations
    )
    {
        std::vector<sample_pair> edges;
        convert_unordered_to_ordered(in_edges, edges);
        std::sort(edges.begin(), edges.end(), &order_by_index);

        labels.clear();
        if (edges.size() == 0)
            return 0;

        std::vector<std::pair<int, int> > neighbors;
        find_neighbor_ranges(edges, neighbors);

        // Initialize the labels, each node gets a different label.
        labels.resize(neighbors.size());
        for (int i = 0; i < labels.size(); ++i)
            labels[i] = i;


        for (int iter = 0; iter < neighbors.size()*num_iterations; ++iter)
        {
            // Pick a random node.
            const int idx = theRNG().uniform(0,neighbors.size());

            // Count how many times each label happens amongst our neighbors.
            std::map<int, double> labels_to_counts;
            const int end = neighbors[idx].second;
            for (int i = neighbors[idx].first; i != end; ++i)
            {
                labels_to_counts[labels[edges[i].second]] += edges[i].dist;
            }

            // find the most common label
            std::map<int, double>::iterator i;
            double best_score = -std::numeric_limits<double>::infinity();
            int best_label = labels[idx];
            for (i = labels_to_counts.begin(); i != labels_to_counts.end(); ++i)
            {
                if (i->second > best_score)
                {
                    best_score = i->second;
                    best_label = i->first;
                }
            }

            labels[idx] = best_label;
        }


        // Remap the labels into a contiguous range.  First we find the
        // mapping.
        std::map<int,int> label_remap;
        for (int i = 0; i < labels.size(); ++i)
        {
            const int next_id = label_remap.size();
            if (label_remap.count(labels[i]) == 0)
                label_remap[labels[i]] = next_id;
        }
        // now apply the mapping to all the labels.
        for (int i = 0; i < labels.size(); ++i)
        {
            labels[i] = label_remap[labels[i]];
        }

        return label_remap.size();
    }



	int cluster(const vector<Mat> &features, vector<int> &labels, double eps)
	{
		PROFILE;
		Mat indices;
	    std::vector<sample_pair> edges;
    	for (int i = 0; i < features.size(); ++i)
    	{
	        for (int j = i+1; j < features.size(); ++j)
	        {
	        	double v = norm(features[i] - features[j]);
	        	//cout << i << " " <<j << " "<< v << endl;
	            if (v < eps)
	                edges.push_back(sample_pair(i,j));
	        }
    	}
       	return chinese_whispers(edges,labels,100);
	}
}
