//
//  Tools.h
//  DecisionTree
//
//  Created by J Zhou on 2017/11/24.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef Tools_h
#define Tools_h
#include <limits>
#include <vector>
#include <string>
using namespace std;
using Samples_ID = vector<int>;

float search_max(vector<vector<float> >& X, Samples_ID& samples_id, int dimension){
    float best = -numeric_limits<float>::max();
    for (auto& sample_id : samples_id){
        if (X[sample_id][dimension] > best)
            best = X[sample_id][dimension];
    }
    return best;
}

float search_min(vector<vector<float> >& X, Samples_ID& samples_id, int dimension){
    float best = numeric_limits<float>::max();
    for (auto& sample_id : samples_id){
        if (X[sample_id][dimension] < best)
            best = X[sample_id][dimension];
    }
    return best;
}

//bool is_in(int i, samples_id ptr){
//    for (auto& elem : *ptr){
//        if (i == elem)
//            return true;
//    }
//    return false;
//}

struct BestContainer{
    string feature_name;
    float split = 0;
    int feature_id = 0;
    float se = numeric_limits<float>::max();
    float se_left = 0, se_right = 0;
    float left_mse = 0, right_mse = 0;
    float left_y_hat = 0, right_y_hat = 0;
    vector<int> left_samples_id, right_samples_id;
    
//    BestContainer(BestContainer&);
    BestContainer(){};
};

//struct QuantileEstimator{
//    QuantileEstimator(const vector<float>& samples);
//    QuantileEstimator(const vector<float>& samples, const vector<float>& sample_weight);
//    float estimate(float alpha);
//    vector<float> sorted_samples;
////    int group_num;
////    vector<vector<float>> group;
////    vector<float> weights;
//    
//};
//
//QuantileEstimator::QuantileEstimator(const vector<float>& samples){
//    auto sample2do = samples;
//    sort(sample2do.begin(), sample2do.end());
//    sorted_samples = sample2do;
//}
//
//float QuantileEstimator::estimate(float alpha){
////    if ((alpha > 1) || (alpha < 0))
////        throw expression;
//    unsigned long index = sorted_samples.size() * alpha;
//    return sorted_samples[index];
//}

float quantile_estimate(const vector<float>& samples, float alpha){
    auto sample2do = samples;
    sort(sample2do.begin(), sample2do.end());
    unsigned long index = sample2do.size() * alpha;
    return sample2do[index];
}
#endif /* Tools_h */
