//
//  AbsoluteDeviation.h
//  GBDT
//
//  Created by J Zhou on 2017/11/25.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef AbsoluteDeviation_h
#define AbsoluteDeviation_h
#include "LossFunction.h"

class AbsoluteDeviation: public LossFunction{
public:
    float init_state(const vector<float>&) override;
    vector<float> negative_gradient(const vector<float>& y_true, const vector<float>& y_pred, float alpha=0) override;
private:
    void update_leaf_node(TreeNode*, const vector<float>&, const vector<float>&) override;
};

float AbsoluteDeviation::init_state(const vector<float>& y){
    auto temp = y;
    nth_element(temp.begin(), temp.begin() + temp.size()/2, temp.end());
    return temp[temp.size()/2];
//    initial guess is the median of y
}

vector<float> AbsoluteDeviation::negative_gradient(const vector<float>& y_true, const vector<float>& y_pred, float){
    vector<float> negative_gradient;
    int temp;
    int i = 0;
    while( i != y_true.size()){
        temp = ((y_true[i] - y_pred[i]) > 0) ? 1 : -1;
        negative_gradient.push_back(temp);
        i++;
    }
    return negative_gradient;
}

void AbsoluteDeviation::update_leaf_node(TreeNode* node, const vector<float>& y_true, const vector<float>& y_pred) {
    vector<float> temp;
    for (auto i : node->get_samples())
        temp.push_back(y_true[i] - y_pred[i]);
    nth_element(temp.begin(), temp.begin() + temp.size() / 2 , temp.end());
    node->reset_y_hat(temp[temp.size() / 2]);
}

#endif /* AbsoluteDeviation_h */
