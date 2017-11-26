//
//  HuberLoss.h
//  GBDT
//
//  Created by J Zhou on 2017/11/25.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef HuberLoss_h
#define HuberLoss_h
#include "LossFunction.h"
#include <math.h>
class HuberLoss: public LossFunction{
public:
    float init_state(const vector<float>&) override;
    vector<float> negative_gradient(const vector<float>& y_true, const vector<float>& y_pred, float alpha = 0) override;
    void update_leaf_nodes(RegressionTree&,  const vector<float>& y_true, const vector<float>& y_pred);
private:
    void update_leaf_node(TreeNode*,  const vector<float>& y_true, const vector<float>& y_pred, float delta) ;
};

float HuberLoss::init_state(const vector<float>& y){
    auto temp = y;
    nth_element(temp.begin(), temp.begin() + temp.size()/2, temp.end());
    return temp[temp.size()/2];
}

vector<float> HuberLoss::negative_gradient(const vector<float>& y_true, const vector<float>& y_pred, float alpha){
    vector<float> abs_residual, negative_gradient;
    int i = 0;
    while( i != y_true.size()){
        abs_residual.push_back(abs(y_true[i] - y_pred[i]));
        i++;
    }
    auto delta = quantile_estimate(abs_residual, alpha);
    
    i = 0;
    int sign;
    while( i != y_true.size()){
        if (abs_residual[i] <= delta)
            negative_gradient.push_back(y_true[i] - y_pred[i]);
        else{
            sign = ((y_true[i] - y_pred[i]) > 0) ? 1 : -1;
            negative_gradient.push_back(sign * delta);
        }
        i++;
    }
    return negative_gradient;
}

void HuberLoss::update_leaf_node(TreeNode* node,  const vector<float>& y_true, const vector<float>& y_pred, float delta){
    vector<float> temp;
    for (auto i : node->get_samples())
        temp.push_back(y_true[i] - y_pred[i]);
    nth_element(temp.begin(), temp.begin() + temp.size() / 2 , temp.end());
    auto r = temp[temp.size() / 2];
    float temp_sum = 0.0;
    for (auto i : node->get_samples())
        temp_sum += (y_true[i] - y_pred[i] - r) / abs(y_true[i] - y_pred[i] - r) * min({delta, abs(y_true[i] - y_pred[i] - r)});
    node->reset_y_hat(r + temp_sum / node->get_samples().size());
}
#endif /* HuberLoss_h */
