//
//  SquareLoss.h
//  GBDT
//
//  Created by J Zhou on 2017/11/25.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef SquareLoss_h
#define SquareLoss_h
#include "LossFunction.h"

class SquareLoss: public LossFunction{
public:
    float init_state(const vector<float>&) override;
    vector<float> negative_gradient(const vector<float>& y_true, const vector<float>& y_pred, float alpha = 0) override;
private:
    void update_leaf_node(TreeNode*, const vector<float>&, const vector<float>&) override;
};

float SquareLoss::init_state(const vector<float>& y){
    float sum = 0;
    for (auto &y_value : y)
        sum += y_value;
    return sum / y.size();
}

vector<float> SquareLoss::negative_gradient(const vector<float>& y_true, const vector<float>& y_pred,float){
    vector<float> negative_gradient;
    int i = 0;
    while( i != y_true.size()){
        negative_gradient.push_back(y_true[i] - y_pred[i]);  // LS-negative gradient equals the true residual
        i++;
    }
    return negative_gradient;
}

void SquareLoss::update_leaf_node(TreeNode* node, const vector<float>& y_true, const vector<float>& y_pred){
//    we don't need to update leaf node when using LeastSquareError
    return;
}
#endif /* SquareLoss_h */
