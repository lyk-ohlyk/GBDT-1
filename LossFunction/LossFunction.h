//
//  LossFunction.h
//  GBDT
//  the Abstract Class of LossFunction
//  Created by J Zhou on 2017/11/23.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef LossFunction_h
#define LossFunction_h

#include <stdio.h>
#include "DMatrix.h"
#include "RegressionTree.h"
#include "TreeNode.h"
class LossFunction{
//    the abstract class
public:
    virtual float init_state(const vector<float>&) = 0;
//    initialize the Boosting Model, return a constant
    virtual vector<float> negative_gradient(const vector<float>& y_true, const vector<float>& y_pred, float alpha=0) = 0;
    void update_leaf_nodes(RegressionTree*,  const vector<float>& y_true, const vector<float>& y_pred);
//    upadte the weight of terminal nodes in a tree
protected:
    virtual void update_leaf_node(TreeNode*,  const vector<float>& y_true, const vector<float>& y_pred) = 0;
};

void LossFunction::update_leaf_nodes(RegressionTree *tree, const vector<float>& y_true, const vector<float>& y_pred){
    auto leaf_nodes = tree->get_leaf_node();
    for (auto &terminal_node : leaf_nodes){
        update_leaf_node(terminal_node, y_true, y_pred);
    }
}

#endif /* LossFunction_h */
