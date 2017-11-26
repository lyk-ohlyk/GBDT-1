//
//  GradientBoosting.h
//  GBDT
//
//  Created by J Zhou on 2017/11/23.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef GradientBoosting_h
#define GradientBoosting_h

#include <stdio.h>
#include "RegressionTree.h"
#include "SquareLoss.h"
#include <vector>
#include "DMatrix.h"
using namespace std;
class GradientBoosting{
public:
    GradientBoosting(){loss = new SquareLoss;};
    ~GradientBoosting(){delete loss;};
    void train(const DMatrix&);
    void predict();
    vector<float> staged_predict(int iter_round, const vector<vector<float>>& X);
    float score();
//    const vector<float>& y;
protected:
    vector<float> init_estimator;
    vector<RegressionTree*> trees;
    RegressionTree* curr_tree;
    
    string loss_type = "ls";
    //    options : "ls", "lad", "huber"
    LossFunction *loss;
    float learning_rate = 1;
    int iteration_rounds = 30;
    int max_depth = 3;
    int min_samples_split = 20;
    int min_samples_leaf = 10;
    float min_weight_fraction_leaf = 0;
    float subsample = 1;
    float colsample = 1;
    float alpha = 0.9;
};

void GradientBoosting::train(const DMatrix& train_data){
    auto X = train_data.X();
    auto y_true = train_data.y();
    init_estimator.assign(y_true.size(), loss->init_state(y_true));
    
    auto y_pred = init_estimator;
    for (int i = 1; i <= iteration_rounds; i++){
        curr_tree = new RegressionTree;
        auto pesudo_residual = loss->negative_gradient(y_true, y_pred);
//        curr_tree->train(X, pesudo_residual);
        trees.push_back(curr_tree);
        loss->update_leaf_nodes(curr_tree, y_true, y_pred);
//        y_pred = staged_predict(i);
    }
}

vector<float> GradientBoosting::staged_predict(int iter_round, const vector<vector<float>>& X){
//    trees.at(iteration_rounds-1);
}
#endif /* GradientBoosting_h */
