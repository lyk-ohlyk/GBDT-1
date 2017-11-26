//
//  RegressionTree.h
//  DecisionTree
//
//  Created by J Zhou on 2017/11/24.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef RegressionTree_h
#define RegressionTree_h

#include <stdio.h>
#include <iostream>
#include "TreeNode.h"
#include "DMatrix.h"
#include <limits>
#include "Tools.h"


class RegressionTree{
public:
    RegressionTree(){};
//    RegressionTree(int max_depth, int min_samples_split, int min_samples_leaf);
    void train(DMatrix&);
    DMatrix predict(DMatrix&) const;
    float score(DMatrix&) const;
    void print_tree() const;
    vector<TreeNode*> get_leaf_node() const;
private:
    TreeNode* root = new TreeNode;
    void split(DMatrix&, TreeNode*, TreeNode*, TreeNode*);
    void compute_se(float, int, DMatrix&, TreeNode*, BestContainer&);
    void rec_fit(DMatrix&, TreeNode*);
    
    int depth = 0;
    int max_depth = 4;
    int min_samples_split = 20;
    int min_samples_leaf = 10;
    float min_weight_fraction_leaf = 0;
};

void RegressionTree::print_tree() const{
    std::vector<TreeNode*> st;
    TreeNode* p = root;
    st.push_back(p);
    
    while (!st.empty()){
        p = st.back();
        st.pop_back();
        for (int i=0; i!=p->get_depth(); i++)
            std::cout << "      ";
        p->print_node();
        std::cout << std::endl;
        if ((p->get_left()) != 0)
            st.push_back(p->get_left());
        if ((p->get_right()) != 0)
            st.push_back(p->get_right());
    }
}

void RegressionTree::train(DMatrix& train_data){
    Samples_ID samples_id;
    for (int i=0; i != train_data.sample_num(); i++)
        samples_id.push_back(i);
    root->reset_samples(samples_id);
    rec_fit(train_data, root);
}

void RegressionTree::compute_se(float split_val, int dimension, DMatrix& train_data,
                                   TreeNode* parent, BestContainer& temp)
{
    auto X = train_data.X();
    auto y = train_data.y();
    auto samples_id = parent->get_samples();
    
    float sum_right = 0, sum_left = 0, square_error_right = 0 , square_error_left = 0;
    float avg_right, avg_left;
    
    auto it = samples_id.begin();
    auto end = samples_id.end();
    int sample_id;
    
    std::vector<int> left_samples, right_samples;
    
    while (it != end){
        sample_id = *it++;
        if (X[sample_id][dimension] < split_val){
            sum_left += y[sample_id];
            left_samples.push_back(sample_id);
        }
        else {
            sum_right += y[sample_id];
            right_samples.push_back(sample_id);
        }
    }
    //    循环计算各个子节点的均值
    avg_left = sum_left / left_samples.size();
    avg_right = sum_right / right_samples.size();
    
    it = samples_id.begin();
    while (it != end){
        sample_id = *it++;
        if (X[sample_id][dimension] < split_val)
            square_error_left += (y[sample_id] - avg_left) * (y[sample_id] - avg_left);
        else
            square_error_right += (y[sample_id] - avg_right) * (y[sample_id] - avg_right);
    }
    //    循环计算各个子节点的平方误差
    temp.se_left = square_error_left;
    temp.se_right = square_error_right;
    
    temp.left_samples_id = left_samples;
    temp.right_samples_id = right_samples;
    
    temp.left_mse = square_error_left / left_samples.size();
    temp.right_mse = square_error_right / right_samples.size();
    
    temp.left_y_hat = avg_left;
    temp.right_y_hat = avg_right;
    
    temp.se = square_error_left + square_error_right;
    
}

void RegressionTree::split(DMatrix& train_data, TreeNode* parent, TreeNode* left, TreeNode* right){
    auto X = train_data.X();
    auto y = train_data.y();
    auto samples_id = parent->get_samples();
    
    auto feature_num = train_data.feature_num();
    auto sample_num = train_data.sample_num();
    
    BestContainer temp, best;
    
    for (int i=0; i!= feature_num; i++){
        temp.feature_id = i;
        temp.feature_name = train_data.feature_name(i);
        auto i_max = search_max(X, samples_id, i);
        auto i_min = search_min(X, samples_id, i);
        auto step = (i_max - i_min) / sample_num;
        for (auto split_val = i_min - 0.5 * step ; split_val < i_max; split_val += step){
            temp.split = split_val;
            compute_se(split_val, i, train_data, parent, temp);
            if (temp.se < best.se)
                best = temp;
        }
    }
    
    left->reset_samples(best.left_samples_id);
    left->reset_mse(best.left_mse);
    left->reset_y_hat(best.left_y_hat);
    right->reset_samples(best.right_samples_id);
    right->reset_mse(best.right_mse);
    right->reset_y_hat(best.right_y_hat);
    
    parent->reset_feature(best.feature_name, best.feature_id);
    parent->reset_data(best.split);
    parent->reset_left(left);
    parent->reset_right(right);
}

void RegressionTree::rec_fit(DMatrix& train_data, TreeNode* current_node){
    auto cur_depth = current_node->get_depth();
    if (cur_depth == max_depth or current_node->get_sample_num() < min_samples_split){
        current_node->set_leaf();
        return;
    }
    
    TreeNode* left_child = new TreeNode;
    TreeNode* right_child = new TreeNode;
    
    left_child->reset_depth(cur_depth+1);
    right_child->reset_depth(cur_depth+1);
    
    split(train_data, current_node, left_child, right_child);
    
    if (left_child->get_sample_num() < min_samples_leaf or right_child->get_sample_num() < min_samples_leaf){
        current_node->set_leaf();
        return;
    }
    rec_fit(train_data, left_child);
    rec_fit(train_data, right_child);
}

DMatrix RegressionTree::predict(DMatrix& test_data) const{
    auto X = test_data.X();
    std::vector<float> y_hat;
    
    auto samples_id = root->get_samples();
    TreeNode* ptr = nullptr;
    
    auto it = samples_id.begin();
    auto end = samples_id.end();
    
    while (it != end){
        ptr = root;
        while ( !ptr->is_leaf_or_not() ){
            if (X[*it][ptr->get_feature_id()] < ptr->get_split_val()){
                ptr = ptr->get_left();
            }
            else{
                ptr = ptr->get_right();
            }
        }
        y_hat.push_back(ptr->get_y_hat());
        it++;
    }
    return DMatrix(X, y_hat);
}

vector<TreeNode*> RegressionTree::get_leaf_node() const{
    vector<TreeNode*> node_vec, st;
    TreeNode* ptr = root;
    st.push_back(ptr);
    while (!st.empty()){
        ptr = st.back();
        st.pop_back();
        if (ptr->is_leaf_or_not())
            node_vec.push_back(ptr);
        st.push_back(ptr->get_right());
        st.push_back(ptr->get_left());
    }
    return node_vec;
}
#endif /* RegressionTree_h */
