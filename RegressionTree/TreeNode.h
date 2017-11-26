//
//  TreeNode.h
//  DecisionTree
//
//  Created by J Zhou on 2017/11/24.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef TreeNode_h
#define TreeNode_h

#include <stdio.h>
#include <string>
#include <iostream>
#include "DMatrix.h"
#include "Tools.h"

class TreeNode{
public:
    TreeNode(){};
//    TreeNode(Samples_ID);
    TreeNode(const string&, int, float, const Samples_ID&);
    TreeNode(const string&, int, float, const Samples_ID&, TreeNode&, TreeNode&);
    ~TreeNode(){delete this;}
    
    TreeNode* get_left();
    TreeNode* get_right();
    int get_depth() const;
    int get_feature_id() const;
    float get_split_val() const;
    float get_y_hat() const;
    bool is_leaf_or_not() const;
    const Samples_ID& get_samples() const;
    unsigned long get_sample_num() const;
    
    void reset_feature(string&, int);
    void reset_depth(int);
    void reset_data(float);
    void reset_samples(const Samples_ID&);
    void reset_left(TreeNode*);
    void reset_right(TreeNode*);
    void reset_y_hat(float);
    void reset_mse(float);
    void set_leaf();
    
    void print_node();
private:
    string feature_name;
    int feature_id = -1;
    float split_val = 0;
    int depth = 0;
    bool is_leaf = false;
    Samples_ID samples_id;
    TreeNode *left = nullptr;
    TreeNode *right = nullptr;
    
    // meaningful only in leaf nodes
    float y_hat = 0;
    float mse = 0;
    unsigned long sample_num = 0;
};

TreeNode::TreeNode(const string& name, int feature, float data, const Samples_ID& ids){
    feature_name = name;
    feature_id = feature;
    split_val = data;
    samples_id = ids;
    sample_num = ids.size();
}

TreeNode::TreeNode(const string& name, int feature, float data,  const Samples_ID& ids, TreeNode& l, TreeNode& r){
    feature_name = name;
    feature_id = feature;
    split_val = data;
    samples_id = ids;
    sample_num = ids.size();
    left = &l;
    right = &r;
}

TreeNode* TreeNode::get_left(){
    return left;
}
TreeNode* TreeNode::get_right(){
    return right;
}
int TreeNode::get_depth() const{
    return depth;
}
int TreeNode::get_feature_id() const{
    return feature_id;
}
float TreeNode::get_split_val() const{
    return split_val;
}
float TreeNode::get_y_hat() const{
    return y_hat;
}
const Samples_ID& TreeNode::get_samples() const{
    return samples_id;
}

unsigned long TreeNode::get_sample_num() const{
    return sample_num;
}

bool TreeNode::is_leaf_or_not() const{
    return is_leaf;
}

void TreeNode::reset_depth(int d){
    depth = d;
}

void TreeNode::reset_feature(string& new_name, int new_id){
    feature_name = new_name;
    feature_id  = new_id;
}

void TreeNode::reset_data(float data){
    split_val = data;
}

void TreeNode::reset_samples(const Samples_ID& ids){
    samples_id = ids;
    sample_num = ids.size();
}

void TreeNode::reset_left(TreeNode* new_left){
    left = new_left;
}

void TreeNode::reset_right(TreeNode* new_right){
    right = new_right;
}

void TreeNode::reset_mse(float mse){
    this->mse = mse;
}

void TreeNode::reset_y_hat(float y){
    y_hat = y;
}

void TreeNode::set_leaf(){
    is_leaf = true;
    left = nullptr;
    right = nullptr;
}

void TreeNode::print_node(){
    if (is_leaf){
        cout << "( terminal node"
        << "    depth:" <<  depth
        << ", y_hat:" << y_hat
        << ", mse:" << mse
        << ", nodes_num:" << sample_num << ")";
    }
    else{
        cout << "( feature_name:" << feature_name
        << ", split_val:" << split_val
        << ", depth:" <<  depth << ")";
    }
}
#endif /* TreeNode_h */
