//
//  main.cpp
//  DecisionTree
//
//  Created by J Zhou on 2017/11/24.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#include <iostream>
//#include "DMatrix.h"
//#include "TreeNode.h"
//#include "RegressionTree.h"
#include <string>
#include "GradientBoostingRegressor.h"
#include "GradientBoosting.h"
#include "LossFunction.h"
#include "AbsoluteDeviation.h"
#include "SquareLoss.h"
#include "HuberLoss.h"
int main(int argc, const char * argv[]) {
    DMatrix x;
//    const DMatrix& xx = x;
    RegressionTree T;
    T.train(x);
    T.print_tree();
    auto y = T.predict(x).y();
    for (auto &elem : y)
        std::cout << elem << std::endl;
    return 0;
    
}
