//
//  DMatrix.h
//  DecisionTree
//
//  Created by J Zhou on 2017/11/24.
//  Copyright © 2017年 J Zhou. All rights reserved.
//

#ifndef DMatrix_h
#define DMatrix_h
#include <vector>

class DMatrix{
public:
    DMatrix();
    DMatrix(const std::vector<std::vector<float>>&, const std::vector<float>&);
    std::vector<std::vector<float>> X() const;
    std::vector<float> y() const;
    std::string feature_name(int) const;
    int feature_num() const;
    unsigned long sample_num() const;
private:
    std::vector<std::vector<float> > samples;
    std::vector<float> label;
    std::vector<std::string> feature_names;
    int feature_num_ = 0;
    unsigned long sample_num_ = 0;
    
};

DMatrix::DMatrix(){
    std::vector<float> temp;
    for (int i=0;i!=200;i++){
        temp= {static_cast<float>((-12 + rand() % 25) / 2.5), static_cast<float>((-12 + rand() % 25) / 2.5)};
        samples.push_back(temp);
        label.push_back(temp[0]*temp[0] + temp[1]*temp[1] + (-12 + rand() % 25) / 10.0);
    }
    feature_num_ = 2;
    sample_num_ = 200;
    feature_names = {"x1","x2"};
}

DMatrix::DMatrix(const std::vector<std::vector<float>>& X, const std::vector<float>& y){
    samples = X;
    label = y;
    sample_num_ = X.size();
}
std::vector<std::vector<float> > DMatrix::X() const{
    return samples;
}

std::vector<float> DMatrix::y() const{
    return label;
}

std::string DMatrix::feature_name(int feature_id) const{
    return feature_names.at(feature_id);
}

int DMatrix::feature_num() const{
    return feature_num_;
}

unsigned long DMatrix::sample_num() const{
    return sample_num_;
}
#endif /* DMatrix_h */
