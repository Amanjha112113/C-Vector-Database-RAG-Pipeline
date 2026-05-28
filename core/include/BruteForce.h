#pragma once

#include "IVectorIndex.h"
#include <vector>
#include <algorithm>

class BruteForce : public IVectorIndex {
private:
    std::vector<VectorItem> items;

public:
    void insert(const VectorItem& v, DistFn dist) override;
    
    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, DistFn dist) override;
        
    void remove(int id) override;
};
