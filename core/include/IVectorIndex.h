#pragma once

#include "VectorItem.h"
#include "Distance.h"
#include <vector>
#include <utility>

class IVectorIndex {
public:
    virtual ~IVectorIndex() = default;

    // Insert a vector item into the index
    virtual void insert(const VectorItem& v, DistFn dist) = 0;

    // Search for top-k nearest neighbors
    // Returns a list of pairs: {distance, item_id}
    virtual std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, DistFn dist) = 0;

    // Remove a vector by id
    virtual void remove(int id) = 0;
};
