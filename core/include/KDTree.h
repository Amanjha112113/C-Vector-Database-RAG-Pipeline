#pragma once

#include "IVectorIndex.h"
#include <vector>
#include <queue>

struct KDNode {
    VectorItem item;
    KDNode* left = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree : public IVectorIndex {
private:
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n);
    KDNode* ins(KDNode* n, const VectorItem& v, int d);
    void search(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
                std::priority_queue<std::pair<float, int>>& heap);

public:
    explicit KDTree(int d);
    ~KDTree() override;

    void insert(const VectorItem& v, DistFn dist) override;
    
    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, DistFn dist) override;
        
    void remove(int id) override;
    
    // KDTree specific method to rebuild after deletes
    void rebuild(const std::vector<VectorItem>& items);
};
