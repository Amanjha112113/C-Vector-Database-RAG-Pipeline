#pragma once

#include <vector>
#include <string>

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};
