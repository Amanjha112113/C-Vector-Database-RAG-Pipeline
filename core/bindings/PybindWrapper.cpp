#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "VectorItem.h"
#include "Distance.h"
#include "BruteForce.h"
#include "KDTree.h"
#include "HNSW.h"

namespace py = pybind11;

PYBIND11_MODULE(core_vectordb, m) {
    m.doc() = "C++ Vector Database core with HNSW, KDTree, and BruteForce";

    // Bind Distance metric getter
    m.def("get_dist_fn", &getDistFn, "Get distance function by name (cosine, euclidean, manhattan)");

    // Bind VectorItem
    py::class_<VectorItem>(m, "VectorItem")
        .def(py::init<int, std::string, std::string, std::vector<float>>())
        .def_readwrite("id", &VectorItem::id)
        .def_readwrite("metadata", &VectorItem::metadata)
        .def_readwrite("category", &VectorItem::category)
        .def_readwrite("emb", &VectorItem::emb);

    // Bind BruteForce
    py::class_<BruteForce>(m, "BruteForce")
        .def(py::init<>())
        .def("insert", &BruteForce::insert)
        .def("knn", &BruteForce::knn)
        .def("remove", &BruteForce::remove);

    // Bind KDTree
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<int>())
        .def("insert", &KDTree::insert)
        .def("knn", &KDTree::knn)
        .def("remove", &KDTree::remove)
        .def("rebuild", &KDTree::rebuild);

    // Bind GraphInfo structures for HNSW
    py::class_<GraphInfo::NV>(m, "GraphInfo_NV")
        .def_readwrite("id", &GraphInfo::NV::id)
        .def_readwrite("metadata", &GraphInfo::NV::metadata)
        .def_readwrite("category", &GraphInfo::NV::category)
        .def_readwrite("maxLyr", &GraphInfo::NV::maxLyr);

    py::class_<GraphInfo::EV>(m, "GraphInfo_EV")
        .def_readwrite("src", &GraphInfo::EV::src)
        .def_readwrite("dst", &GraphInfo::EV::dst)
        .def_readwrite("lyr", &GraphInfo::EV::lyr);

    py::class_<GraphInfo>(m, "GraphInfo")
        .def_readwrite("topLayer", &GraphInfo::topLayer)
        .def_readwrite("nodeCount", &GraphInfo::nodeCount)
        .def_readwrite("nodesPerLayer", &GraphInfo::nodesPerLayer)
        .def_readwrite("edgesPerLayer", &GraphInfo::edgesPerLayer)
        .def_readwrite("nodes", &GraphInfo::nodes)
        .def_readwrite("edges", &GraphInfo::edges);

    // Bind HNSW
    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int>(), py::arg("m") = 16, py::arg("efBuild") = 200)
        .def("insert", &HNSW::insert)
        .def("knn", &HNSW::knn)
        .def("knn_ef", &HNSW::knn_ef)
        .def("remove", &HNSW::remove)
        .def("get_info", &HNSW::getInfo)
        .def("size", &HNSW::size);
}
