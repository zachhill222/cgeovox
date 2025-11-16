#pragma once

#include <concepts>


//types that are allowed for scalar computations (e.g. gv::util::Point, gv::util::Matrix data types)
//matricis of integral types are allowed, but may not behave as expected as integral arithmetic will be used.
template<typename T>
concept Scalar = std::integral<T> || std::floating_point<T>;

template<typename T>
concept Float = std::floating_point<T>;