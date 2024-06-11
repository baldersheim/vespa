// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

#pragma once

#include "distance_function.h"
#include "distance_function_factory.h"
#include <vespa/eval/eval/typed_cells.h>
#include <mutex>
#include <memory>

namespace search::tensor {

/**
 * Thread-safe storage of maximum value for squared vector norm.
 * sq_norm = |x|^2 = sum(x[i]*x[i]) = dotproduct(x,x)
 * Note that the initial value is 1.0; so even if all
 * vectors seen have 0 or very small length, you will never
 * get a value < 1.0.
 */
class MaximumSquaredNormStore {
private:
    std::mutex _lock;
    double _max_sq_norm;
public:
    MaximumSquaredNormStore() noexcept : _lock(), _max_sq_norm(1.0) {}
    /**
     * Fetch the maximum value seen so far.
     * Usually you will also supply a value computed for a newly seen
     * vector, which may update the maximum value.
     */
    double get_max(double value = 0.0) {
        std::lock_guard<std::mutex> guard(_lock);
        if (value > _max_sq_norm) [[unlikely]] {
            _max_sq_norm = value;
        }
        return _max_sq_norm;
    }
};

class MipsDistanceFunctionFactoryBase : public DistanceFunctionFactory {
protected:
    std::shared_ptr<MaximumSquaredNormStore> _sq_norm_store;
public:
    MipsDistanceFunctionFactoryBase()
        : _sq_norm_store(std::make_shared<MaximumSquaredNormStore>())
    {
    }
    ~MipsDistanceFunctionFactoryBase() override = default;
    MaximumSquaredNormStore& get_max_squared_norm_store() noexcept { return *_sq_norm_store; }
};

/**
 * Factory for distance functions which can apply a transformation
 * mapping Maximum Inner Product Search to a nearest neighbor
 * problem.  When inserting vectors, an extra dimension is
 * added ensuring behavior "as if" all vectors had length equal
 * to the longest vector inserted so far, or at least length 1.
 *
 * When reference_insertion_vector == true:
 *   - Vectors passed to for_insertion_vector() and BoundDistanceFunction::calc() are assumed to have the same type as FloatType.
 *   - The TypedCells memory is just referenced and used directly in calculations,
 *     and thus no transformation via a temporary memory buffer occurs.
 */
template <typename FloatType>
class MipsDistanceFunctionFactory : public MipsDistanceFunctionFactoryBase {
private:
    bool _reference_insertion_vector;
public:
    MipsDistanceFunctionFactory() noexcept : MipsDistanceFunctionFactory(false) {}
    MipsDistanceFunctionFactory(bool reference_insertion_vector) noexcept : _reference_insertion_vector(reference_insertion_vector) {}
    ~MipsDistanceFunctionFactory() override = default;

    BoundDistanceFunction::UP for_query_vector(TypedCells lhs) const override;
    BoundDistanceFunction::UP for_insertion_vector(TypedCells lhs) const override;
};

}
