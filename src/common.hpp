#pragma once

#include <assert.h>
#include <memory>
#include <ATen/ATen.h>
#include <matrix/kaldi-matrix.h>
#include <cudamatrix/cu-matrix.h>

namespace common {
    template <typename T>
    struct ScalarTypeof;

    using at::Half;

#define ATNN_SCALAR_TYPE_OF(_1,n,_2) \
    template <> struct ScalarTypeof<_1> { constexpr static at::ScalarType value = at::k##n ; };
    AT_FORALL_SCALAR_TYPES(ATNN_SCALAR_TYPE_OF)
#undef ATNN_SCALAR_TYPE_OF

    template <typename Matrix>
    using ElemT = typename std::decay<decltype(*(std::declval<Matrix>().Data()))>::type;

    // NVCC's BUG
    // template <typename T>
    // constexpr at::ScalarType scalar_typeof = ScalarTypeof<T>::value;

    template <typename Elem, at::Backend device, typename Matrix>
    at::Tensor make_tensor_impl(std::shared_ptr<Matrix> ptr) {
        constexpr auto s = ScalarTypeof<Elem>::value;
        auto deleter = [ptr](void*) mutable { ptr.reset(); };
        auto t = at::getType(device, s).tensorFromBlob(ptr->Data(),
                                                       {ptr->NumRows(), ptr->NumCols()},
                                                       {ptr->Stride(), 1},
                                                       deleter);
        return t;
    }

    template <typename Elem>
    at::Tensor make_tensor(std::shared_ptr<kaldi::Matrix<Elem>> ptr) {
        return make_tensor_impl<Elem, at::kCPU>(ptr);
    }

    template <typename Elem>
    at::Tensor make_tensor(std::shared_ptr<kaldi::CuMatrix<Elem>> ptr) {
        return make_tensor_impl<Elem, at::kCUDA>(ptr);
    }

    template <typename Matrix>
    at::Tensor make_tensor(Matrix m) {
        using Elem = ElemT<Matrix>;
        constexpr auto s = ScalarTypeof<Elem>::value;
        return at::getType(at::kCUDA, s).tensorFromBlob(m.Data(),
                                                        {m.NumRows(), m.NumCols()},
                                                        {m.Stride(), 1});
    }

    template <typename Matrix>
    struct Deleter {
        at::Tensor t;

        void operator()(Matrix* p) {
            t.reset();
            delete p;
        }
    };

    template <typename Matrix>
    // std::unique_ptr<Matrix, Deleter<Matrix>>
    Matrix make_matrix(at::Tensor t) {
        using Elem = ElemT<Matrix>;

        constexpr bool is_cpu = std::is_same<Matrix, kaldi::SubMatrix<Elem>>::value;
        constexpr bool is_cuda = std::is_same<Matrix, kaldi::CuSubMatrix<Elem>>::value;
        static_assert(is_cpu || is_cuda, "type Matrix should be kaldi::SubMatrix or kaldi::CuSubMatrix");
        if (is_cpu && t.type().backend() != at::kCPU) {
            t = t.toBackend(at::kCPU);
        }
        if (is_cuda && t.type().backend() != at::kCUDA) {
            t = t.toBackend(at::kCUDA);
        }
        if (t.dim() != 2) throw std::runtime_error("at::Tensor.ndim() != 2");
        if (t.stride(1) != 1) throw std::runtime_error("at::Tensor.stride(1) != 1");

        return Matrix(t.template data<Elem>(), t.size(0), t.size(1), t.stride(0));
        // auto ptr = new Matrix(
        //     t.template data<Elem>(), t.size(0), t.size(1), t.stride(0));
        // return std::unique_ptr<Matrix, decltype(deleter)>(ptr, deleter);
    }
}
