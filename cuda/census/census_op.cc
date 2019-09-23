#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "census_op.h"

typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void Census(const GPUDevice& d,
                 typename TTypes<float, 4>::ConstTensor input_0,
                 typename TTypes<float, 4>::ConstTensor input_1,
                 typename TTypes<float, 4>::Tensor output,
                 typename TTypes<float, 4>::Tensor padded_0,
                 typename TTypes<float, 4>::Tensor padded_1,
                 CensusAttrs params);

class CensusOp : public OpKernel {
public:
  explicit CensusOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);
    const Tensor& input_1 = context->input(1);

    OP_REQUIRES(context, input_0.shape() == input_1.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    typename TTypes<float, 4>::ConstTensor input_0_data = input_0.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor input_1_data = input_1.tensor<float, 4>();

    const int batch = input_0_data.dimension(0);
    const int in_channels = input_0_data.dimension(1);
    const int in_height = input_0_data.dimension(2);
    const int in_width = input_0_data.dimension(3);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid correlation settings"));

    Tensor* output = NULL;
    TensorShape output_shape({batch, st.out_channels, st.out_height, st.out_width});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    Tensor* padded_0 = NULL;
    Tensor* padded_1 = NULL;
    TensorShape padded_shape({batch, st.padded_height, st.padded_width, in_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, padded_shape, &padded_0));
    OP_REQUIRES_OK(context, context->allocate_output(2, padded_shape, &padded_1));

    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor padded_0_data = padded_0->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor padded_1_data = padded_1->tensor<float, 4>();

    Correlation2D(context->eigen_device<GPUDevice>(),
                input_0_data, input_1_data, output_data,
                padded_0_data, padded_1_data,
                st);
  }

private:
  CensusAttrs attrs;
};


using shape_inference::DimensionHandle;;

REGISTER_OP("Census")
  .Input("input_0: float")
  .Input("input_1: float")
  .Attr("ndisp: int = 256")
  .Attr("wsize: int = 9")
  .Output("census: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    CensusAttrs attrs;
    c->GetAttr("ndisp", &attrs.ndisp);
    c->GetAttr("wsize", &attrs.wsize);

    DimensionHandle batch = c->Dim(c->input(0), 0);

    int out_channels = 1;

    // TODO: support passing on output width and height

    c->set_output(0, c->MakeShape({batch, out_channels, c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
  });
