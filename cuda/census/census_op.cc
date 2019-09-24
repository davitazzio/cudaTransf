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
                 typename TTypes<float, 3>::ConstTensor input_0,
                 typename TTypes<float, 3>::ConstTensor input_1,
                 typename TTypes<float, 3>::Tensor output,
                 CensusState params);

class CensusOp : public OpKernel {
public:
  explicit CensusOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);
    const Tensor& input_1 = context->input(1);

    OP_REQUIRES(context, input_0.shape() == input_1.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    typename TTypes<float, 3>::ConstTensor imgl_d = input_0.tensor<float, 3>();
    typename TTypes<float, 3>::ConstTensor imgr_d = input_1.tensor<float, 3>();
   
    const int in_channels = input_0_data.dimension(1);
    const int in_height = input_0_data.dimension(2);
    const int in_width = input_0_data.dimension(3);
    
    CensusState st(attrs, in_height, in_width, in_channels);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid census settings"));

    Tensor* output = NULL;
    Tensor* census_l = NULL;
    Tensor* census_r = NULL;
    Tensor* padded_imgl_d = NULL;
    Tensor* padded_imgr_d = NULL;

    TensorShape output_shape({out_channels, out_height, out_width});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    TensorShape padded_shape({batch, st.padded_height, st.padded_width, in_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, padded_shape, &padded_imgl_d));
    OP_REQUIRES_OK(context, context->allocate_output(2, padded_shape, &padded_imgr_d));
    OP_REQUIRES_OK(context, context->allocate_output(3, padded_shape, &census_l));
    OP_REQUIRES_OK(context, context->allocate_output(4, padded_shape, &census_r));

    typename TTypes<float, 3>::Tensor output_data = output->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor padded_imgl_data = padded_imgl_d->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor padded_imgr_data = padded_imgr_d->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor census_l_data = census_l->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor census_r_data = census_r->tensor<float, 3>();
    Census(context->eigen_device<GPUDevice>(), imgl_d, imgr_d, padded_imgl_data, padded_imgr_data, census_l_data, census_r_data, output_data, st);
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
  .Attr("pad: int=4")
  .Output("census: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    CensusAttrs attrs;
    c->GetAttr("ndisp", &attrs.ndisp);
    c->GetAttr("wsize", &attrs.wsize);
    c->GetAttr("pad", &attr.pad);

    DimensionHandle batch = c->Dim(c->input(0), 0);

    int out_channels = 1;

    c->set_output(0, c->MakeShape({batch, out_channels, c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
  });

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Census").Device(DEVICE_GPU), CensusOp);
#endif 