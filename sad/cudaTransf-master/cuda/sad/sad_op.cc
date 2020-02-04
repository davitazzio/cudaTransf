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

#include "sad_op.h"

typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void Sad(const GPUDevice& d,
                typename TTypes<uint64, 3>::ConstTensor input_census_left,
                typename TTypes<uint64, 3>::ConstTensor input_census_right,
                typename TTypes<float, 3>::Tensor sad_output,
                typename TTypes<float, 3>::Tensor output,
                SadState params);

class SadOp : public OpKernel {
public:
  explicit SadOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);
    typename TTypes<uint64, 3>::ConstTensor input_left = input_0.tensor<uint64, 3>();

    const Tensor& input_1 = context->input(1);
    typename TTypes<uint64, 3>::ConstTensor input_right = input_1.tensor<uint64, 3>();

    const int in_height_left = input_left.dimension(0);
    const int in_width_left = input_left.dimension(1);
    const int in_channels_left = input_left.dimension(2);

    const int in_height_right = input_right.dimension(0);
    const int in_width_right = input_right.dimension(1);
    const int in_channels_right = input_right.dimension(2);

    SadState st(attrs, in_height_left, in_width_left, in_channels_left);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid sad settings, invalid shapes"));

    OP_REQUIRES(context, ((2*XDIM_Q_THREADS+st.ndisp)*st.tchuncks)*sizeof(uint64)< SHARED_MEMORY, 
                errors::InvalidArgument("Invalid sad settings, larger than shared memory"));

    OP_REQUIRES(context, in_height_left==in_height_right && in_width_left==in_width_right && in_channels_left==in_channels_right,
                errors::InvalidArgument("Invalid sad settings, input images must have the same dimension"));

    Tensor* output = NULL;
    Tensor* sad_output = NULL;

    TensorShape output_shape({st.out_height, st.out_width, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    TensorShape sad_output_shape({st.out_height, st.out_width, st.sad_out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, sad_output_shape, &sad_output));

    typename TTypes<float, 3>::Tensor sad_output_data = sad_output->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor output_data = output->tensor<float, 3>();
    Sad(context->eigen_device<GPUDevice>(), input_left, input_right, sad_output_data, output_data, st);
  }

private:
  SadAttrs attrs;
};


using shape_inference::DimensionHandle;;

REGISTER_OP("Sad")
  .Input("input_0: uint64")
  .Input("input_1: uint64")
  .Attr("ndisp: int =128")
  .Output("output: float")
  .Output("sad_output: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    SadAttrs attrs;
    c->GetAttr("ndisp", &attrs.ndisp);

    DimensionHandle batch = c->Dim(c->input(0), 0);

    c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 1}));
    return Status::OK();
  });

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Sad").Device(DEVICE_GPU), SadOp);
#endif 
