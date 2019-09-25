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
                 typename TTypes<float, 3>::ConstTensor input,
                  typename TTypes<float, 3>::Tensor padded_input,
                 typename TTypes<float, 3>::Tensor output,
                 CensusState params);

class CensusOp : public OpKernel {
public:
  explicit CensusOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);

    typename TTypes<float, 3>::ConstTensor input_data = input_0.tensor<float, 3>();
    
    const int batch = 0;
    const int in_height = input_data.dimension(0);
    const int in_width = input_data.dimension(1);
    const int in_channels = input_data.dimension(2);

    CensusState st(attrs, in_height, in_width);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid census settings, invalid shapes"));

    OP_REQUIRES(context, ((2*XDIM_Q_THREADS+st.ndisp)*st.tchuncks)*sizeof(uint64)< SHARED_MEMORY, 
                errors::InvalidArgument("Invalid census settings, larger than shared memory"));

    Tensor* output = NULL;
    Tensor* padded_input = NULL;

    TensorShape output_shape({st.out_height, st.out_width, st.out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    TensorShape padded_shape({st.padded_height, st.padded_width, in_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, padded_shape, &padded_input));

    typename TTypes<float, 3>::Tensor padded_input_data = padded_input->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor output_data = output->tensor<float, 3>();
    Census(context->eigen_device<GPUDevice>(), input_data, padded_input_data, output_data, st);
  }

private:
  CensusAttrs attrs;
};


using shape_inference::DimensionHandle;;

REGISTER_OP("Census")
  .Input("input_0: float")
  .Attr("ndisp: int = 256")
  .Attr("wsize: int = 9")
  .Attr("pad: int=4")
  .Output("census: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    CensusAttrs attrs;
    c->GetAttr("ndisp", &attrs.ndisp);
    c->GetAttr("wsize", &attrs.wsize);
    c->GetAttr("pad", &attrs.pad_size);

    DimensionHandle batch = c->Dim(c->input(0), 0);

    int out_channels = 1;

    c->set_output(0, c->MakeShape({batch, out_channels, c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
  });

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Census").Device(DEVICE_GPU), CensusOp);
#endif 