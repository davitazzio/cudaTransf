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
                typename TTypes<uint64, 3>::Tensor output,
                CensusState params);

class CensusOp : public OpKernel {
public:
  explicit CensusOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);
    typename TTypes<float, 3>::ConstTensor input_data = input_0.tensor<float, 3>();

    const int in_height = input_data.dimension(0);
    const int in_width = input_data.dimension(1);
    const int in_channels = input_data.dimension(2);

    CensusState st(attrs, in_height, in_width);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid census settings, invalid shapes"));

    OP_REQUIRES(context, (st.tchuncks)*sizeof(uint64)< SHARED_MEMORY, 
                errors::InvalidArgument("Invalid census settings, larger than shared memory"));

    OP_REQUIRES(context, st.wsize%2==1,
                errors::InvalidArgument("Invalid census settings, wsize must be an odd number"));

    Tensor* output = NULL;
    Tensor* padded_input = NULL;

    TensorShape output_shape({st.out_height, st.out_width, st.out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    TensorShape padded_shape({st.padded_height, st.padded_width, in_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, padded_shape, &padded_input));

    typename TTypes<float, 3>::Tensor padded_input_data = padded_input->tensor<float, 3>();
    typename TTypes<uint64, 3>::Tensor output_data = output->tensor<uint64, 3>();
    Census(context->eigen_device<GPUDevice>(), input_data, padded_input_data, output_data, st);
  }

private:
  CensusAttrs attrs;
};


using shape_inference::DimensionHandle;;

REGISTER_OP("Census")
  .Input("input_0: float")
  .Attr("wsize: int =5")
  .Output("output: uint64")
  .Output("padded_input: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    CensusAttrs attrs;
    c->GetAttr("wsize", &attrs.wsize);

    DimensionHandle batch = c->Dim(c->input(0), 0);
    
    int vecsize =attrs.wsize * attrs.wsize;
	  if(vecsize%64 > 0)
		  vecsize += 64-(vecsize&63);
	  int tchuncks = vecsize/64; 
    int out_channels = tchuncks*64;

    c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), out_channels}));
    return Status::OK();
  });

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Census").Device(DEVICE_GPU), CensusOp);
#endif 
