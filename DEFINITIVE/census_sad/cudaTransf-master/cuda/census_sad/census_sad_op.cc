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

#include "census_sad_op.h"

typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void CensusSad(const GPUDevice& d,
                typename TTypes<float, 3>::ConstTensor input_left,
                typename TTypes<float, 3>::ConstTensor input_right,
                typename TTypes<float, 3>::Tensor padded_input,
                typename TTypes<uint64, 3>::Tensor census_left,
                typename TTypes<uint64, 3>::Tensor census_right,
                typename TTypes<float, 3>::Tensor sad_output,
                typename TTypes<float, 3>::Tensor output,
                CensusSadState params);

class CensusSadOp : public OpKernel {
public:
  explicit CensusSadOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);
    typename TTypes<float, 3>::ConstTensor input_left = input_0.tensor<float, 3>();

    const Tensor& input_1 = context->input(1);
    typename TTypes<float, 3>::ConstTensor input_right = input_1.tensor<float, 3>();

    const int in_height = input_left.dimension(0);
    const int in_width = input_left.dimension(1);
    const int in_channels = input_left.dimension(2);

    CensusSadState st(attrs, in_height, in_width);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid census_sad settings, invalid shapes"));

    OP_REQUIRES(context, ((2*XDIM_Q_THREADS+st.ndisp)*st.tchuncks)*sizeof(uint64)< SHARED_MEMORY, 
                errors::InvalidArgument("Invalid census_sad settings, larger than shared memory"));

    OP_REQUIRES(context, st.wsize%2==1, 
                errors::InvalidArgument("Invalid census_sad settings, wsize must be an odd number"));

    Tensor* output = NULL;
    Tensor* padded_input = NULL;
    Tensor* census_left = NULL;
    Tensor* census_right = NULL;
    Tensor* sad_output = NULL;

    TensorShape output_shape({st.out_height, st.out_width, 1});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    TensorShape census_left_shape({st.census_out_height, st.census_out_width, st.census_out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, census_left_shape, &census_left));

    TensorShape census_right_shape({st.census_out_height, st.census_out_width, st.census_out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(2, census_right_shape, &census_right));

    TensorShape sad_output_shape({st.out_height, st.out_width, st.sad_out_channels});
    OP_REQUIRES_OK(context, context->allocate_output(3, sad_output_shape, &sad_output));

    TensorShape padded_shape({st.padded_height, st.padded_width, in_channels});
    OP_REQUIRES_OK(context, context->allocate_output(4, padded_shape, &padded_input));

    typename TTypes<float, 3>::Tensor padded_data = padded_input->tensor<float, 3>();
    typename TTypes<uint64, 3>::Tensor census_left_data = census_left->tensor<uint64, 3>();
    typename TTypes<uint64, 3>::Tensor census_right_data = census_right->tensor<uint64, 3>();
    typename TTypes<float, 3>::Tensor sad_output_data = sad_output->tensor<float, 3>();
    typename TTypes<float, 3>::Tensor output_data = output->tensor<float, 3>();
    CensusSad(context->eigen_device<GPUDevice>(), input_left, input_right, padded_data, census_left_data, census_right_data, sad_output_data, output_data, st);
  }

private:
  CensusSadAttrs attrs;
};


using shape_inference::DimensionHandle;;

REGISTER_OP("CensusSad")
  .Input("input_0: float")
  .Input("input_1: float")
  .Attr("ndisp: int =128")
  .Attr("wsize: int = 5")
  .Output("output: float")
  .Output("census_left: uint64")
  .Output("census_right: uint64")
  .Output("sad_output: float")
  .Output("padded_input: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    CensusSadAttrs attrs;
    c->GetAttr("ndisp", &attrs.ndisp);
    c->GetAttr("wsize", &attrs.wsize);

    DimensionHandle batch = c->Dim(c->input(0), 0);
    
    int vecsize =attrs.wsize * attrs.wsize;
	  if(vecsize%64 > 0)
		  vecsize += 64-(vecsize&63);
	  int tchuncks = vecsize/64; 
    int out_channels = tchuncks;
    int maxcost=tchuncks*64;

    c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim(), 1}));
    return Status::OK();
  });

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("CensusSad").Device(DEVICE_GPU), CensusSadOp);
#endif 
