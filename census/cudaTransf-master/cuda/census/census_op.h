#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

#define STREAM_BLOCK 16
#define BLOCK_SIZE 32
#define BLOCK_D_SIZE 64
#define INTEGRAL_BLOCK_SIZE 8
#define XDIM_MAX_THREADS 120
#define SHARED_MEMORY 49152
#define XDIM_H_THREADS 5
#define XDIM_Q_THREADS 2

struct CensusAttrs {
  CensusAttrs(OpKernelConstruction* c) {
    OP_REQUIRES_OK(c, c->GetAttr("wsize", &wsize));
  }

  CensusAttrs() {}
  int wsize;    //window size
};


struct CensusState {
  CensusState(CensusAttrs attrs, int in_height, int in_width) {
    wsize = attrs.wsize;
    pad_size=(wsize-1)/2;
    padded_height = in_height + 2 * pad_size;
    padded_width = in_width + 2 * pad_size;

    out_width = in_width;
    out_height = in_height;


    vecsize = wsize*wsize;
	  if(vecsize%64 > 0)
		  vecsize += 64-(vecsize&63);
	  tchuncks = vecsize/64; 
    out_channels = tchuncks;
  }
  int pad_size;
  int wsize;
  int padded_height;
  int padded_width;
  int out_height;
  int out_width;
  int out_channels;
  int tchuncks;
  int vecsize;
};
