#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

#define STREAM_BLOCK 16
#define BLOCK_SIZE 32
#define BLOCK_D_SIZE 64
#define INTEGRAL_BLOCK_SIZE 8
#define XDIM_MAX_THREADS 1024
#define SHARED_MEMORY 49152
#define XDIM_H_THREADS 512
#define XDIM_Q_THREADS 256

struct CensusAttrs {
  CensusAttrs(OpKernelConstruction* c) {
    OP_REQUIRES_OK(c, c->GetAttr("ndisp", &ndisp));
    OP_REQUIRES_OK(c, c->GetAttr("wsize", &wsize));
    OP_REQUIRES_OK(c, c->GetAttr("pad", &pad_size));
  }

  CensusAttrs() {}
  int ndisp;    //number of disparities
  int wsize;    //window size
  int pad_size; //pad size
};


struct CensusState {
  CensusState(CensusAttrs attrs, int in_height, int in_width) {
    pad_size = attrs.pad_size;
    wsize = attrs.wsize;
    ndisp = attrs.ndisp;
    kernel_radius = (wsize -1) / 2 ;
    padded_height = in_height + 2 * pad_size;
    padded_width = in_width + 2 * pad_size;

    out_width = padded_width;
    out_height = padded_height;

    out_channels = 1;

    vecsize = wsize*wsize;
	  if(vecsize%64 > 0)
		  vecsize += 64-(vecsize&63);
	  tchuncks = vecsize/64;   
  }
  int ndisp;
  int pad_size;
  int wsize;
  int padded_height;
  int padded_width;
  int out_height;
  int out_width;
  int out_channels;
  int kernel_radius;
  int tchuncks;
  int vecsize;
};