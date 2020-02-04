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

struct SadAttrs {
  SadAttrs(OpKernelConstruction* c) {
    OP_REQUIRES_OK(c, c->GetAttr("ndisp", &ndisp));
  }

  SadAttrs() {}
  int ndisp;    //number of disparities
};


struct SadState {
  SadState(SadAttrs attrs, int in_height, int in_width, int in_channels) {
    ndisp = attrs.ndisp;

    out_width = in_width;
    out_height = in_height;

    tchuncks=in_channels;
    maxcost=tchuncks*64;

    sad_out_channels= ndisp;
  }
  int ndisp;
  int tchuncks;
  int maxcost;
  int out_width;
  int out_height;
  int sad_out_channels;
};
