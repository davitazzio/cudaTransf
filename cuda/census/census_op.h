#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

struct CensusAttrs {
  CensusAttrs(OpKernelConstruction* c) {
    OP_REQUIRES_OK(c, c->GetAttr("ndisp", &ndisp));
    OP_REQUIRES_OK(c, c->GetAttr("wsize", &wsize));
  }
  
  CensusAttrs() {}
  int ndisp;
  int wsize;
};