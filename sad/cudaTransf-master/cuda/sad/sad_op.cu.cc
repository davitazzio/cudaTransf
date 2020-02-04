#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


#include "sad_op.h"
typedef uint8_t uint8;
typedef unsigned int uint32;
typedef unsigned long long int uint64;


__global__ void SadKernel(const uint64* censusl, const uint64* censusr, float* cost, int rows, int cols, int ndisp, int maxcost,int chunks,int sm_offset ){

	extern __shared__ uint64 rc_sm[];

	const int Row = blockIdx.y;
    const int Col =blockIdx.x*blockDim.x + threadIdx.x;
    uint64* lbs = &rc_sm[sm_offset];

    int threaddispl = 0;
    if(blockIdx.x >0){
    	threaddispl=ndisp;
    }

    int rp = (int)ceil( (float)ndisp/blockDim.x  );

    for(int b=0; b<rp; b++){
    
    	if(blockIdx.x > 0 && threadIdx.x < ndisp && (int)(Col -(ndisp-b*blockDim.x))>=0 ){
    	
    		for (int ch=0; ch< chunks; ch++){
                rc_sm[(threadIdx.x+b*blockDim.x)*chunks+ch] = censusr[  ((Row)*cols  +  (Col -(ndisp-b*blockDim.x)))*chunks +ch  ];
    		}
    	}
    }
    __syncthreads();

    if(Row < rows && Col < cols){
    	const int index = ((Row)*cols+ (Col))*chunks;
    	for(int ch=0; ch< chunks; ch++){
    		lbs[threadIdx.x*chunks+ch] = censusl[index+ch];
            rc_sm[(threaddispl+ threadIdx.x )*chunks+ch] = censusr[index+ch];
    	}

    	__syncthreads();

	    for (int d=0; d< ndisp; d++){
	    	const int dindex = threaddispl+threadIdx.x-d;

	    	if(Col < cols && dindex >=0 && (int)Col-d>=0){

	    		float sum =0;
					
				for(int ch=0; ch<chunks; ch++){
                    int lbs_i= lbs[ threadIdx.x*chunks+ch ];
                    int rc_i= rc_sm[dindex*chunks + ch ];
                    uint64 r=lbs_i^rc_i;
					sum +=(float)__popcll(r);
                }
				cost[Row*cols*ndisp+Col*ndisp+d]=sum;
	    	}
	    }
    }
}

__global__ void inti_cost( float* cost, int rows, int cols, int ndisp,int maxcost ){

    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x; 

    if( Row < rows && Col < cols){
	    for(int i=0; i<ndisp; i++){
	    	cost[ i*rows*cols+Row*cols+Col ] = (float)maxcost;
	    }
	}
}

template<typename Dtype>
__global__ void argmin(Dtype* cost, float* disp_d, int rows, int cols, int ndisp ){

    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x; 

    if( Row < rows && Col < cols){

	    Dtype mincost=cost[ Row*cols*ndisp+Col*ndisp ];
	    int d=0;
	    for(int i=1; i<ndisp; i++){
            float cd =  cost[ Row*cols*ndisp+Col*ndisp +i ];
	    	if( cd < mincost ){
	    		mincost = cd;
	    		d = i;
	    	}
	    }
	    disp_d[ Row*cols+Col ] = (float)d;
	}
}

void Sad(const GPUDevice& d,
                typename TTypes<uint64, 3>::ConstTensor input_census_left,
                typename TTypes<uint64, 3>::ConstTensor input_census_right,
		        typename TTypes<float, 3>::Tensor sad_output,
                typename TTypes<float, 3>::Tensor output,
                SadState params) {

    const int tchuncks = params.tchuncks;
    const int maxcost = params.maxcost;
    const int ndisp = params.ndisp;

    const int bnum = 1;
    const int bheight = input_census_left.dimension(0);
    const int bwidth = input_census_left.dimension(1);
    const int bchannels = input_census_left.dimension(2);
    const int bwidthheight = bwidth * bheight;
    const int output_size=bwidthheight * ndisp;
   
	//SAD
	dim3 dimBlock(XDIM_Q_THREADS);
    dim3 dimGrid(ceil((float)bwidth/XDIM_Q_THREADS),bheight);

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) bwidth / BLOCK_SIZE),ceil( (float)bheight/ BLOCK_SIZE));
    cudaMemset(sad_output.data(), 0, sad_output.size()*sizeof(float));

    inti_cost<<< argGrid, argBlock,0 >>>( sad_output.data(), bheight, bwidth, ndisp, maxcost);
	SadKernel<<<dimGrid, dimBlock,((2*XDIM_Q_THREADS+ndisp)*tchuncks)*sizeof(uint64)>>>(input_census_left.data(), input_census_right.data(), sad_output.data(), bheight, bwidth, ndisp, maxcost, tchuncks,((XDIM_Q_THREADS+ndisp)*tchuncks));

	argmin<<<argGrid, argBlock, 0>>>(sad_output.data(), output.data(), bheight, bwidth, ndisp );	
}
#endif  // GOOGLE_CUDA


