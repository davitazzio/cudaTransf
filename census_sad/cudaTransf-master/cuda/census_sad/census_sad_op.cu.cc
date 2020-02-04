#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


#include "census_sad_op.h"
typedef uint8_t uint8;
typedef unsigned int uint32;
typedef unsigned long long int uint64;


template <typename Dtype>
__global__ void blob_rearrange_kernel2(const Dtype* in, Dtype* out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    Dtype value=in[(n*channels+ch)+xy];
    __syncthreads();
    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;
    int position = ((n*pwidthheight+xypad)*channels + ch);

    out[position] = value;  
}

__global__ void CensusTransformKernel(const float* image, uint64* census ,int rows, int cols, int windows, int chunks ){
    const int wsize= windows-1;
    const int shift = blockIdx.x*wsize;
    extern __shared__ float cens_slice[];
    int Row = blockIdx.y;
    int Col = blockIdx.x*blockDim.x+ threadIdx.x-shift; 
    int wc = wsize/2;
    float center;
    uint pos = 0;
    uint64 cens=0;
    int ch=0;

   	if(Col < cols-wsize && Row< rows-wsize)
	    center = image[(Row+wc)*cols + Col+wc ];
        
	if(Col <= cols && Row< rows-wsize){
    	for(int i=0; i<windows; i++){
    		cens_slice[threadIdx.x] = image[(Row+i)*cols + Col ];
	    	
	    	__syncthreads();

	        if(threadIdx.x < blockDim.x-wsize && Col<cols-wsize && Row<rows-wsize){
                for(int ww=0; ww<windows;ww++){ 
                    if(ww==wc && i==wc){}
                    else{
                        if( center < cens_slice[threadIdx.x+ww])
                            cens ^= 1UL << pos;
                        pos++;	
                        if( (pos & 63) == 63 ){
                            census[(Row*(cols-wsize)*chunks) + (Col*chunks)+ch] = cens;
                            ch++;
                            cens=0;
                            pos=0;
                        }
                    }
		        }
		    }
		    
		__syncthreads();
	    }
   
	    if(threadIdx.x < blockDim.x-wsize && ch<chunks && Col<cols-wsize && Row<rows-wsize){
    	    census[(Row*(cols-wsize)*chunks) + (Col*chunks)+ch] = cens;
    	}
    }
}

__global__ void SadKernel(uint64* censusl, uint64* censusr, float* cost, int rows, int cols, int ndisp, int wsize,int maxcost,int chunks,int sm_offset ){

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

void CensusSad(const GPUDevice& d,
                typename TTypes<float, 3>::ConstTensor input_left,
                typename TTypes<float, 3>::ConstTensor input_right,
                typename TTypes<float, 3>::Tensor padded_input,
                typename TTypes<uint64, 3>::Tensor census_left,
                typename TTypes<uint64, 3>::Tensor census_right,
		        typename TTypes<float, 3>::Tensor sad_output,
                typename TTypes<float, 3>::Tensor output,
                CensusSadState params) {

    const int wsize = params.wsize;
    const int width = params.padded_width;
    const int height = params.padded_height;
    const int tchuncks = params.tchuncks;
    const int pad_size = params.pad_size;
    const int maxcost = params.maxcost;
    const int ndisp = params.ndisp;

    const int bnum = 1;
    const int bheight = input_left.dimension(0);
    const int bwidth = input_left.dimension(1);
    const int bchannels = input_left.dimension(2);
    const int bwidthheight = bwidth * bheight;
    const int output_size=bwidthheight * ndisp;
    const int pwidthheight = (bwidth + 2 * pad_size) * (bheight + 2 * pad_size);
   
    //DIM BLOCk AND GRID FOR CENSUS TRANSORM
    dim3 dimBlockCens(XDIM_MAX_THREADS);                                                //max threads TMAX in each block
    float blockx = (float)width / XDIM_MAX_THREADS;                                     //split the width into B blocks, each of them with TMAX threads
    dim3 dimGridCens(ceil((float) blockx + (blockx*wsize)/XDIM_MAX_THREADS) ,bheight);  //create G grids, where G cover image width in x and height in y
	
    //DIM BLOCK AND GRID FOR BLOB REARRANGE
    int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    
    //CENSUS LEFT IMAGE
    cudaMemset(census_left.data(), 0, census_left.size()*sizeof(uint64));
    cudaMemset(padded_input.data(), 0, padded_input.size()*sizeof(float));
    blob_rearrange_kernel2<float><<<totalBlocksRearr,threads_per_block>>>(input_left.data(), padded_input.data(), bnum, bchannels, bwidth, bheight, bwidthheight, pad_size, pwidthheight);
    CensusTransformKernel<<<dimGridCens, dimBlockCens, XDIM_MAX_THREADS*sizeof(float)>>>(padded_input.data(), census_left.data(), height,width,wsize,tchuncks);
    
    //CENSUS RIGHT IMAGE
    cudaMemset(census_right.data(), 0, census_right.size()*sizeof(uint64));
    cudaMemset(padded_input.data(), 0, padded_input.size()*sizeof(float));
    blob_rearrange_kernel2<float><<<totalBlocksRearr,threads_per_block>>>(input_right.data(), padded_input.data(), bnum, bchannels, bwidth, bheight, bwidthheight, pad_size, pwidthheight);
    CensusTransformKernel<<<dimGridCens, dimBlockCens, XDIM_MAX_THREADS*sizeof(float)>>>(padded_input.data(), census_right.data(), height,width,wsize,tchuncks);
    
	//SAD
	dim3 dimBlock(XDIM_Q_THREADS);
    dim3 dimGrid(ceil((float)bwidth/XDIM_Q_THREADS),bheight);

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) bwidth / BLOCK_SIZE),ceil( (float)bheight/ BLOCK_SIZE));
    cudaMemset(sad_output.data(), 0, sad_output.size()*sizeof(float));

    inti_cost<<< argGrid, argBlock,0 >>>( sad_output.data(), bheight, bwidth, ndisp, maxcost);
	SadKernel<<<dimGrid, dimBlock,((2*XDIM_Q_THREADS+ndisp)*tchuncks)*sizeof(uint64)>>>(census_left.data(),census_right.data(),sad_output.data(), bheight, bwidth, ndisp, wsize, maxcost, tchuncks,((XDIM_Q_THREADS+ndisp)*tchuncks));

	argmin<<<argGrid, argBlock, 0>>>(sad_output.data(), output.data(), bheight, bwidth, ndisp );	
}
#endif  // GOOGLE_CUDA


