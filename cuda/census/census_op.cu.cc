#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "census_op.h"

template <typename Dtype>
__global__ void blob_rearrange_kernel2(const Dtype* in, Dtype* out, int num, int channels, int width, int height, int widthheight, int padding, int pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;

    Dtype value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}

template<typename Dtype>
__global__ void CensusTransformKernel(const Dtype* image, Dtype* census ,int rows, int cols, int wsize, int chunks ){

	const int shift = blockIdx.x*wsize;
    extern __shared__ float cens_slice[];
    int Row = blockIdx.y;
    int Col = blockIdx.x*blockDim.x+ threadIdx.x-shift; 
    int wc = wsize/2;
    float center;
    uint pos = 0;
    Dtype cens=0;
    int ch=0;

   	if(Col < cols-wsize && Row< rows-wsize)
	    center = image[(Row+wc)*cols + Col+wc ];



	if(Col < cols && Row< rows-wsize){

    	for(int i=0; i<wsize; i++){
    		cens_slice[threadIdx.x] = image[(Row+i)*cols + Col ];
	    		
	    	__syncthreads();

	    	if(threadIdx.x < blockDim.x-wsize && Col<cols-wsize){

		    	for(int ww=0; ww<wsize;ww++){ 
		    		if( center < cens_slice[threadIdx.x+ww])
		    			cens ^= 1UL << pos;
		    		pos++;

		    		if( (pos & 63) == 63 ){
		    			census[ ((Row+wc)*cols + (Col+wc))*chunks+ch ] = cens;
		    			ch++;
		    			cens=0;
		    			pos=0;
		    		}
		    	}
		    }
		    
		    __syncthreads();
		}


		if(threadIdx.x < blockDim.x-wsize && ch<chunks){
    		census[ ((Row+wc)*cols + (Col+wc))*chunks+ch] = cens;
    	}
    }
}

void Census(const GPUDevice& d,
                 typename TTypes<float, 3>::ConstTensor imgl_d,
                 typename TTypes<float, 3>::ConstTensor imgr_d,
                 typename TTypes<float, 3>::Tensor padded_imgl_d,
                 typename TTypes<float, 3>::Tensor census_l,
                 typename TTypes<float, 3>::Tensor census_r,
                 typename TTypes<float, 3>::Tensor output,
                 CensusState params) {				

    const int wsize = params.wsize;
    const int width = params.padded_width;
    const int height = params.padded_height;
    const int tchuncks = params.tchuncks;
    const int pad_size = params.pad_size;

    // input shape
    const int bnum = 0;//input_0.dimension(0);
    const int bchannels = imgl_d.dimension(2);
    const int bheight = imgl_d.dimension(0);
    const int bwidth = imgl_d.dimension(1);
    const int bwidthheight = bwidth * bheight;
    
    dim3 dimBlockCens(XDIM_MAX_THREADS);
    float blockx = (float)width / XDIM_MAX_THREADS;
    dim3 dimGridCens(ceil((float) blockx) + (blockx*wsize)/XDIM_MAX_THREADS ,height-wsize);

	dim3 dimBlock(XDIM_Q_THREADS);
    dim3 dimGrid(ceil((float)width /XDIM_Q_THREADS), height-wsize);

    dim3 argBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 argGrid(ceil((float) width / BLOCK_SIZE),ceil( (float)height/ BLOCK_SIZE));

    cudaMemset(census_l.data(), 0, census_l.size()*sizeof(float));
 	cudaMemset(census_r.data(), 0, census_r.size()*sizeof(float));
    int threads_per_block=16;
    dim3 totalBlocksRearr((bwidthheight-1)/threads_per_block+1, bchannels, bnum);
    const int pwidthheight = (bwidth + 2 * pad_size) * (bheight + 2 * pad_size);

    blob_rearrange_kernel2<float><<<totalBlocksRearr,threads_per_block>>>
          (imgl_d.data(),padded_imgl_d.data(),bnum,bchannels,bwidth,bheight,bwidthheight,pad_size,pwidthheight);

    //blob_rearrange_kernel2<float><<<totalBlocksRearr,threads_per_block>>>
    //      (input_1.data(),padded_1.data(),bnum,bchannels,bwidth,bheight,bwidthheight,pad_size_,pwidthheight);
	
    CensusTransformKernel<<<dimGridCens, dimBlockCens,XDIM_MAX_THREADS*sizeof(float)>>>(padded_imgl_d, census_l, height, width, wsize, tchuncks);
	//CensusTransformKernel<<<dimGridCens, dimBlockCens,XDIM_MAX_THREADS*sizeof(float)>>>(imgr_d,census_r_d,height, width,wsize,tchuncks);
	//CensusSADKernel<<<dimGrid, dimBlock,((2*XDIM_Q_THREADS+ndisp)*tchuncks)*sizeof(uint64)>>>(census_l_d,census_r_d,cost_d,height, width,ndisp
	//   																									,wsize,maxcost,tchuncks,((XDIM_Q_THREADS+ndisp)*tchuncks));
				
}

#endif  // GOOGLE_CUDA
