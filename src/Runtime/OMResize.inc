#include <math.h>
#include <stdio.h>
#include "OnnxMlirRuntime.h"
#include "onnx-mlir/Runtime/OMTensor.h"

/* 
issues:

python on Z:
floor(10*1.3) = 1.2
half-pixel: 6.5/1.3 = 5.500000183398914
*/


/*
void resizeRuntime(OMTensor *input, int coordinate_transformation_mode, float cubic_coeff_a, int exclude_outside, float extrapolation_value, int mode, int nearest_mode, OMTensor *output) {
   int64_t n = omTensorGetRank(output);
}
*/

void nearest_coeffs(float ratio, float *coeffs_buffer, int mode) {
	/*
	    if type(ratio) == int or ratio.is_integer():
        return np.array([0, 1])
    elif mode == 'round_prefer_floor':
        return np.array([ratio <= 0.5, ratio > 0.5])
    elif mode == 'round_prefer_ceil':
        return np.array([ratio < 0.5, ratio >= 0.5])
    elif mode == 'floor':
        return np.array([1, 0])
    elif mode == 'ceil':
        return np.array([0, 1])
	*/
   /* integer ratio is handled outside */
   switch(mode) {
   case 0: // round_prefer_float
     coeffs_buffer[0] = ratio <= 0.5;
     coeffs_buffer[1] = ratio > 0.5;
     break;
   case 1: // round_prefer_ceil
     coeffs_buffer[0] = ratio < 0.5;
     coeffs_buffer[1] = ratio >= 0.5;
     break;
   case 2: // floor
     coeffs_buffer[0] = 1;
     coeffs_buffer[1] = 0;
     break;
   case 4: // ceil
     coeffs_buffer[0] = 0;
     coeffs_buffer[1] = 1;
     break;
   }
}

void  get_neighbor(float x,
	int64_t n,
	//OMTensor *data, 
	int limit,
	float *data,
	float *points,
	int exclude_outside) {
  // inline the python get_neighbor_idx function without real padding
  // Avoid malloc/free

  // nearest indx: identify the central idx first, then select from both side
  // If the central idx is right to x (>= x), favor the left one
  // == from the example
  

  int pad_width = ceil(n/2);
  x += pad_width;
  float r = x-floor(x);

  int start, end;
  int c;
  if (r > 0.5) {
     c = (int)(floor(x))+1;
  } else {
     c = (int)(floor(x));
  }

     int rest = n-1;
     int half = rest/2;
     if (rest == 0) {
	     start=end=c;
     } else if (rest%2 == 0) {
       start = c - half;
       end = c + half;
     } else if (r == 0) { 
       start = c - half -1;
       end = c + half;
     } else {
       if (r > 0.5) {
         end = c + half;
         start = c - half-1;
       } else {
         end = c + half+1;
         start = c - half;
       }
     }  

  //printf("start %d end %d \n", start, end);
  start -= pad_width;
  end -= pad_width;

  for (int i = start; i <= end; i++) {
	  if (i<0) {
		  if (exclude_outside)
		    points[i-start] = 0;
		  else 
		    points[i-start] = data[0];
	  } else if (i >= limit) { 
		  if (exclude_outside)
		    points[i-start] = 0;
		  else 
		    points[i-start] = data[limit-1];
	  } else {
	    points[i-start] = data[i];
          }
  }
  /*
  for(int i=0; i<n; i++) {
	  printf("%f ", points[i]);
  }
  printf("\n");
  */
}

typedef void (*Coeff_Func_t)(float, float*, int mode);

static float interpolate_1d_with_x(
	OMTensor *data,
	float scale_factor,
	float x,
	Coeff_Func_t get_coeffs,
	float *coeffs_buffer,
	int coeffs_n,
	float roi,
	float extrapolation_value,
	int coordinate_transformation_mode,
	int exclude_outside,
        int mode) {

  int64_t input_width = omTensorGetShape(data)[0];
  float output_width_f = scale_factor * input_width;
  float x_ori;
  switch (coordinate_transformation_mode) {
    case 0 : //half_pixel
      x_ori = (x+0.5) / scale_factor - 0.5;
      break;
    case 1 : //asymmetric
      x_ori = x / scale_factor;
      break;
  }
  int64_t x_ori_int = floor(x_ori);

  //if x_ori.is_integer()
  //  ratio = 1
  float ratio = x_ori - x_ori_int;
  if (ratio == 0)
    ratio = 1;

  get_coeffs(ratio, coeffs_buffer, mode);
  int64_t n = coeffs_n;

  float points[coeffs_n];
  int64_t idxes[coeffs_n];

  //printf("x_ori %f ratio %f ", x_ori, ratio);
  get_neighbor(x_ori, n, input_width, (float *)omTensorGetDataPtr(data), points, exclude_outside);
  float sum = 0.;
  for (int i = 0; i < n; i++) {
    sum += coeffs_buffer[i]*points[i];
  }
  //printf("result %f %f\n", x, sum);
  return sum;
}

static float interpolate_nd_with_x(
	OMTensor *data,
	int n,
	float* scale_factors,
	float* xs,
	Coeff_Func_t get_coeffs,
	float *coeffs_buffer,
	int coeffs_n,
	float roi,
	float extrapolation_value,
	int coordinate_transformation_mode,
	int exclude_outside,
        int mode) {
  if (n == 1) {
    return interpolate_1d_with_x(data, scale_factors[0], xs[0],
	    get_coeffs, coeffs_buffer, coeffs_n, roi, extrapolation_value,
	    coordinate_transformation_mode, exclude_outside, mode);
  } else {
    int64_t input_width = omTensorGetShape(data)[0];
    float tempData[input_width];
    int64_t tempShape[] = {input_width};

    int64_t rank = omTensorGetRank(data);
    int64_t stride = 1;
    for (int i = 1; i < n; i++) {
	    stride *= omTensorGetShape(data)[i];
    }
    for (int i = 0; i < input_width; i++) {
       float *dataPtr = (float *)omTensorGetDataPtr(data) + i*stride;
       OMTensor *data1 = omTensorCreate(dataPtr, omTensorGetShape(data)+1, n-1, ONNX_TYPE_FLOAT);
       tempData[i] = interpolate_nd_with_x(data1, n-1, scale_factors+1, xs+1,
	    get_coeffs, coeffs_buffer, coeffs_n, roi, extrapolation_value,
	    coordinate_transformation_mode, exclude_outside, mode);
    }
    OMTensor *tempT = omTensorCreate(tempData, tempShape, 1, 1);
    return interpolate_1d_with_x(tempT, scale_factors[0], xs[0],
	    get_coeffs, coeffs_buffer, coeffs_n, roi, extrapolation_value,
	    coordinate_transformation_mode, exclude_outside, mode);
  }
}

void coordinate_step(int64_t rank, int64_t *output_size, int64_t  allCoordinates[][rank], int64_t currentRank, int64_t *currentIter,  int64_t *currentPosition){
  for(int i = 0; i < output_size[currentRank]; i++) {
    if (currentRank == rank-1) {
      for (int j = 0; j < currentRank; j++) {
        allCoordinates[*currentPosition][j] = currentIter[j];
      }
      allCoordinates[*currentPosition][currentRank] = i;
      (*currentPosition)++;
    } else {
      currentIter[currentRank] = i;
      coordinate_step(rank, output_size, allCoordinates, currentRank+1, currentIter, currentPosition);
    }
  }
}

void generate_coordinates(int64_t rank, int64_t *output_size, int64_t  allCoordinates[][rank]){
  int64_t position = 0;
  int64_t currentIter[rank];
  coordinate_step(rank, output_size, allCoordinates, 0, currentIter, &position);
}

void interpolate_nd(OMTensor *data,
	int64_t mode,
	int64_t *output_size,
	float   *scale_factor,
	OMTensor *roi,
	float *extrapolation_value,
	int coordinate_transformation_mode,
	int exclude_outside) {
   int64_t rank = omTensorGetRank(data);
   int64_t *inputShape = omTensorGetShape(data);
   if (scale_factor == NULL) {
     scale_factor = malloc(sizeof(float)*rank);
     for (int i = 0; i < rank; i++) {
       scale_factor[i] = (float)output_size[i]/inputShape[i];
     }
   } else {
     output_size = malloc(sizeof(int64_t)*rank);
     for (int i = 0; i < rank; i++) {
       output_size[i] = scale_factor[i]*inputShape[i];
     }
   }


   int64_t outputSize = 1;
   for (int i = 0; i < rank; i++) {
     outputSize *= output_size[i];
   }
   //float *outputData = (float*) malloc(sizeof(float)*outputSize);

   int64_t allCoordinates[outputSize][rank];
   generate_coordinates(rank, output_size, allCoordinates);
   /* debug
   for(int i = 0; i < outputSize; i++) {
     for(int j = 0; j < rank; j++) {
       printf("%ld\t", allCoordinates[i][j]);
     } 
     printf("\n");
   }
   */
   const int  COEFFS_N = 2;
   float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};

   for(int i = 0; i < outputSize; i++) {
      float Xs[rank];
      for(int j = 0; j < rank; j++) {
	Xs[j] = allCoordinates[i][j];
      }
      float r = interpolate_nd_with_x(
	/*OMTensor */data,
	/* n */4,
	/*float scale_factor*/ scale_factor,
	/*floats *x*/ Xs,
	/* Coeff_Func_t*/ nearest_coeffs,
	/*float */coeffs_buffer,
	/*int coeffs_n*/ COEFFS_N,
	/*float roi*/ 0.,
	/*float extrapolation_value*/ 0.,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0,
        /*mode */0);
       printf("%f ", r);
   }
   printf("\n");
}

void interpolate_nd_OMTensor(
        OMTensor *output_OMT,
        OMTensor *data,
	int64_t mode,
	OMTensor *output_size_OMT,
	OMTensor *scale_factor_OMT,
	OMTensor *roi,
	float *extrapolation_value,
	int coordinate_transformation_mode,
	int exclude_outside) {
   int64_t rank = omTensorGetRank(data);
   int64_t *inputShape = omTensorGetShape(data);
   float *scale_factor = NULL;
   int64_t * output_size = NULL;
   if (scale_factor_OMT != NULL) 
     scale_factor = (float*) omTensorGetDataPtr(scale_factor_OMT);
   if (output_size_OMT != NULL)
     output_size = (int64_t*)omTensorGetDataPtr(output_size_OMT);
   if (scale_factor == NULL) {
     scale_factor = malloc(sizeof(float)*rank);
     for (int i = 0; i < rank; i++) {
       scale_factor[i] = (float)output_size[i]/inputShape[i];
     }
   } else {
     output_size = malloc(sizeof(int64_t)*rank);
     for (int i = 0; i < rank; i++) {
       output_size[i] = scale_factor[i]*inputShape[i];
     }
   }

   int64_t outputSize = 1;
   for (int i = 0; i < rank; i++) {
     outputSize *= output_size[i];
   }
   float *outputData = (float*) omTensorGetDataPtr(output_OMT);

   int64_t allCoordinates[outputSize][rank];
   generate_coordinates(rank, output_size, allCoordinates);
   const int  COEFFS_N = 2;
   float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};

   for(int i = 0; i < outputSize; i++) {
      float Xs[rank];
      for(int j = 0; j < rank; j++) {
	Xs[j] = allCoordinates[i][j];
      }
      float r = interpolate_nd_with_x(
	/*OMTensor */data,
	/* n */4,
	/*float scale_factor*/ scale_factor,
	/*floats *x*/ Xs,
	/* Coeff_Func_t*/ nearest_coeffs,
	/*float */coeffs_buffer,
	/*int coeffs_n*/ COEFFS_N,
	/*float roi*/ 0.,
	/*float extrapolation_value*/ 0.,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0,
        /*mode */0);
      outputData[i] = r;
   }
}


void test_2D_1(){
  const int  COEFFS_N = 2;
  float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};
  const int  D1 =  1;
  const int D2 = 1;
  const int D3 = 2;
  const int D4 = 4;
   float data[D1*D2*D3*D4];
   for(int i = 0; i< D1*D2*D3*D4; i++) {
	   data[i] = (i+1)*1.0; // backend test starts from 1
   }
   int64_t shape[] = {D1, D2, D3, D4};
   OMTensor *myT = omTensorCreate(data, shape, 4, 1);
   float scale_factors[] = {1., 1., 0.6, 0.6};
      interpolate_nd(
	/*OMTensor */myT,
	/*mode*/ 0,
	/* output size */ NULL,
	/*float scale_factor*/ scale_factors,
	/* roi */ NULL,
	///* Coeff_Func_t*/ nearest_coeffs,
	///*float */coeffs_buffer,
	///*int coeffs_n*/ COEFFS_N,
	/*float * extrapolation_value*/ 0,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0);
}

void test_scalar_down_2D() {
  const int  COEFFS_N = 2;
   float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};
  const int  D1 =  1;
  const int D2 = 1;
  const int D3 = 2;
  const int D4 = 4;
   float data[D1*D2*D3*D4];
   for(int i = 0; i< D1*D2*D3*D4; i++) {
	   data[i] = (i+1)*1.0; // backend test starts from 1
   }
   int64_t shape[] = {D1, D2, D3, D4};
   OMTensor *myT = omTensorCreate(data, shape, 4, 1);
   float scale_factors[] = {1., 1., 0.6, 0.6};
   float Xs[4];
   printf("[");
   for(int x1 = 0; x1 < floor(D1*scale_factors[0]); x1++) {
     printf("\n[");
     for(int x2 = 0; x2 < floor(D2*scale_factors[1]); x2++) {
     printf("\n[");
     for(int x3 = 0; x3 < floor(D3*scale_factors[2]); x3++) {
     printf("\n[");
     for(int x4 = 0; x4 < floor(D4*scale_factors[3]); x4++) {
	     Xs[0] = x1;
	     Xs[1] = x2;
	     Xs[2] = x3;
	     Xs[3] = x4;
      float r = interpolate_nd_with_x(
	/*OMTensor */myT,
	/* n */4,
	/*float scale_factor*/ scale_factors,
	/*floats *x*/ Xs,
	/* Coeff_Func_t*/ nearest_coeffs,
	/*float */coeffs_buffer,
	/*int coeffs_n*/ COEFFS_N,
	/*float roi*/ 0.,
	/*float extrapolation_value*/ 0.,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0,
        /*mode */0);
       printf("%f ", r);
     }
     printf("]\n");
     }
     printf("]\n");
     }
     printf("]\n");
   }
   printf("]\n");
}

void test_scalar_up_2D() {
  const int  COEFFS_N = 2;
   float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};
  const int  D1 =  2;
  const int D2 = 3;
   float data[D1*D2];
   for(int i = 0; i< D1*D2; i++) {
	   data[i] = (i+1)*1.0; // backend test starts from 1
   }
   int64_t shape[] = {D1, D2};
   OMTensor *myT = omTensorCreate(data, shape, 2, 1);
   float scale_factors[] = {1, 2};
   float Xs[2];
   printf("[");
   for(int x1 = 0; x1 < floor(D1*scale_factors[0]); x1++) {
     printf("\n[");
     for(int x2 = 0; x2 < floor(D2*scale_factors[1]); x2++) {
	     Xs[0] = x1;
	     Xs[1] = x2;
      float r = interpolate_nd_with_x(
	/*OMTensor */myT,
	/* n */2,
	/*float scale_factor*/ scale_factors,
	/*floats *x*/ Xs,
	/* Coeff_Func_t*/ nearest_coeffs,
	/*float */coeffs_buffer,
	/*int coeffs_n*/ COEFFS_N,
	/*float roi*/ 0.,
	/*float extrapolation_value*/ 0.,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0,
        /*mode */0);
       printf("%f ", r);
     }
     printf("]\n");
   }
   printf("]\n");
}

void test_scalar1() {
  const int  COEFFS_N = 2;
   float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};
  const int  MAXN =  2;
   float data[MAXN];
   for(int i = 0; i< MAXN; i++) {
	   data[i] = (i+1)*1.0; // backend test starts from 1
   }
   int64_t shape[] = {MAXN};
   OMTensor *myT = omTensorCreate(data, shape, 1, 1);
   float scale_factor = 3;
   for(int x = 0; x < floor(MAXN*scale_factor); x++) {
   interpolate_1d_with_x(
	/*OMTensor */myT,
	/*float scale_factor*/scale_factor,
	/*float x*/ (float)x,
	/* Coeff_Func_t*/ nearest_coeffs,
	/*float */coeffs_buffer,
	/*int coeffs_n*/ COEFFS_N,
	/*float roi*/ 0.,
	/*float extrapolation_value*/ 0.,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0,
        /*mode */0);
   }
}

void test_scalar2() {
  const int COEFFS_N = 2;
  float coeffs_buffer[COEFFS_N]; // = {1.0, 0.};
  const int  MAXN = 4;
   float data[MAXN];
   for(int i = 0; i< MAXN; i++) {
	   data[i] = (i+1)*1.0; // backend test starts from 1
   }
   int64_t shape[] = {MAXN};
   OMTensor *myT = omTensorCreate(data, shape, 1, 1);
   float scale_factor = 0.6;
   for(int x = 0; x < floor(MAXN*scale_factor); x++) {
   interpolate_1d_with_x(
	/*OMTensor */myT,
	/*float scale_factor*/scale_factor,
	/*float x*/ (float)x,
	/* Coeff_Func_t*/ nearest_coeffs,
	/*float */coeffs_buffer,
	/*int coeffs_n*/ COEFFS_N,
	/*float roi*/ 0.,
	/*float extrapolation_value*/ 0.,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0,
        /*mode */0);
   }
}

void test_neighbor_idx(){
#define MAXM 10
   float data[MAXM];
   for(int i = 0; i< MAXM; i++) {
	   data[i] = (i)*1.0; // backend test starts from 1
   }
   int64_t n = 3;
   float x;
   printf("x = ");
   scanf("%f", &x);
   printf("n = ");
   scanf("%lld", &n);
   float points[3];
   get_neighbor(x,
	n,
	MAXM,
	data,
	points,
	0);
}

void Resize_Scales_Nearest(OMTensor *output, OMTensor *data, OMTensor *scales, char *mode_str) {
      interpolate_nd_OMTensor(
        /*OMTensor */ output,
	/*OMTensor */ data,
	/*mode*/ 0,
        /*OMTensor output size */ NULL,
        /*OMTensor scales */ scales,
	/* roi */ NULL,
	///* Coeff_Func_t*/ nearest_coeffs,
	///*float */coeffs_buffer,
	///*int coeffs_n*/ COEFFS_N,
	/*float * extrapolation_value*/ 0,
	/*int coordinate_transformation_mode*/0,
	/*exclude */0);
}

/*
int main() {
   test_2D_1();
}
*/

