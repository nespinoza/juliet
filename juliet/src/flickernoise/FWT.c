#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/time.h>
#define ARRAYD(p) ((double *) (((PyArrayObject *)p)->data)) 

/* 
 *                                [INITIALIZATION]
 * ------------------ PROTOTYPES FOR FUNCTIONS AND EXTERNAL VARIABLES -----------------------
 *
 */

double** MakeArray(int rows, int columns);                           /* Function that makes/allocates an array of pointers    */
int** MakeIntArray(int rows, int columns);
int* MakeIntVector(int nelements);
double* MakeVector(int nelements);                                   /* Function that makes/allocates a pointer in memory     */
void FreeArray(double** theArray,int rows);                          /* Function that frees an array of pointers from memory  */
void FreeIntArray(int** theArray,int rows);
double** Transpose(double** O,int rowsIN, int colsIN);
void Convolve(double* input,double* c,double* c2,double *output_aproximation,double *output_details,int len_input,int len_c);
void IConvolve(double* input,double* c,double* c2,double *output,int len_input,int len_c);
void UnPermutation(double* data,double *wavelet,double *output,int i);
/*
                 [INITIALIZATION OF A METHOD]
                 
                 getWC = get Wavelet Coefficients
*/

static PyObject *FWT_getWC(PyObject *self, PyObject *args){
	double *data,*coefficients;
        int data_len,coefficients_len,M,i,j,len_rows,len_cols;
	PyObject *dataarray,*coefficientsarray;	
	PyArg_ParseTuple(args,"OOiii",&dataarray,&coefficientsarray,&data_len,&coefficients_len,&M);
        data = ARRAYD(dataarray);
	coefficients = ARRAYD(coefficientsarray);
// 	printf("%f, %f, coeffs: %f, %f...\n",data[0],data[1],coefficients[0],coefficients[1]);
	len_cols=data_len/2;
	len_rows=M;
	double** WT = MakeArray(len_rows,len_cols); // FREED
	double* v = MakeVector(data_len/2); // FREED
	double* v2 = MakeVector(data_len/2); // FREED
        double* c2 = MakeVector(coefficients_len);
        c2[0]=coefficients[3];             // We make the high pass filter.
        c2[1]=-coefficients[2];
        c2[2]=coefficients[1];
        c2[3]=-coefficients[0];
//        printf("Wavelet coefficients used: %f,%f,%f and %f \n",coefficients[0],coefficients[1],coefficients[2],coefficients[3]);
//        printf("Scaling Coeff. used: %f, %f, %f and %f \n",c2[0],c2[1],c2[2],c2[3]);
        for(i=0;i<M;i++){
	  Convolve(data,coefficients,c2,&v[0],&v2[0],data_len,coefficients_len);
	  for(j=0;j<data_len/2;j++){
	    WT[i][j]=v2[j];
	    v2[j]=0.0;
	    data[j]=v[j];    // I pass the aproximation coefficients to the data vector, to be filtered.
	    if((data_len/2)!=2)
               v[j]=0.0;
	  }
	  if(data_len/2!=2){
            data_len=data_len/2;
	  }
	  else{
	    break;
	  }
	}
	free(c2);
/* End of the matrix-to-vector conversion part of the code. Everything's ok up to here...now I have my theArray[i][j] array (matrix) ready to be called...*/
        double* theArray;  // NOT FREED
//        theArray = (double*) malloc((len_cols*len_rows)*sizeof(double));
//        for(i=0;i<len_rows;i++){
//           for(j=0;j<len_cols;j++){
//               theArray[i*len_cols+j]=WT[i][j];
//            }
//        }
        int max_j=0,counter=0,flen=0;
        for(i=0;i<(len_rows);i++){
           flen=flen+pow(2,i);
        }
        flen=flen-1;
        theArray = (double*) malloc((flen)*sizeof(double));
        for(i=1;i<(len_rows);i++){
             max_j=pow(2,i);
             for(j=0;j<max_j;j++){
                theArray[counter]=WT[len_rows-i-1][j];
                counter++;
             }
        }
//        for(i=0;i<(len_rows-1);i++){
//             for(j=0;j<len_cols;j++){
//                printf(" %f ",WT[i][j]);
//             }
//             printf("\n");
//        }
        free(v2);
        FreeArray(WT,len_rows);
/* Finally, we create a Python "Object" List that contains the WT coefficients and return it back to Python */
        
        // PyObject *lst = PyList_New(len_rows*len_cols);
        PyObject *lst = PyList_New(flen);
        PyObject *lst2 = PyList_New(2);
        PyObject *num;
	PyObject *num2;
        if (!lst)
           return NULL;
       // for (i = 0; i < len_rows*len_cols; i++) {
        for (i = 0; i < flen; i++){
	  if(i==0 || i==1){
            num2=PyFloat_FromDouble(v[i]);
            if (!num2) {
              Py_DECREF(lst2);
              return NULL;
            }
            PyList_SET_ITEM(lst2, i, num2);    
	  }
          num=PyFloat_FromDouble(theArray[i]);
          if (!num) {
            Py_DECREF(lst);
            return NULL;
          }
          PyList_SET_ITEM(lst, i, num);
        }
        free(theArray);
	free(v);
        PyObject *MyResult = Py_BuildValue("OO",lst,lst2);
        Py_DECREF(lst);
	Py_DECREF(lst2);
        return MyResult;
}

static PyObject *FWT_getSignal(PyObject *self, PyObject *args){
	double *data,*coefficients;
        int data_len,coefficients_len,M,i,len_input,len;
	PyObject *dataarray,*coefficientsarray;	
	PyArg_ParseTuple(args,"OOiii",&dataarray,&coefficientsarray,&data_len,&coefficients_len,&M);
        data = ARRAYD(dataarray);
	coefficients = ARRAYD(coefficientsarray);
// 	printf("%f, %f, coeffs: %f, %f...\n",data[0],data[1],coefficients[0],coefficients[1]);
	len_input=4;
	len=pow(2,M);
	double* v = MakeVector(len); // FREED
        double* input = MakeVector(len); // FREED
        double* c2 = MakeVector(coefficients_len); //FREED
	v[0]=data[0];
	v[1]=data[1];
        c2[0]=coefficients[3];             // We make the high pass filter.
        c2[1]=-coefficients[2];
        c2[2]=coefficients[1];
        c2[3]=-coefficients[0];
        for(i=0;i<(M-1);i++){
  	  len_input=pow(2,i+2);
	  UnPermutation(data,&v[0],&input[0],i);
	  IConvolve(input,coefficients,c2,&v[0],len_input,coefficients_len);
	}
	free(c2);
	free(input);
/* End of the matrix-to-vector conversion part of the code. Everything's ok up to here...now I have my theArray[i][j] array (matrix) ready to be called...*/
//        theArray = (double*) malloc((len_cols*len_rows)*sizeof(double));
//        for(i=0;i<len_rows;i++){
//           for(j=0;j<len_cols;j++){
//               theArray[i*len_cols+j]=WT[i][j];
//            }
//        }
//        for(i=0;i<(len_rows-1);i++){
//             for(j=0;j<len_cols;j++){
//                printf(" %f ",WT[i][j]);
//             }
//             printf("\n");
//        }
/* Finally, we create a Python "Object" List that contains the WT coefficients and return it back to Python */
        
        // PyObject *lst = PyList_New(len_rows*len_cols);
        PyObject *lst = PyList_New(len);
        PyObject *num;
        if (!lst)
           return NULL;
       // for (i = 0; i < len_rows*len_cols; i++) {
        for (i = 0; i < len; i++){
          num=PyFloat_FromDouble(v[i]);
          if (!num) {
            Py_DECREF(lst);
            return NULL;
          }
          PyList_SET_ITEM(lst, i, num);
        }
	free(v);
        PyObject *MyResult = Py_BuildValue("O",lst);
        Py_DECREF(lst);
        return MyResult;
}

static PyMethodDef FWTMethods[] = {
	{"getWC", FWT_getWC, METH_VARARGS, "Obtention of the aproximation and detail coefficients of the WT."},
	{"getSignal", FWT_getSignal, METH_VARARGS, "Given aproximation and detail coeffs, we get the signal back (IWT)."},
	{NULL, NULL, 0, NULL}
};

void initFWT(void){
	(void) Py_InitModule("FWT", FWTMethods);
}


/*********************************************************************
 *          [START OF THE FUNCTIONS OF THE WT ALGORITHM]             *
 *********************************************************************
 */

void UnPermutation(double* data,double *wavelet,double *output,int i){
  int j,max_j=pow(2,i+2),middle=max_j/2;
  for(j=0;j<middle;j++){
      output[2*j]=wavelet[j];
      wavelet[j]=0.0;
      output[2*j+1]=data[middle+j];
  }
}

void Convolve(double* input,double* c,double* c2,double *output_aproximation,double *output_details,int len_input,int len_c){
  int i,k,j=0,max_len=(len_input/2)-1; // The max_len is len_input/2-1 because the last WC's are special ones.
  for(j=0;j<max_len;j++){
      i=2*j;
      for(k=0;k<len_c;k++){
        output_aproximation[j]=output_aproximation[j]+input[i+k]*c[k]; // Low-pass filtered data (aproximations).
        output_details[j]=output_details[j]+input[i+k]*c2[k]; // High pass filtered data (details).
      }
//      printf("%f, %f\n",output_aproximation[j],output_details[j]);
  }
  for(k=0;k<2;k++){
    output_aproximation[max_len]=output_aproximation[max_len]+input[max_len*2+k]*c[k]; // Low-pass filtered data (aproximations).
    output_details[max_len]=output_details[max_len]+input[max_len*2+k]*c2[k]; // High pass filtered data (details).
  }  
  output_aproximation[max_len]=output_aproximation[max_len]+input[0]*c[2]+input[1]*c[3];
  output_details[max_len]=output_details[max_len]+input[0]*c2[2]+input[1]*c2[3];
}

void IConvolve(double* input,double* c,double* c2,double *output,int len_input,int len_c){
  int i,k,j=0,max_len=(len_input/2); // The max_len is len_input/2.
  for(k=0;k<2;k++){
    output[0]=output[0]+input[(max_len-1)*2+k]*c[k]; // Low-pass filtered data (aproximations).
    output[1]=output[1]+input[(max_len-1)*2+k]*c2[k]; // High pass filtered data (details).
  }   
  output[0]=output[0]+input[0]*c[2]+input[1]*c[3];
  output[1]=output[1]+input[0]*c2[2]+input[1]*c2[3];  
  for(j=1;j<max_len;j++){
      i=2*(j-1);
      for(k=0;k<len_c;k++){
        output[2*j]=output[2*j]+input[i+k]*c[k]; // Low-pass filtered data (aproximations).
        output[2*j+1]=output[2*j+1]+input[i+k]*c2[k]; // High pass filtered data (details).
      }
//      printf("%f, %f\n",output_aproximation[j],output_details[j]);
  }
}

double** Transpose(double **O,int rowsIN,int colsIN){
  int i,j;
  double **OT=MakeArray(colsIN,rowsIN);
  for(i=0;i<rowsIN;i++){
     for(j=0;j<colsIN;j++){
        OT[j][i]=O[i][j];
     }
  }
  FreeArray(O,rowsIN);
  return OT;
}


int** MakeIntArray(int rows, int columns){
  int i,j;
  int** theArray;
  theArray = (int**) malloc(rows*sizeof(int*));
  for(i=0;i<rows;i++)
     theArray[i] = (int*) malloc(columns*sizeof(int));

/* Fill the array with zeroes (i.e. we clean it) */

  for(i=0;i<rows;i++){
     for(j=0;j<columns;j++){
       theArray[i][j]=0;
     }
  }

  return theArray;
}

double** MakeArray(int rows, int columns){
  int i,j;
  double** theArray;
  theArray = (double**) malloc(rows*sizeof(double*));
  for(i=0;i<rows;i++)
      theArray[i] = (double*) malloc(columns*sizeof(double));

/* Fill the array with zeroes (i.e. we clean it) */

  for(i=0;i<rows;i++){
     for(j=0;j<columns;j++){
       theArray[i][j]=0.0;
     }
  }

  return theArray;
}

double* MakeVector(int nelements){
  double* Vector;
  int j;
  Vector = (double*) malloc(nelements*sizeof(double));

  for(j=0;j<nelements;j++){
       Vector[j]=0.0;
  }
  return Vector;
}

int* MakeIntVector(int nelements){
  int* Vector;
  int j;
  Vector = (int*) malloc(nelements*sizeof(int));

  for(j=0;j<nelements;j++){
       Vector[j]=0;
  }
   return Vector;
}

void FreeArray(double** theArray,int rows){
  int i;
  for(i=0;i<rows;i++){
     free(theArray[i]);
  }
  free(theArray);
}

void FreeIntArray(int** theArray,int rows){
  int i;
  for(i=0;i<rows;i++){
     free(theArray[i]);
  }
  free(theArray);
}
