#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h> 
#include <stddef.h>
#include <assert.h>

//THIS CODE IS ESSENTIALLY THE BASIS OF THE IMPLEMENTATION OF A NEURAL NETWORK USING THE METHOD OF FINITE DIFFERENCES TO FIND THE GRADIENT

//DEFINIG THE STRUCTURE OF A MATRIX AS AN ARRAY OF FLOATS , THE DIMENSIONS OF THE MATRIX 
//AND THE STRIDE IN CASE WE WANT TO EXTRACT SUBMATRICES
typedef struct {
    size_t rows   ;  
    size_t cols    ; 
    size_t stride ; 
    float  *es    ;  
}MAT ; 


//DEFINING THE STRUCTURE OF A NEURAL NETWORK AS THREE ARRAYS OF MATRICES (ACTIVATION "as" /  WEIGHTS "ws / BIASES "bs" )

typedef struct NN {
    size_t count ;
    MAT *as ;
    MAT *bs ; 
    MAT *ws ;  
}NN ; 

//MACRO DEFINITION THAT CAN MAKE THE CODE SIMPLER AND MORE READABLE 

#define MAT_AT(m , i , j ) (m).es[(i)*(m).cols + j ] 

#define INPUT_SIZE 2

#define OUTPUT_SIZE 1

#define training_size 4

#define MAT_PRINT(m)  print_Matrix(m , #m , 0) ; 

#define NN_PRINT(nn)  nn_print (nn , #nn) ; 


// "arch" REFERS TO THE ARCHITECTURE OF THE NEURAL NETWORK , THE ith ELEMENT OF THE LIST REFERS TO THE NUMBER OF NEURONS IN THE ith LAYER
size_t arch[] = {INPUT_SIZE , 2 , OUTPUT_SIZE } ; 

//"arch_count" IS THE SIZE OF "arch" <-> IT IS THE NUMBER OF LAYERS IN THE NEURAL NETWORK 
size_t arch_count = sizeof (arch)/ sizeof (arch[0]) ; 
 

//"alpha" <->HYPER_PARAMETER REFERS TO THE LEARNING RATE OF THE NEURAL NETWORK 
float alpha = 1 ; 


float eps = 0.0001 ; 

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


//=============================FUNCTIONS_TO===================================
//------------------------------MANIPULATE-------------------------------------
//===============================MATRICES======================================





//MAT_FILL HAS THE ROLE OF FILLING A MATRICE WITH RANDOM FLOATS BETWEEN 0 AND 1 
float randf(){
    return (float)rand ()/RAND_MAX ; 
}

void MAT_FILL(MAT a ){
    for (int i = 0 ; i < a.rows ; i ++ ){
        for (int j = 0 ; j < a.cols ; j ++ ){
            MAT_AT(a , i ,j ) = randf()  ; 
        }
    }
}

void MAT_ZERO (MAT A ){
    for (int i = 0 ; i < A.rows ; i ++ ){
        for (int j = 0 ; j < A.cols ; j ++ ){
            MAT_AT(A , i , j ) = 0 ; 
        }
   }
}

//MAT_SUM <-> a <- a+b 
void MAT_SUM (MAT a , MAT b ){
    assert(a.rows== b.rows ) ; 
    assert(a.cols== b.cols) ; 

    for (int i = 0 ; i < a.rows ; i ++ ){
        for (int j = 0 ; j < a.cols ; j ++ ){
            MAT_AT(a , i , j ) +=  MAT_AT(b , i ,  j ) ; 
        }
    }
}


//MAT_PRODUCT <-> DST = A . B 
void MAT_PRODUCT(MAT dst , MAT a , MAT b  ){
    assert (dst.rows == a.rows); 
    assert (dst.cols==b.cols) ; 
    assert (a.cols == b.rows) ; 
    for (int i = 0 ; i < dst.rows ; i ++ ){
        for (int j = 0 ; j < dst.cols ; j ++ ){
            MAT_AT (dst , i ,j ) = 0 ; 
            for (int k = 0 ; k < b.rows ; k ++ ){
                MAT_AT(dst,i,j) += MAT_AT(a , i , k ) * MAT_AT(b , k , j ) ; 
            }
        }
    }
}



//WE WILL BE USING INSTEAD THE MACRO MAT_PRINT(MAT M) THAT WILL REFER TO THE FUNCTION print_Matrix 
//YOU CAN SEE THE IMPLEMENTATION IN THE EXEMPLES.C FILE "IF IT'S STILL NOT THERE IT MEANS I AM STILL WORKING ON THEM :)" 
void print_Matrix(MAT m , char* name , int padding  ){
    
    printf("%*s%s = \n" , (int)padding , "" , name) ; 
    for (int i = 0 ; i < m.rows ; i ++ ){
        printf("%*s|" , (int)padding , "") ; 
        for (int j = 0 ; j < m.cols ; j ++ ){
            
            printf("%*s  %f    " ,(int)padding , "",  MAT_AT (m , i , j )) ; 
        }
        printf("|  \n") ; 
    }
    
    printf("%*s\n",(int)padding , "" ) ; 

}





//MAT_SIGMOID <-> "a an element of MAT m -> a = sigmoid (a)"
float sigmoidf(float x ){
    return (float)1/(float) (1 + exp (-x)) ; 
}
void MAT_SIGMOID (MAT m){
    for (int i = 0 ; i < m.rows ; i ++  ){
        for (int j = 0 ; j < m.cols ; j ++ ){
            MAT_AT(m , i , j) = sigmoidf(MAT_AT(m , i , j )) ; 
        }
    }
}


//MAT_COPY <-> COPIES A IN DST 
void MAT_COPY (MAT dst , MAT a ){
    assert (dst.cols == a.cols) ; 
    assert (dst.rows == a.rows ) ;
    for (int i = 0 ; i < dst.rows ; i ++ ){
        for (int j = 0 ; j < dst.cols ; j++ ){
            MAT_AT(dst , i , j ) = MAT_AT (a , i , j ); 
        }
    }
}

//MAT_ALLOC <-> ALLOCATES MEAMORY TO THE MATRICES 
MAT MAT_ALLOC (int rows , int cols ){
    MAT m ; 
    m.rows = rows ;
    m.cols = cols ; 
    m.es = (float *)malloc (sizeof (m.es) * rows * cols )  ;
    return m ; 
}



//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


//=============================FUNCTIONS_TO==================================
//----------------------------MANIPULATE_THE---------------------------------
//============================NEURAL_NETWORK=================================


//NN_OUTPUT <-> RETURN THE OUTPUT LAYER OF THE NEURAL NETWORK NN 
MAT NN_OUTPUT (NN nn ){
    return nn.as[(nn.count)] ; 
}


//NN_INPUT <-> RETURN THE INPUT LAYER OF THE NEURAL NETWORK 
MAT NN_INPUT (NN nn ){
    return nn.as[0] ; 
} 

//NN_ROW copies the rowth row in dst 
void MAT_ROW (MAT dst , MAT src , int row ){
    for (int i = 0 ; i < src.cols  ;i ++ ){
        MAT_AT(dst , 0 , i ) = MAT_AT (src , row ,  i ) ;  
    }
}


//NN_ZERO <-> FILLS THE MATRICES OF THE NEURAL NETWORK WITH 0s
void NN_ZERO (NN nn){
    for (int i = 0 ;  i < nn.count ; i ++){
        
            MAT_ZERO (nn.ws[i]) ; 
            MAT_ZERO (nn.bs[i]) ; 
        
            MAT_ZERO(nn.as[i]) ; 
    }
    MAT_ZERO(NN_OUTPUT(nn)) ; 
}
//NN_ALLOC <-> ALLOCATES MEMORY TO THE NEURAL NETWORK 
NN NN_ALLOC(size_t *arch , size_t arch_count ){
    assert (arch_count > 0) ; 
    NN nn ; 
    nn.count = arch_count -1 ; 
    nn.as =(MAT*) malloc (sizeof (MAT) * (nn.count +1) ) ; 
    nn.bs =(MAT*)malloc (sizeof(MAT) * nn.count ) ; 
    nn.ws =(MAT*)malloc (sizeof (MAT)*nn.count) ; 
    nn.as[0] = MAT_ALLOC(1 , arch[0] ) ;
    for (int i = 1 ; i < arch_count ; i ++ ) {
        nn.ws[i-1] = MAT_ALLOC(nn.as[i-1].cols , arch[i]) ; 
        nn.bs[i-1] = MAT_ALLOC(1, arch[i]) ; 
        nn.as[i]   = MAT_ALLOC(1 , arch[i]) ; 
        MAT_FILL(nn.as[i]) ; 
        MAT_FILL(nn.ws[i-1]) ; 
        MAT_FILL(nn.bs[i-1]) ; 


    }
    return nn ;  
}


//INSTEAD WE WILL BE USING THE MACRO DEFINED NN_PRINT(MAT M) WHICH WILL REFER TO nn_print 
//YOU CAN SEE THE IMPLEMENTATION IN THE EXEMPLES.C FILE "IF IT'S STILL NOT THERE IT MEANS I AM STILL WORKING ON THEM :)" 
void nn_print(NN nn , const char* name ){
    printf("NEURAL NETWORK %s = [\n" , name ) ;
    char buffer[256] ; 

    MAT* w = nn.ws ; 
    MAT* b = nn.bs  ; 
    printf("%s = [ \n" , name ) ; 
    for (size_t i = 0 ; i <= nn.count ; i++ ){
       
        snprintf(buffer , sizeof (buffer) , "as[%zu]" , i); 
        print_Matrix(nn.as[i] , buffer , 4 ) ; 
        if (i < nn.count ){
            snprintf(buffer , sizeof (buffer ) , "ws[%zu] : " , i ) ; 
            print_Matrix(w[i] , buffer , 4 ) ; 
            snprintf(buffer , sizeof (buffer) , "bs[%zu]" , i ); 
            print_Matrix(b[i] , buffer , 4 ) ; 
        }
        
        

    }
    printf("]\n===========================================================\n") ; 
}



//NN_RAND <-> FILLS THE MATRICES OF THE NEURAL NETWORK WITH RANDOM FLOATS VARYING BETWEEN 0 AND 1 
void NN_RAND (NN nn ){
    for (int i = 0 ; i < nn.count  ; i ++ ){
        MAT_FILL(nn.ws[i]) ; 
        MAT_FILL(nn.bs[i]) ; 
    }
}
//NN_FORWARD <-> COMPUTES THE FORWARD PROPAGATION OF THE NEURAL NETWORK 
void NN_FORWARD (NN m ){
for (int i = 0 ; i < m.count ; ++i ){
       MAT_PRODUCT(m.as[i+1] , m.as[i] , m.ws[i]) ; 
       MAT_SUM (m.as[i+1] , m.bs[i]) ; 
        MAT_SIGMOID(m.as[i+1]) ;
       
       
    }
}



//NN_COST <-> COMPUTES THE COST OF A NEURAL NETWORK GIVEN INPUT X AND EXPECTED OUTPUT Y 

 float NN_COST(NN nn, MAT ti, MAT to , MAT x , MAT y ){
    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0; i < n; ++i) {
        MAT_ROW (x, ti, i);
        MAT_ROW (y , to, i);

        MAT_COPY(NN_INPUT(nn), x);
        NN_FORWARD(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;
        }
    }

    return c/n;
}

//NN_BACK_PROP <-> uses back propagation algorithm to find the gradient matrices of the neural network nn in g 
void NN_BACK_PROP(NN nn, NN g, MAT ti, MAT to){
    assert(ti.rows == to.rows);
    size_t n = ti.rows;
    assert(NN_OUTPUT(nn).cols == to.cols);

    NN_ZERO(g);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i) {
    MAT_ROW(NN_INPUT(nn), ti, i);
    NN_FORWARD(nn);

    for (size_t j = 0; j <= nn.count; ++j) {
        MAT_ZERO(g.as[j]);
    }

    for (size_t j = 0; j < to.cols; ++j) {
        MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
    }

    for (size_t l = nn.count; l > 0; --l) {
        for (size_t j = 0; j < nn.as[l].cols; ++j) {
            float a = MAT_AT(nn.as[l], 0, j);
            float da = MAT_AT(g.as[l], 0, j);
            MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1 - a);
            for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
                // j - weight matrix col
                // k - weight matrix row
                float pa = MAT_AT(nn.as[l-1], 0, k);
                float w = MAT_AT(nn.ws[l-1], k, j);
                MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1 - a)*pa;
                MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1 - a)*w;
            }
        }
    }
    }
    }


//NN_LEARN <-> CHANGES THE NEURAL NETWORK'S PARAMETERS (W[] AND B[] ) TO REDUCE THE COST FUNCTION
void NN_LEARN(NN nn , NN g , float alpha  , MAT X , MAT Y){
    for (int i = 0 ; i <nn.count ; i ++ ){
        for(int j = 0 ; j  < nn.ws[i].rows ; j ++ ){
            for (int k = 0 ; k  < nn.ws[i].cols ; k ++ ){
                MAT_AT( nn.ws[i] , j , k ) += -alpha * MAT_AT(g.ws[i] , j , k )  ; 
            }
        }
        for (int j = 0 ; j < nn.bs[i].rows ; j ++ ){
            for (int k = 0 ; k < nn.bs[i].cols ; k ++ ){
                MAT_AT( nn.bs[i] , j , k ) +=  -alpha * MAT_AT(g.bs[i] , j , k ); 
            }
        }
    }
    
}

int main(void ){
    float f1[] ={
        0 , 0, 
        0 , 1 , 
        1 , 0 , 
        1 , 1, 
    } ; 
    float f2[] = {
        0 , 1 , 1 , 1
    } ; 
    srand(time(0)) ; 
    MAT X ={.rows = training_size , .cols = INPUT_SIZE , .es = f1 } ; 
    MAT Y ={.rows = training_size , .cols = OUTPUT_SIZE , .es = f2} ; 
    MAT ti ={.rows = 1 , .cols = INPUT_SIZE , .es = (float*) malloc (sizeof (float)*INPUT_SIZE)} ; 
    MAT to ={.rows = 1 , .cols = INPUT_SIZE , .es = (float*) malloc (sizeof(float)*OUTPUT_SIZE)} ; 
    NN g = NN_ALLOC (arch , arch_count) ; 
    NN nn = NN_ALLOC (arch , arch_count ) ;
    NN_RAND(g) ;  
    NN_RAND(nn) ; 
    NN_PRINT(nn) ; 
    for (int i = 0; i < 5000; i ++ ){
        NN_BACK_PROP(nn , g , X , Y ) ; 
        NN_LEARN(nn , g , alpha , X , Y) ; 
        printf("c : %f\n" , NN_COST(nn , X , Y , ti , to )) ; 
    }
    float f1_test [] = {0 , 0 } ; 
    float f2_test [] = {0} ; 
    MAT X_test  = {.rows = 1 , .cols = INPUT_SIZE , .es = f1_test   } ; 
    MAT Y_test  ={.rows = 1 , .cols = OUTPUT_SIZE , .es = f2_test  } ; 


    
   
 
    
}