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
size_t arch[] = {INPUT_SIZE , 8, OUTPUT_SIZE } ; 

//"arch_count" IS THE SIZE OF "arch" <-> IT IS THE NUMBER OF LAYERS IN THE NEURAL NETWORK 
size_t arch_count = sizeof (arch)/ sizeof (arch[0]) ; 

//"eps" IS A CONSTANT THAT WE WILL BE USING TO FIND THE GRADIENT OF THE COST FUNCTION USING THE FINITE DIFFERENTIATION METHOD 
float eps =  1e-1 ; 

//"alpha" <->HYPER_PARAMETER REFERS TO THE LEARNING RATE OF THE NEURAL NETWORK 
float alpha = 1e-1 ; 


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


//NN_ALLOC <-> ALLOCATES MEMORY TO THE NEURAL NETWORK 
NN NN_ALLOC(size_t *arch , size_t arch_count ){
    assert (arch_count > 0) ; 
    NN nn ; 
    nn.count = arch_count -1 ; 
    nn.as =(MAT*) malloc (sizeof (MAT) * (nn.count +1) ) ; 
    nn.bs =(MAT*)malloc (sizeof(MAT) * nn.count ) ; 
    nn.ws =(MAT*)malloc (sizeof (MAT)*nn.count) ; 
    nn.as[0] = MAT_ALLOC(training_size , arch[0] ) ;
    for (int i = 1 ; i < arch_count ; i ++ ) {
        nn.ws[i-1] = MAT_ALLOC(nn.as[i-1].cols , arch[i]) ; 
        nn.bs[i-1] = MAT_ALLOC(training_size , arch[i]) ; 
        nn.as[i]   = MAT_ALLOC(training_size , arch[i]) ; 
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
float NN_COST (NN nn , MAT X , MAT Y ){
    assert(X.rows == Y.rows ) ;
    assert(NN_OUTPUT(nn).rows == Y.rows ) ; 
    assert(NN_OUTPUT(nn).cols == Y.cols ) ;  
    float cost = 0 ; 
    float diff ; 

    NN_FORWARD(nn) ; 
    
     
    for (int i = 0 ; i < NN_OUTPUT(nn).rows ; i ++){ 
        for (int j = 0 ; j < NN_OUTPUT(nn).cols ; j ++ ){
            diff =  MAT_AT(NN_OUTPUT(nn) , i , j ) - MAT_AT(Y , i , j ) ;              
            cost +=  diff * diff ; 
        }

    }
    return cost / (OUTPUT_SIZE * training_size)    ; 
}


//NN_GRADIENT <-> COMPUTES THE GRADIENT MATRICES OF THE NEURAL NETWORK USING FINITE DIFFERENTIATION
void NN_GRADIENT (NN nn , NN g , MAT X , MAT Y ){
    float saved ; 
    float c = NN_COST(nn , X , Y ) ;
    for (int i = 0 ; i <nn.count ; i ++ ){
        for(int j = 0 ; j  < (nn.ws[i]).rows ; j ++ ){
            for (int k = 0 ; k  < (nn.ws[i]).cols ; k ++ ){
                saved = MAT_AT( nn.ws[i] , j , k ) ; 
                MAT_AT( nn.ws[i] , j , k ) = MAT_AT( nn.ws[i] , j , k )+ eps ; 
         
                MAT_AT((g.ws[i]) , j , k ) = (NN_COST(nn , X , Y ) - c) /eps  ; 
                MAT_AT(nn.ws[i] , j , k ) = saved ; 
            }
        }
        for (int j = 0 ; j < nn.bs[i].rows ; j ++ ){
            for (int k = 0 ; k < nn.bs[i].cols ; k ++ ){
                saved = MAT_AT( nn.bs[i] , j , k ) ; 
                MAT_AT( nn.bs[i] , j , k ) += eps ; 
                MAT_AT((g).bs[i] , j , k ) = (NN_COST(nn , X , Y )-c)/eps ; 
                MAT_AT( nn.bs[i] , j , k ) = saved ; 
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


