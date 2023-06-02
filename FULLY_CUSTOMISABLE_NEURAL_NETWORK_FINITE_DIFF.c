#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h> 
#include <stddef.h>
#include <assert.h>

typedef struct {
    size_t rows   ;  
    size_t cols    ; 
    size_t stride ; 
    float  *es    ;  
}MAT ; 

typedef struct NN {
    size_t count ;
    MAT *as ;
    MAT *bs ; 
    MAT *ws ;  
}NN ; 

#define MAT_AT(m , i , j ) (m).es[(i)*(m).cols + j ] 

#define INPUT_SIZE 2

#define OUTPUT_SIZE 1

#define training_size 4

#define MAT_PRINT(m)  print_Matrix(m , #m , 0) ; 

#define NN_PRINT(nn)  nn_print (nn , #nn) ; 

size_t arch[] = {INPUT_SIZE , 8, OUTPUT_SIZE } ; 

size_t arch_count = sizeof (arch)/ sizeof (arch[0]) ; 

float eps =  1e-1 ; 

float alpha = 1e-1 ; 

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

void MAT_SUM (MAT a , MAT b ){
    assert(a.rows== b.rows ) ; 
    assert(a.cols== b.cols) ; 

    for (int i = 0 ; i < a.rows ; i ++ ){
        for (int j = 0 ; j < a.cols ; j ++ ){
            MAT_AT(a , i , j ) +=  MAT_AT(b , i ,  j ) ; 
        }
    }
}

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


void MAT_COPY (MAT dst , MAT a ){
    assert (dst.cols == a.cols) ; 
    assert (dst.rows == a.rows ) ;
    for (int i = 0 ; i < dst.rows ; i ++ ){
        for (int j = 0 ; j < dst.cols ; j++ ){
            MAT_AT(dst , i , j ) = MAT_AT (a , i , j ); 
        }
    }
}

MAT MAT_ALLOC (int rows , int cols ){
    MAT m ; 
    m.rows = rows ;
    m.cols = cols ; 
    m.es = (float *)malloc (sizeof (m.es) * rows * cols )  ;
    return m ; 
}
MAT NN_OUTPUT (NN nn ){
    return nn.as[(nn.count)] ; 
}


MAT NN_INPUT (NN nn ){
    return nn.as[0] ; 
} 

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


void NN_RAND (NN nn ){
    for (int i = 0 ; i < nn.count  ; i ++ ){
        MAT_FILL(nn.ws[i]) ; 
        MAT_FILL(nn.bs[i]) ; 
    }
}

void NN_FORWARD (NN m ){
    for (int i = 0 ; i < m.count ; ++i ){
       MAT_PRODUCT(m.as[i+1] , m.as[i] , m.ws[i]) ; 
       MAT_SUM (m.as[i+1] , m.bs[i]) ; 
        MAT_SIGMOID(m.as[i+1]) ;
       
       
    }
}


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
        0 , 1 , 1 , 0
    } ; 

    MAT X ={.rows = training_size , .cols = INPUT_SIZE , .es = f1 } ; 
    MAT Y ={.rows = training_size , .cols = OUTPUT_SIZE , .es = f2} ; 
    NN g = NN_ALLOC (arch , arch_count) ; 
    NN nn = NN_ALLOC (arch , arch_count ) ;
    NN_RAND(g) ;  
    NN_RAND(nn) ; 
    nn.as[0] = X ; 
    
    for (int i = 0; i < 100000*4 ; i ++ ){
        NN_GRADIENT(nn , g , X , Y ) ; 
        NN_LEARN(nn ,g , alpha , X , Y ) ; 
        printf("%f  \n" , NN_COST(nn , X , Y )) ; 
    }
    MAT_PRINT(Y) ; 
    MAT_PRINT(nn.as[nn.count]) ; 
    NN_PRINT(nn) ; 
   


    
}