# linformer

This is a test implementation of the linear attention as described in the
[linformer paper](https://arxiv.org/pdf/2006.04768.pdf). The goal was to take
the current [`MultiHeadAttention` layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention)
implementation and make the smallest change possible to get it working.

## Basic Results

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 2000)]       0           []                               
                                                                                                  
 embedding (Embedding)          (None, 2000, 256)    25600       ['input_1[0][0]']                
                                                                                                  
 linear_multi_head_attention (L  (None, 2000, 256)   1179904     ['embedding[0][0]',              
 inearMultiHeadAttention)                                         'embedding[0][0]']              
                                                                                                  
==================================================================================================
Total params: 1,205,504
Trainable params: 1,205,504
Non-trainable params: 0
__________________________________________________________________________________________________
1/1 [==============================] - 1s 752ms/step
SEQUENCE LENGTH: 2000
BATCH SIZE:      25
INFERENCE TIME:  0.7883796691894531
```
