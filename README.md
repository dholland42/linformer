# linformer

This is a test implementation of the linear attention as described in the
[linformer paper](https://arxiv.org/pdf/2006.04768.pdf). The goal was to take
the current [`MultiHeadAttention` layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention)
implementation and make the smallest change possible to get it working.

The basic idea is that we set up [2 additional projections](https://github.com/dholland42/linformer/blob/main/linformer/attention.py#L374-L385)
that will transform the sequence dimension in the key and value array. Then we perform the projection
[here](https://github.com/dholland42/linformer/blob/main/linformer/attention.py#L523-L527) to avoid the
`O(sequence_length^2)` operation in the attention computation.

Random thought process can be found in [this notebook](./understanding.ipynb).

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
1/1 [==============================] - 1s 738ms/step
SEQUENCE LENGTH: 2000
BATCH SIZE:      25
EMBEDDING DIM:   256
PROJECTION DIM:  64
NUM HEADS:       8
HIDDEN DIM:      128
TIME:            0.775 seconds
```

Different configurations can be played around with via the `entrypoint`.

```
Usage: entrypoint [OPTIONS]

  Run a prediction time experiment.

Options:
  --sequence-length INTEGER       [default: 2000]
  --batch-size INTEGER            [default: 25]
  --embedding-dim INTEGER         [default: 256]
  --projection-dim INTEGER        [default: 64]
  --num-heads INTEGER             [default: 8]
  --hidden-dim INTEGER            [default: 128]
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.
```