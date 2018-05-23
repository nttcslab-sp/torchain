# PyTorch FFI package with Kaldi and ATen

``` console
# with pytorch 0.4.0
$ source /data/work49/skarita/tool/miniconda3/bin/activate torch
# or with pytorch 0.3.1
$ source /data/work70/skarita/exp/chime5/venv/bin/activate

$ make test KALDI_ROOT=/data/work70/skarita/exp/chime5/kaldi-22fbdd
```

then you will see stdout like this

``` console
0 ChainResults(loss=1523440.000000, objf=-65544.968750, l2_term=-1523429.125000, weight=6016.000000)
1 ChainResults(loss=90851.000000, objf=-11141.706055, l2_term=-90848.437500, weight=4352.000000)
2 ChainResults(loss=1633.794800, objf=-5968.860352, l2_term=-1632.802612, weight=6016.000000)
3 ChainResults(loss=824.511902, objf=-6661.337402, l2_term=-823.548157, weight=6912.000000) 
4 ChainResults(loss=411.817657, objf=-4280.249023, l2_term=-410.834137, weight=4352.000000) 
5 ChainResults(loss=408.518799, objf=-5801.333496, l2_term=-407.554474, weight=6016.000000) 
6 ChainResults(loss=233.465958, objf=-4343.767090, l2_term=-232.467850, weight=4352.000000) 
7 ChainResults(loss=243.598633, objf=-4366.199707, l2_term=-242.595367, weight=4352.000000) 
8 ChainResults(loss=387.405365, objf=-5805.361816, l2_term=-386.440369, weight=6016.000000) 
9 ChainResults(loss=253.922623, objf=-4391.189453, l2_term=-252.913620, weight=4352.000000) 
10 ChainResults(loss=389.562714, objf=-6743.458496, l2_term=-388.587097, weight=6912.000000)
11 ChainResults(loss=271.677155, objf=-5866.956543, l2_term=-270.701935, weight=6016.000000)
12 ChainResults(loss=203.969925, objf=-4334.875000, l2_term=-202.973862, weight=4352.000000)
13 ChainResults(loss=297.204376, objf=-5742.427734, l2_term=-296.249847, weight=6016.000000)
14 ChainResults(loss=190.700623, objf=-4397.112305, l2_term=-189.690262, weight=4352.000000)
15 ChainResults(loss=160.930191, objf=-4469.153809, l2_term=-159.903275, weight=4352.000000)
16 ChainResults(loss=196.294601, objf=-5793.267578, l2_term=-195.331619, weight=6016.000000)
17 ChainResults(loss=243.681747, objf=-6467.249512, l2_term=-242.746094, weight=6912.000000)
18 ChainResults(loss=158.063736, objf=-4443.177734, l2_term=-157.042786, weight=4352.000000)
19 ChainResults(loss=181.009201, objf=-5824.614258, l2_term=-180.041016, weight=6016.000000)
20 ChainResults(loss=113.364677, objf=-4284.602051, l2_term=-112.380165, weight=4352.000000) 
```


## TODO

- implement decode script using `forward.py` and `latgen-faster-mapped`
- use TDNN and check setup/speed/logprob/WER compatible with kaldi s5 recipe http://kishin-gitlab.cslab.kecl.ntt.co.jp/KOJIONO/pysoliton2_models/blob/master/psl2models/basic/tdnns.py

## known issues

- `torchain.io.open_example` return nullptr when batchsize is changed
