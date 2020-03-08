# Environment-Aware Codebook
Learning environment-aware and hardware-compatible mmWave beam-forming codebooks using Artificial Neural Networks (ANN). This repository provides two models. Both are self-supervised and can handle unknown antenna manifold and learn phase-vector codebooks. One requires explicit mmWave channel knowledge while the other does not. Please refer to [paper]() for more details.

# Requirment:

1-Keras with Theano backend.

2-Numpy library.

3-Python 3.6 or above.

Optional:

1- MATLAB for data preperation. 

# Note:
The implementation of the complex-valued fully-connected layer is guided by that in [Deep Complex Nets](https://github.com/ChihebTrabelsi/deep_complex_networks) with some modifications.

# Citation
'''
@article{zhang2020learning,
  title={Learning Beam Codebooks with Neural Networks: Towards Environment-Aware mmWave MIMO},
  author={Zhang, Yu and Alrabeiah, Muhammad and Alkhateeb, Ahmed},
  journal={arXiv preprint arXiv:2002.10663},
  year={2020}
}
'''
