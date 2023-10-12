# Deep Spiking Reinforcement Learning
Implementations of Deep Reinforcement Learning (DRL) algorithms with
Spiking Neural Netowrks (SNNs) in PyTorch. SNNs are based on the
SpyTorch implementations, with custom encoding and decoding mechanisms.

## Dependency installation
Tested on Ubuntu 20.04 and Python 3.8.12. Creating a virtual environment is recommended.

```
pip install -r requirements.txt
```
For MuJoCo based environments (Ant-v3, HalfCheetah-v3, and Hopper-v3), install MuJoCo as described [here](https://github.com/openai/mujoco-py#install-mujoco).

## Citation

If you use our code, please consider citing our research:

```bibtex
@ARTICLE{10.3389/fnbot.2022.1075647,
AUTHOR={Akl, Mahmoud and Ergene, Deniz and Walter, Florian and Knoll, Alois},   
TITLE={Toward robust and scalable deep spiking reinforcement learning},      
JOURNAL={Frontiers in Neurorobotics},      
VOLUME={16},           
YEAR={2023},      
URL={https://www.frontiersin.org/articles/10.3389/fnbot.2022.1075647},       
DOI={10.3389/fnbot.2022.1075647},      
ISSN={1662-5218},   
}
```

```bibtex
@inproceedings{10.1145/3546790.3546804,
author = {Akl, Mahmoud and Sandamirskaya, Yulia and Ergene, Deniz and Walter, Florian and Knoll, Alois},
title = {Fine-Tuning Deep Reinforcement Learning Policies with r-STDP for Domain Adaptation},
year = {2022},
isbn = {9781450397896},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3546790.3546804},
doi = {10.1145/3546790.3546804},
abstract = {Using deep reinforcement learning policies that are trained in simulation on real robotic platforms requires fine-tuning due to discrepancies between simulated and real environments. Multiple methods like domain randomization and system identification have been suggested to overcome this problem. However, sim-to-real transfer remains an open problem in robotics and deep reinforcement learning. In this paper, we present a spiking neural network (SNN) alternative for dealing with the sim-to-real problem. In particular, we train SNNs with backpropagation using surrogate gradients and the (Deep Q-Network) DQN algorithm to solve two classical control reinforcement learning tasks. The performance of the trained DQNs degrades when evaluated on randomized versions of the environments used during training. To compensate for the drop in performance, we apply the biologically plausible reward-modulated spike timing dependent plasticity (r-STDP) learning rule. Our results show that r-STDP can be successfully utilized to restore the network’s ability to solve the task. Furthermore, since r-STDP can be directly implemented on neuromorphic hardware, we believe it provides a promising neuromorphic solution to the sim-to-real problem.},
booktitle = {Proceedings of the International Conference on Neuromorphic Systems 2022},
articleno = {14},
numpages = {8},
keywords = {neural networks, spiking neural networks, reinforcement learning},
location = {Knoxville, TN, USA},
series = {ICONS '22}
}
```

```bibtex
@inproceedings{10.1145/3477145.3477159,
author = {Akl, Mahmoud and Sandamirskaya, Yulia and Walter, Florian and Knoll, Alois},
title = {Porting Deep Spiking Q-Networks to Neuromorphic Chip Loihi},
year = {2021},
isbn = {9781450386913},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477145.3477159},
doi = {10.1145/3477145.3477159},
abstract = {Deep neural networks (DNNs) set the benchmark in many tasks in perception and control. Spiking versions of DNNs, implemented on neuromorphic hardware can enable orders of magnitude lower power consumption and low latency during network use. In this paper, we explore behavior and generalization capability of spiking, quantized spiking, and hardware implementation of deep Q-networks in two classical reinforcement learning tasks. We found that spiking neural networks have slightly decreased performance compared to non-spiking network, but we can avoid performance degradation from quantization and in-chip implementation. We conclude that since hardware implementation leads to lower power consumption and low latency, neuromorphic approach is a promising avenue for deep Q-learning. Furthermore, online learning, enabled in neuromorphic chips, can be used to compensate for the performance decrease in environments with parameter variations.},
booktitle = {International Conference on Neuromorphic Systems 2021},
articleno = {13},
numpages = {7},
keywords = {neuromorphic hardware, reinforcement learning, Spiking neural networks},
location = {Knoxville, TN, USA},
series = {ICONS 2021}
}
```
