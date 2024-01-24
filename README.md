# Deformable Linear Objects Manipulation with Online Model Parameters Estimation

Manipulating Deformable Linear Objects (DLOs) is a challenging task for a robotic system due to their unpredictable configuration, high-dimensional state space and complex nonlinear dynamics. 
This paper presents a framework addressing the manipulation of DLOs, specifically targeting the model-based shape control task with the simultaneous online gradient-based estimation of model parameters.
In the proposed framework, a neural network is trained to mimic the DLO dynamics using the data generated with an analytical DLO model for a broad spectrum of its parameters.
The neural network-based DLO model is conditioned on these parameters and employed in an online phase to perform the shape control task by estimating the optimal manipulative action through a gradient-based procedure.
In parallel, gradient-based optimization is used to adapt the DLO model parameters to make the neural network-based model better capture the dynamics of the real-world DLO being manipulated and match the observed deformations.
To assess its effectiveness, the framework is tested across a variety of DLOs, surfaces, and target shapes in a series of experiments. The results of these experiments demonstrate the validity and efficiency of the proposed methodology compared to existing methods.
<div align="center">
 :arrow_right:  Project website: https://sites.google.com/view/dlo-manipulation	
 
 :arrow_right:  Paper (Open Access): https://ieeexplore.ieee.org/document/10412116
</div>

### Python environment

```
python 3.9
pytorch 1.10
```

### Download dataset DLO manipulation for training

https://drive.google.com/file/d/1CGk9T3X2u0Fun8uwNLIU-l7HV6RKNut1/view?usp=sharing


### Citation
If you find our research interesting, please cite the following manuscript.
```
@article{caporali2024deformable,
  author={Caporali, Alessio and Kicki, Piotr and Galassi, Kevin and Zanella, Riccardo and Walas, Krzysztof and Palli, Gianluca},
  journal={IEEE Robotics and Automation Letters}, 
  title={Deformable Linear Objects Manipulation with Online Model Parameters Estimation}, 
  year={2024},
  doi={10.1109/LRA.2024.3357310}
}
```
