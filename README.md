 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
# InVAErt networks for amortized inference and identifiability analysis of lumped parameter hemodynamic models

This repository contains pre-trained neural network models, training dataset and all related scripts. For additional information, please refer to the publications below:

1. InVAErt networks for amortized inference and identifiability analysis of lumped parameter hemodynamic models, [Guoxiang Grayson Tong](https://grayson3455.github.io/), [Carlos A. Sing-Long Collao](https://www.ing.uc.cl/academicos-e-investigadores/carlos-alberto-sing-long-collao/), and [Daniele E. Schiavazzi](https://www3.nd.edu/~dschiava/).

2. [InVAErt networks: A data-driven framework for model synthesis and identifiability analysis](https://www-sciencedirect-com.proxy.library.nd.edu/science/article/pii/S0045782524001026), [Guoxiang Grayson Tong](https://grayson3455.github.io/), [Carlos A. Sing-Long Collao](https://www.ing.uc.cl/academicos-e-investigadores/carlos-alberto-sing-long-collao/), and [Daniele E. Schiavazzi](https://www3.nd.edu/~dschiava/).

### Description of the ```Tools``` folder
1. ```DNN_tools.py```: common functions for deep neural network modeling.
2. ```Model.py```: neural network modules.
3.  ```NF_tools.py```: functions used by the [Real-NVP](https://arxiv.org/abs/1605.08803) based normalizing flow model.
4.  ```Training_tools.py```: training and testing functions of the inVAErt networks.
5.  ```EHR_tools.py```: functions used in the study of the EHR dataset.
6.  ```cvsim6_scripts.py```: functions used in the study of the [CVSim-6](https://dspace.mit.edu/handle/1721.1/13823) model.
7.  ```plotter.py```: common plotting functions.

### Description of the ```Solver``` folder
1. ```CVSIM6-Training-Data```: dataset used in this paper.
2. ```CVSim6_stiffness_pictures```: results of the CVSim-6 stiffness analysis.
3. ```ExternalData```: the [EHR dataset](https://github.com/desResLab/supplMatHarrod20/blob/master/data/EHR_dataset.csv).
4. ```cvsim6-explicit-RK4.py```: explicit RK4 solver for the CVSim-6 system.
5. ```cvsim6-stiffness_analysis.py``` : functions for the stiffness analysis of the CVSim-6 system.
6. ```cvsim6-system-test.py```: testing CVSim-6 solver with the reference parameter set.
7. ```cvsim6-training-data-generator.py```: parallel training data generator.
8. ```cvsim6_plotting_functions.py```: plotting functions for the stiffness analysis.
9. ```cvsim6_simulator.py```: an implicit, adaptive [solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for the CVSim-6 system.

### Description of the ```Model``` folder
1. ```Structural_id_study```: pre-trained models for the structural identifiability analysis.
2. ```EHR```: pre-trained models for the study of the EHR dataset.

#### Please stay tuned for the Jupyter Notebook tutorials!
