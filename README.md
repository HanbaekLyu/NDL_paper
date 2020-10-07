<p align="center">
<img width="600" src="https://github.com/ykureh/NMF-Networks/blob/master/NDL_logo.png?raw=true" alt="logo">
</p>


## Network Dictionary Learning (repository for paper)

This repository contains the scripts that generate the main figures reported in the paper:\
*Hanbaek Lyu, Yacoub Kureh, Joshua Vendrow, and Mason A. Porter*\
*"Learning low-rank latent mesoscale structures in networks*" (2020) [Link](https://hanbaeklyudotcom.files.wordpress.com/2020/10/ndl-1.pdf)

&nbsp;
&nbsp;

![](Figures/Figure1.png)
&nbsp;
![](Figures/Figure2.png)
&nbsp;
![](Figures/Figure3.png)
&nbsp;
![](Figures/Figure4.png)
&nbsp;



## Usage

First add network files for UCLA, Caltech, MIT, Harvard to Data/Networks_all_NDL\
Ref: Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter.\
Comparing community structure tocharacteristics in online collegiate social networks.SIAM Review, 53:526–543, 2011.

Then run generate_figures.py:
```python
>>> generate_figures.py
```
## File description 

  1. **utils.ndl.py** : main Network Dictionary Learning (NDL) and Network Reconstruction and Denoising (NDR) functions. 
  2. **utils.NNetwork.py** : Weighted network class (see https://github.com/HanbaekLyu/NNetwork). 
  3. **onmf.py**: Online Nonnegative Matrix Factorization algorithms (see https://github.com/HanbaekLyu/ONMF_ONTF_NDL)
  4. **helper_functions.final_plots_display.py**: helper functions for making plots 
  5. **helper_functions.main_script_NDL.py**: Main script for NDL experiments (hyper parameters can be tuned here)
  6. **generate_figures.py**: Run to generate figures (see description in the file) 
  
## Authors

* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)
* **Yakoub Kureh** - *Initial work* - [Website](https://www.math.ucla.edu/~ykureh/)
* **Joshua Vendrow** - *Initial work* - [Website](https://github.com/jvendrow)
* **Mason A. Porter** - *Initial work* - [Website](https://www.math.ucla.edu/~mason/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

