# AutoPQ: Automated point forecast-based quantile forecasts

AutoPQ addresses three challenges:
- Many state-of-the-art forecasting methods are still point forecasts and remain unused for probabilistic forecasts
- According to the no-free-lunch theorem, no forecasting method exists that excels in all forecasting tasks
- Smart grid applications typically require forecasts with customized probabilistic characteristics

## Methodology

The underlying idea of AutoPQ is to generate a probabilistic forecast based on an arbitrary point forecast using a conditional Invertible Neural Network (cINN) and to make corresponding design decisions automatically, aiming to increase the probabilistic performance. To account for different computing systems and performance requirements, two variants are available: AutoPQ-default suitable for standard computing systems achieving competitive forecasting performance, and AutoPQ-advanced requiring High-Performance Computing (HPC) systems to further increase forecasting performance for smart grid applications with high decision costs.

![concept_pipeline_github](https://github.com/SMEISEN/AutoPQ/assets/33990691/40344260-77ee-4515-9964-16875b9383d7)

## Installation

To install this project, perform the following steps.
1) Clone the project
2) Open a terminal of the virtual environment where you want to use the project
3) cd AutoPQ
4) pip install . or pip install -e . if you want to install the project editable.

## How to use

Exemplary evaluations using AutoPQ are given in the examples folder.

### Hyperparameter optimization

- The default configuration optimizes the sampling hyperparameter $\lambda_\text{q}$ for generating samples in the latent space of the cINN.
- The advanced configuration simultaneously optimizes the point forecasting method's hyperparameters $\boldsymbol{\lambda_\text{p}}$ and the sampling hyperparameter $\lambda_\text{q}$.

### Evaluation types

The evaluation trains the models using the training data sub-set, optimizes hyperparameters based on the validation data sub-set, and makes probabilistic forecasts for the test data sub-set.

## Citation

If you use this method please cite the corresponding papers:
> Kaleb Phipps, Stefan Meisenbacher, Benedikt Heidrich, Marian Turowski, Ralf Mikut, and Veit Hagenmeyer. 2023. Loss-customised probabilistic energy time series forecasts using automated hyperparameter optimisation. In Proceedings of the 14th ACM International Conference on Future Energy Systems (e-Energy ’23), Association for Computing Machinery, New York, NY, USA, 271–286. [https://doi.org/10.1145/3575813.3595204](https://doi.org/10.1145/3575813.3595204)

> Stefan Meisenbacher et al. 2024. AutoPQ: Automated point forecast-based quantile forecasts. In preparation.

## Funding

This project is funded by the Helmholtz Association under the Program “Energy System Design” and the Helmholtz Association's Initiative and Networking Fund through Helmholtz AI.

## References

The cINN is based on:
> B. Heidrich, M. Turowski, K. Phipps, K. Schmieder, W. Süß, R. Mikut, and V. Hagenmeyer, “Controlling non-stationarity and periodicities in time series generation using conditional invertible neural networks”, Applied Intelligence, vol. 53, no. 8, pp. 8826–8843, 2023.

Generating probabilistic forecasts by sampling in the cINN's latent space is based on:
> K. Phipps, B. Heidrich, M. Turowski, M. Wittig, R. Mikut, and V. Hagenmeyer, “Generating probabilistic forecasts from arbitrary point forecasts using a conditional invertible neural network”, Applied Intelligence, 2024.

Optimization of the sampling hyperparameter is performed using [Hyperopt](https://github.com/hyperopt/hyperopt)
> J. Bergstra, D. Yamins, and D. Cox, “Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures”, in Proceedings of the 30th International Conference on Machine Learning, ser. ICML ’13, Proceedings of Machine Learning Research, PMLR, 2013, pp. 115–123.

Optimization of the point forecasting method's hyperparameters is performed using [Propulate](https://github.com/Helmholtz-AI-Energy/propulate)
> O. Taubert, M. Weiel, D. Coquelin, A. Farshian, C. Debus, A. Schug, A. Streit, and M. Götz, “Massively parallel genetic optimization through asynchronous propagation of populations”, in High Performance Computing, A. Bhatele, J. Hammond, M. Baboulin, and C. Kruse, Eds., Cham, Switzerland: Springer Nature, 2023, pp. 106–124.

The Load-BW data is taken from the Open Power System Data (OPSD) portal:
> F. Wiese et al., “Open Power System Data: Frictionless data for electricity system modelling”, Applied Energy, vol. 236, pp. 401–409, 2019.

The Load-GCP data set is taken from the UCI Machine Learning Repository:
> A. Trindade, Electricity load diagrams 2011-2014, UCI Machine Learning Repository, 2015.

The Mobility data set is taken from the UCI Machine Learning Repository:
> H. Fanaee-T, Bike sharing dataset, UCI Machine Learning Repository, 2013.

The Price, PV, and WP data are from the price, solar power, and wind power forecasting tracks of the Global Energy Forecasting Competition (GEFCom) 2014:
> T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli, and R. J. Hyndman, “Probabilistic energy forecasting: Global energy forecasting competition 2014 and beyond”, International Journal of Forecasting, vol. 32, no. 3, pp. 896–913, 2016.
