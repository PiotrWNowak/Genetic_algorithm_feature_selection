# Feature selection based on genetic algorithm

In machine learning feature selection is one of crucial parts. The are some algorithms including Recursive feature elimination and feature importance from Random Forest estimator. Using genetic algorithm to calculate models for chosen features is one of the most accurate but very time consuming methods. It's important that in case of genetic algoritm after getting best performance features there is no additional computing. Using preprocesing data transformation there is need to do tranformation every time model run.

## Example of usage

In example_feature_selection.py and example_feature_selection.ipynb there is comparision genetic algorithm method to most popular preprocessing feature selection methods and data transformations.

There is important addition to genetic algoritm. There is 2 addition regulation that add penalty to result for using too many features. One is number of features used multiply constant, second is number of features multiply percent of result.

### Breast cancer dataset (clasification) feature selection and data transformation comparision (logistic regression)

<p align="center">
  <img src="https://github.com/PiotrWNowak/LHCb_trigger_genetic_algorithm/raw/master/images/Figure_2-2.png">
</p>



Now using scikit-learn function PolynomialFeatures we large amound of features, most of them unnecessary for model performance. Using regulation in genetic algorithm we can decrease a lot of features that are not making our model more accurate.


<p align="center">
  <img src="https://github.com/PiotrWNowak/LHCb_trigger_genetic_algorithm/raw/master/images/Figure_2-1.png">
</p>

### Wine dataset (regression) feature selection and data transformation comparision (linear regression)

<p align="center">
  <img src="https://github.com/PiotrWNowak/LHCb_trigger_genetic_algorithm/raw/master/images/Figure_2-3.png">
</p>



<p align="center">
  <img src="https://github.com/PiotrWNowak/LHCb_trigger_genetic_algorithm/raw/master/images/Figure_2-4.png">
</p>

## Using genetic algorithm for neural network for LHCb track classifier

Main reason to create genetic algorithm feature selection was to use it on Neural Network classifier for LHCb experiment. In keras_model_feature_selection.py is used genetic algorithm to select features. Because of long time to computate every case for checking model performance we use only 5*10^5 events.

Results of running genetic alghoritm with best and mean performance every generation.
<p align="center">
  <img src="https://github.com/PiotrWNowak/LHCb_trigger_genetic_algorithm/raw/master/images/Figure_5.png">
</p>

Best feature selected model results was comparise with model trained without feature selection.
<p align="center">
  <img src="https://github.com/PiotrWNowak/LHCb_trigger_genetic_algorithm/raw/master/images/Figure_8.png">
</p>

In repository https://github.com/PiotrWNowak/LHCb_track_classifier there is analysys for selected features in keras neural network and later usage.

## Author

**Piotr Nowak**
