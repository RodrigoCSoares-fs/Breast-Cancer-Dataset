# Evaluating the performance of machine learning techniques in breast Cancer classification from the India database acquired through Kaggle

Breast Cancer Dataset - Binary Classification Prediction for type of Breast Cancer by Kaggle author M Yasser H published December 29, 2021 with KNN, SVM and MLP.

The studied database contains 32 columns composed of data related to breast cancer.
breast in India, which presents the characteristics and classes (diagnoses).

It was divided by the cancerData variable that has 29 columns [radius_mean,
texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean,
concave points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave points_se,
symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
smoothness_worst, compactness_worst, concavity_worst, concave points_worst, symmetry_worst], and
diagnosisClasses with 1 column [diagnosis] referring to the classes of the samples defined as Benigno
and Malignant.

After dividing the base, an analysis was performed to verify if it had null values ​​in the
columns provided. After the analysis, it was found that the base was prepared to normalize the
data on scales of 0 and 1 so as not to distort differences in the ranges of values ​​and to model the
data correctly.

To arrive at the results presented below, 80% of all data were used to
training, and 20% for testing. All this data has been randomly divided to ensure the best
training possible for the project. The general code was placed in an execution loop, in which it was
executed 20 consecutive times, after each execution the values ​​of Accuracy, Precision, Recall and F1,
of each of the classifiers were stored in a list, at the end of this loop, all the
results obtained in each of the evaluation metrics were added and divided by the total of
runs, thus resulting in the average of each metric in each of the classifiers.

After averaging all the data presented, it was seen that all classifiers
had a very high level of prediction, among the techniques, the most accurate was the SVM classifier
as shown in the table above.
