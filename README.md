# DeepAdipose
Analysis of CT texture features of visceral adipose tissue for evaluation of metabolic disorders and surgery-induced weight loss effects

![](/diagram.png)

## Prerequisites

### The following libraries are required:

R, python 3.6, SPSS, SimpleITK, radiomics, matplotlib, pandas, scipy, scikit-learn, tensorflow 1.x and keras

## The dataset _(Folder "Data")_

The dataset contains extracted CT imaging features from 675 volunteer studies and 63 obesity patients (with bariatric surgery).

Features were extracted from umbilical level _(allBlockData.xlsx and allSliceData.xlsx)_

Major features with routine clinical parameters and metabolic outcomes were provided _(all_major_block_features_with_clinical.xlsx)_

Features with surgery outcomes were provided in _surgery_merged_with_clinical.xlsx_

## The Code

We provided both python/R source code and corresponding jupyter notebooks.

### 1. Extract CT features from segmented adipose tissues

After segmentation of visceral adipose tissue (VAT) from CT scans with an in-house software, texture features were extracted from umbilical CT blocks and slices using the following code:

_ExtractFeaturesFromBlocksSlices.py (for volunteer cohort)_

_ExtractFeaturesFromSurgery.py (for surgery cohort)_

### 2. Build deep learning models for evaluation of metabolic outcomes

Deep feed-forward models were built based on major features extracted from CT blocks and slices.

_DeepAdiposeBlock.py (with features obtained from CT blocks)_

_DeepAdiposeSlice.py (with features obtained from CT slices)_

### 3. Evaluate the performance of deep learning models on test data

Load pretrained models from "Results/Models" and evalaute on test data _(DeepAdiposeTest.py)_

### 4. Visualization analysis of the correlation between imaging features and metabolic outcomes as well as clinical parameters

Create a Sankey diagram to visualize the correlations _(AdiposeSankey.py)_

Create heatmaps to visualize the pattern of imaging features in relation to metabolic outcomes _(AdiposeHeatmap.r for major features, AdiposeHeatmapFull.r for all features)_

ROC analysis of single biomarkers for Metabolic syndrome and insulin resistance _(SPSS_ROC_Single_Marker.spv)_

### 5. Surgery analysis with SPSS

Analysis of imaging features in relation to surgery outcomes with SPSS _(Surgery_SPSS_Analysis.spv)_
