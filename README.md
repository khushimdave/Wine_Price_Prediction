# Wine_Price_Prediction
This project provides a predictive model for estimating wine prices based on various features. It includes preprocessing techniques and label encoding.

**1.	Overview**
This code provides a predictive model for estimating wine prices based on various features. It preprocesses the input data, performs label encoding, and uses a trained model for making predictions. The code is designed to handle CSV files as input and generates an output file with predicted prices.

**2.	Prerequisites**



To use this code, you need the following:


•	Python installed on your system (version 3.10.5).


•	Required Python packages: (requirements.txt)

_o	joblib==1.2.0_

_o	numpy==1.24.3_

_o	pandas==2.0.1_

_o	python-dateutil==2.8.2_

_o	pytz==2023.3_

_o	scikit-learn==1.0.2_

_o	scipy==1.10.1_

_o	six==1.16.0_

_o	sklearn==0.0.post5

_o	threadpoolctl==3.1.0_

_o	tzdata==2023.3_

•	model.pkl file should be present in the same folder

**3.	Installation**



To install the required packages, follow these steps:



i.	Open a terminal or command prompt.

ii.	Navigate to the directory or the folder where you want to keep the code.

_cd path/to/code/directory_

iii.	Create a virtual environment (optional but recommended):

_python -m venv env_

iv.	Activate the virtual environment:

_For Windows: env\Scripts\activate_

_For macOS/Linux: source env/bin/activate_

v.	Install the required packages:

_pip install -r requirements.txt_

**4.	Usage**



_python pipeline.py <dataset.csv>_

Arguments:

dataset.csv (required): Input CSV file containing wine data.

**5.	Run the code**



i.)	Prepare the CSV file:

a.	The CSV file should contain the following columns: winery, wine, year, rating, num_reviews, region, type, body, acidity, and price (optional).

b.	Save the CSV file in a location accessible by the code.

ii.)	Open the terminal or command prompt.


iii.)	Navigate to the directory containing the code files:


_cd path/to/code/directory_

iv.)	Activate the virtual environment (if created):


_For Windows: env\Scripts\activate_

_For macOS/Linux: source env/bin/activate_

v.)	Run the pipeline.py:

In the cmd only, in continuation, copy the code:

_python pipeline.py dataset.csv_
 
 
 
**6.	Functionality**



The pipeline consists of the following steps:



_•	Drop Columns:_ The script drops columns from the input dataset that are not present in the predefined list of column names. If the dataset contains a "price" column, it is not dropped.


_•	Merge Price Dataset:_ If there is a "price" column in the dataset, it is temporarily removed, and the preprocessing steps are applied. After preprocessing, the predicted prices are merged back into the dataframe.


_•	Handle Missing Values and Change Datatypes:_ This step handles missing values and changes the datatypes of certain columns to avoid errors during preprocessing. The "type" column is filled with the mode value, the "acidity" column is filled with the median value, and the "year" column is filled and interpolated appropriately.


_•	Categorization:_ The "num_reviews" column is divided into four groups based on the number of reviews. The "rating" and "num_reviews_group" columns are used to categorize the data into different quality levels.


_•	Perform Label Encoding:_ Label encoding is applied to categorical columns such as "winery", "wine", "region", "type", and "combined_info". Unknown values are filled with "Unknown", and the original values are stored for later use.


_•	Use Pre-trained Model:_ The script loads the pre-trained machine learning model from a pickle file and uses it to predict the prices of wines.


_•	Merge Predictions:_ The predicted prices are merged back into the dataframe.


_•	Transform Encoded Data:_ The encoded values in the dataframe are replaced with their original values.


_•	Output:_ The preprocessed dataframe with predicted prices is saved as an output CSV file named "Output.csv".

 
**7.	Example**



The below command will preprocess the wine data in the "sample_dataset.csv" file and generate an output CSV file named "Output.csv" with predicted prices.

_python pipeline.py sample_dataset.csv_
 
 ![image](https://github.com/khushimdave/Wine_Price_Prediction/assets/94516006/8cb1068c-6d6c-4b1b-8be9-1199facad9f0)

Fig 7.1: Sample_dataset.csv

 ![Untitled](https://github.com/khushimdave/Wine_Price_Prediction/assets/94516006/1b88cf99-3b45-4c44-877a-38c7d377b726)


Fig 7.2: Output


Note: Please make sure to have the required dependencies installed before running the script.



**8.	Model Comparision (Results & Interpretation)**



Among the models that I have experimented with, here is a summary of their performance based on the evaluation metrics:

_Mean Squared Error (MSE):_

The model with the lowest MSE is Gradient Boosted Regression Trees, with a value of 6387.0844502514.

The model with the highest MSE is Linear Regression, with a value of 16553.065013070274.

_R2 Score:_

The model with the highest R2 score, indicating better prediction capability, is Gradient Boosted Regression Trees with a value of 0.7024446010804166.

The model with the lowest R2 score is Linear Regression with a value of 0.22884159405281468.

It appears that Gradient Boosted Regression Trees and Gradient Boosting Machines performed similarly in terms of MSE and R2 score, with slightly better results for Gradient Boosted Regression Trees.


So, based on the results, I have used _GRADIENT BOOSTED REGRESSSION TREES_ as my final model.



**9.	Limitations**



The code expects a specific column format in the input CSV file. Any deviations may result in errors or incorrect predictions.

The code assumes that a pre-trained model named "model.pkl" is available in the same directory as the code file. Ensure that you have the trained model file before running the code.



**10.	Conclusion**



To conclude, this code provides a predictive model for estimating wine prices based on various features. It utilizes preprocessing techniques, label encoding, and a trained model to make predictions. By following the provided instructions, users can run the code on their dataset and generate an output file with predicted prices.


The code has certain prerequisites, such as Python 3.10.5 and specific packages listed in the requirements.txt file. Users are advised to install these packages using a virtual environment for better isolation. Additionally, the code requires a pre-trained model file named "model.pkl" in the same directory.


The functionality of the code involves dropping unnecessary columns, handling missing values, changing data types, categorizing data, performing label encoding, using the pre-trained model, merging predictions, transforming encoded data, and generating the output CSV file.


A detailed example is provided to guide users through running the code on a sample dataset, showcasing the expected input format and the resulting output file.


It's important to note that the code has certain limitations. It expects a specific column format in the input CSV file, and any deviations may lead to errors or incorrect predictions. Users should also ensure that the required dependencies and the pre-trained model file are present before running the code.


In conclusion, this code provides a convenient solution for predicting wine prices based on given features, allowing users to make informed decisions and gain insights into wine valuation.
