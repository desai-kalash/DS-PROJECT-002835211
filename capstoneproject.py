import streamlit as st


# Set page configuration
st.set_page_config(layout="wide")

# Define CSS for the gradient background
style = """
        <style>
        .stApp {
            background-image: linear-gradient(to right, #00008B, #800080);
            color: #FFFFFF;  /* Ensures text is white for better contrast */
        }
        </style>
        """

# Apply the style
st.markdown(style, unsafe_allow_html=True)


# Use markdown for custom text colors and styling
st.markdown("""
    <style>
    .big-font {
        font-size:300% !important;
        color: white;
    }
    .medium-font {
        font-size:200% !important;
        color: #34A853;
    }
    .small-font {
        font-size:150% !important;
        color: #4285F4;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="big-font">Welcome to My Streamlit App- CHURN PREDICTION</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="medium-font">DATA SCIENCE CAPSTONE PROJECT-- by Kalash Desai</h2>', unsafe_allow_html=True)


st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

#Import the necessary Libraries
import streamlit as st
import pandas as pd

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/desai-kalash/DS-PROJECT-002835211/main/Dataset---WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first 5 rows of the dataset

st.markdown("<h1 style='color: yellow;'>ANALYSING THE DATASET: 'Telco Customer Churn Dataset'</h1>", unsafe_allow_html=True)

st.write(df.head())

st.write("")  # Add an empty line for spacing

# Drop the 'customerID' column
df.drop('customerID', axis='columns', inplace=True)

# Display the number of rows and columns in the dataset
st.markdown("<h2 style='color: yellow;'>NUMBER OF ROWS AND COLUMNS IN THE DATASET:</h2>", unsafe_allow_html=True)
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing

# Display the list of column names
st.markdown("<h2 style='color: yellow;'>COLUMN NAMES IN THE DATASET:</h2>", unsafe_allow_html=True)

st.write(df.columns.tolist())
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing

# Display the information about the dataset
st.markdown("<h2 style='color: yellow;'>INFORMATION ABOUT THE DATASET:</h2>", unsafe_allow_html=True)

st.write(df.info())
st.write("")  # Add an empty line for spacing


# Convert 'TotalCharges' column to numeric datatype
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Drop rows with NaN values in the 'TotalCharges' column
df.dropna(subset=['TotalCharges'], inplace=True)


# Display the description of columns with numeric values
st.markdown("<h2 style='color: yellow;'>COLUMN DESCRIPTION (with Numeric Values)</h2>", unsafe_allow_html=True)

st.write(df.describe().T)

st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
# Display the number of null values in each column
st.markdown("<h2 style='color: yellow;'>NULL VALUES IN EACH COLUMN</h2>", unsafe_allow_html=True)

st.write(df.isnull().sum())
st.write("")  # Add an empty line for spacing


# Display the data for row 200
st.markdown("<h2 style='color: yellow;'>DATA FOR ROW 200</h2>", unsafe_allow_html=True)

st.write(df.iloc[200])
st.write("")  # Add an empty line for spacing
#
#
#
#

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt # type: ignore

# Calculate the number of customers who churned vs not churned
churn_distribution = df['Churn'].value_counts()

# Set up the plot for a horizontal bar chart
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(churn_distribution.index, churn_distribution, color=['green', 'maroon'])

# Annotate bars with numeric values
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, width, ha='left', va='center')

# Add labels and title
ax.set_xlabel('Number of Customers')
ax.set_ylabel('Churn Status')
ax.set_title('CHURN DISTRIBUTION')

# Adjust the layout and display the plot
st.markdown("<h2 style='color: yellow;'>CUSTOMER CHURN STATUS DISTRIBUTION</h2>", unsafe_allow_html=True)

st.pyplot(fig)
st.write("")  # Add an empty line for spacing
multiline_string = """
INFERENCE: 

 Counts occurrences of customer churn and visualizes it using a horizontal bar chart with green and maroon bars.

Annotates each bar with the count of customers.

Adds axis labels and a title, adjusts layout for readability, and displays the chart.

There are significantly more customers who have not churned (5,163) compared to those who have (1,869), indicating a churn rate below 30%
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing

#
#
#
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors for the heatmap
colors = ['#3498DB', '#2D2926']

# Calculate statistics for churned and not churned customers
churn = df[df['Churn'] == 'Yes'].describe().T
not_churn = df[df['Churn'] == 'No'].describe().T

# Create subplots for the heatmaps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plot heatmap for churned customers
plt.subplot(1, 2, 1)
sns.heatmap(churn[['mean']], annot=True, cmap=colors, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('Churned Customers')

# Plot heatmap for not churned customers
plt.subplot(1, 2, 2)
sns.heatmap(not_churn[['mean']], annot=True, cmap=colors, linewidths=0.4, linecolor='black', cbar=False, fmt='.2f')
plt.title('Not Churned Customers')

# Adjust layout and display the plot
fig.tight_layout(pad=0)
st.markdown("<h3 style='color: yellow;'>Customer Churn Statistics</h3>", unsafe_allow_html=True)

st.pyplot(fig)
st.write("")  # Add an empty line for spacing
multiline_string = """
INFERENCE:  
Calculates descriptive statistics for customers who have churned versus those who haven't.

Displays the means of numeric features for both groups side-by-side using heatmaps.

Sets specific colors for the heatmaps and formats the layout for clarity.

Churned customers have a lower tenure and higher monthly charges on average compared to those who haven't churned, suggesting these factors may influence churn decisions.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Count the occurrences of 'Yes' in 'Dependents' when 'Churn' is 'Yes'
count_yes_dependents = df[df['Churn'] == 'Yes']['Dependents'].value_counts().get('Yes', 0)
count_no_dependents = df[df['Churn'] == 'Yes']['Dependents'].value_counts().get('No', 0)

# Create a bar chart
categories = ['Churn with dependents', 'Churn without dependents']
values = [count_yes_dependents, count_no_dependents]
fig, ax = plt.subplots()
ax.bar(categories, values, color=['green', 'red'])

# Annotate bars with numeric values
for i, value in enumerate(values):
    ax.text(categories[i], value + 1, str(value), ha='center')

# Add title and labels
ax.set_title('Number of Customers - Churn with dependents v/s Churn without dependents')
ax.set_ylabel('Number of Customers')

# Display the plot
st.markdown("<h3 style='color: yellow;'>Number of Customers - Churn with dependents vs Churn without dependents</h3>", unsafe_allow_html=True)

st.pyplot(fig)
st.write("")  # Add an empty line for spacing
multiline_string = """
INFERENCE: 
Counts and compares the number of churned customers with and without dependents.

Plots these counts in a bar chart, annotated with the actual numbers.

Uses green for 'Churn with dependents' and red for 'Churn without dependents'.

More customers without dependents churn compared to those with dependents.

Dependency status could be a factor in churn behavior 
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# Calculating the number of churned customers with and without partners
partner_churn_yes = df.loc[df['Churn'] == 'Yes', 'Partner'].value_counts()

# Setting up the plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#4CAF50', '#F44336']  # green for "Yes", red for "No"
ax.bar(partner_churn_yes.index, partner_churn_yes, color=colors, edgecolor='black')

# Annotating the bar chart with the count above each bar
for index, value in enumerate(partner_churn_yes):
    ax.text(index, value + 3, str(value), ha='center', va='bottom')

# Setting labels and title
ax.set_xlabel('Partner Status')
ax.set_ylabel('Number of Churned Customers')
ax.set_title('Churn Analysis Based on Partner Status')
ax.set_xticklabels(['With Partner', 'Without Partner'])

# Display the plot
st.markdown("<h3 style='color: yellow;'>Churn Analysis Based on Partner Status</h3>", unsafe_allow_html=True)

st.pyplot(fig)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE: 
Computes and plots the number of churned customers based on their partner status.

Uses a bar chart with green for 'With Partner' and red for 'Without Partner', along with annotations for each bar.

Customers without partners are more likely to churn than those with partners.

Partner status seems to be a significant factor in churn likelihood.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Filter the DataFrame for Churn = 'Yes' and 'No'
churn_yes = df[df['Churn'] == 'Yes']
churn_no = df[df['Churn'] == 'No']

# Count the occurrences of each InternetService category for Churn = 'Yes' and 'No'
counts_yes = churn_yes['InternetService'].value_counts()
counts_no = churn_no['InternetService'].value_counts()

# Create two subplots for the pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Pie chart for Churn = 'Yes'
ax1.pie(counts_yes, labels=counts_yes.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Churn = Yes')

# Pie chart for Churn = 'No'
ax2.pie(counts_no, labels=counts_no.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Churn = No')

# Display the pie charts
st.markdown("<h3 style='color: yellow;'>Distribution of InternetService Categories for Churned Customers</h3>", unsafe_allow_html=True)

st.pyplot(fig)

st.write("")  # Add an empty line for spacing
multiline_string = """
INFERENCE: 
Filters customers by churn status and counts how many use each type of internet service.

Creates pie charts to visualize the proportions of internet service types among churned and retained customers.

A higher percentage of churned customers used fiber optic compared to those who did not churn.

DSL service seems to be less associated with customer churn.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Filter the DataFrame for Churn = 'Yes' and 'No'
churn_yes = df[df['Churn'] == 'Yes']
churn_no = df[df['Churn'] == 'No']

# Count the occurrences of each Contract category for Churn = 'Yes' and 'No'
counts_yes = churn_yes['Contract'].value_counts()
counts_no = churn_no['Contract'].value_counts()

# Create two subplots for the pie charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Pie chart for Churn = 'Yes'
ax1.pie(counts_yes, labels=counts_yes.index, autopct='%1.1f%%', startangle=90)
ax1.set_title('Churn = Yes')

# Pie chart for Churn = 'No'
ax2.pie(counts_no, labels=counts_no.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Churn = No')

# Display the pie charts
st.markdown("<h3 style='color: yellow;'>Distribution of Contract Categories for Churned Customers</h3>", unsafe_allow_html=True)

st.pyplot(fig)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE: 
Filters and counts contract types among churned and non-churned customers.

Displays the results using pie charts for visual comparison of contract preferences between the two groups.

Month-to-month contracts are heavily prevalent among those who churned.

Longer-term contracts are associated with lower churn rates.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
# Display unique values for each column
st.markdown("<h3 style='color: yellow;'>Unique Values in Each Column</h3>", unsafe_allow_html=True)

for column in df.columns:
    st.write(f"**{column}:** {df[column].unique()}")
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:

Iterates through each column in the DataFrame df to print out the unique values present in that column.

The dataset features binary, categorical, and continuous variables.

Service features have 'No internet service' as a separate category.

Tenure and charges vary widely among customers.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
def display_unique_values(dataframe):
    object_cols = dataframe.select_dtypes(include='object').columns
    for col in object_cols:
        unique_values = dataframe[col].dropna().unique()
        unique_str = ', '.join(unique_values)
        st.write(f"Unique values in '{col}': {unique_str}\n")

# Display unique values for object columns
st.markdown("<h3 style='color: yellow;'>Unique Values in Object Columns</h3>", unsafe_allow_html=True)

display_unique_values(df)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:

Defines a function to display unique values for all object (categorical) columns in the DataFrame.

Filters out numerical data, focusing on textual information.

Categorical variables show binary and multiple categories.

Services categories include an option for non-subscription.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing

#
#
#
import streamlit as st
import pandas as pd

def consolidate_services_and_display_uniques(dataframe):
    # Consolidate 'No internet service' and 'No phone service' into 'No'
    no_service_values = ['No internet service', 'No phone service']
    dataframe.replace(no_service_values, 'No', inplace=True)
    
    # Iterate over columns and display unique values for columns with object type
    for col in dataframe.select_dtypes('object').columns:
        unique_values = ', '.join(dataframe[col].dropna().unique())
        st.write(f"Unique values in '{col}': {unique_values}")

# Display unique values for object columns after consolidation
st.markdown("<h3 style='color: yellow;'>Unique Values in Object Columns after Consolidation</h3>", unsafe_allow_html=True)

consolidate_services_and_display_uniques(df)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:

The function standardizes 'No internet service' and 'No phone service' entries to 'No' for simplicity.

Then it displays the unique values in categorical columns.

Simplification of service-related categories may aid in analysis.

Streamlined categorical data could improve model performance and interpretability.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

# Specify columns to convert from 'Yes'/'No' to 1/0
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

# Replace 'Yes' and 'No' with 1 and 0 in specified columns
for col in yes_no_columns:
    df[col] = df[col].replace({'Yes': 1, 'No': 0})

# Display the modified DataFrame
st.markdown("<h3 style='color: yellow;'>Modified DataFrame</h3>", unsafe_allow_html=True)

st.write(df)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
The code snippet iterates over specified columns with binary 'Yes'/'No' responses.

Replaces 'Yes' with 1 and 'No' with 0 to convert these columns to a numeric binary format suitable for machine learning algorithms.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import pandas as pd

def display_object_column_uniques(dataframe):
    # Get a list of columns with the object data type
    object_columns = dataframe.select_dtypes(include='object').columns
    # Iterate through the list and display unique values
    for col in object_columns:
        unique_vals = ', '.join(str(v) for v in dataframe[col].unique())
        st.write(f"{col} has the unique values: {unique_vals}")

# Display unique values for object columns
st.markdown("<h3 style='color: yellow;'>Unique Values in Object Columns</h3>", unsafe_allow_html=True)

display_object_column_uniques(df)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
This function identifies columns with non-numeric (object) data types and prints their unique values.

Useful for understanding categorical data before encoding for modeling.

Remaining object-type columns represent categorical variables requiring encoding before use in most machine learning models.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
# Replace values in the "gender" column
df["gender"] = df["gender"].replace({'Female': 1, 'Male': 0})

# Display the modified DataFrame
st.markdown("<h3 style='color: yellow;'>Modified DataFrame</h3>", unsafe_allow_html=True)

st.write(df)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE: 
The code converts the 'gender' column to a binary format, where 'Female' is encoded as 1 and 'Male' is encoded as 0.

This transformation prepares the column for use in machine learning algorithms that require numerical input.
"""
st.write(multiline_string)

st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

st.markdown("<h3 style='color: yellow;'>Unique Values in Object Columns</h3>", unsafe_allow_html=True)

display_object_column_uniques(df)
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import pandas as pd

# Perform one-hot encoding on the specified columns
df = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Display a random sample of 4 rows in the dataset
st.markdown("<h3 style='color: yellow;'>Random Sample of 4 Rows After One-Hot Encoding</h3>", unsafe_allow_html=True)
st.write(df.sample(4))
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Applies one-hot encoding to the 'InternetService', 'Contract', and 'PaymentMethod' columns, creating a binary column for each category within those columns.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
#
#
#
from sklearn.preprocessing import MinMaxScaler

# Define the columns to scale
columns_for_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Import and configure the MinMaxScaler
norm_scaler = MinMaxScaler()

# Normalize the selected columns
df[columns_for_scaling] = norm_scaler.fit_transform(df[columns_for_scaling])

# Display a random sample of 3 rows in the dataset after normalization
st.markdown("<h3 style='color: yellow;'>Random Sample of 3 Rows After Normalization</h3>", unsafe_allow_html=True)

st.write(df.sample(3))
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Selects 'tenure', 'MonthlyCharges', and 'TotalCharges' columns for scaling.

Uses MinMaxScaler to normalize these columns, scaling the data to the range [0, 1].
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
# Split the dataset into independent (X) and dependent (y) features
X = df.drop('Churn', axis='columns')
y = df['Churn']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Display the shapes of the train and test sets
st.markdown("<h3 style='color: yellow;'>Train-Test Split</h3>", unsafe_allow_html=True)

st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Separates the dataset into independent features (X) and the dependent (target) feature (y), where y represents the 'Churn' column and X includes all other columns.

Uses train_test_split from sklearn to divide the data into training and testing sets.

It sets aside 20% of the data for testing and uses a random state for reproducibility.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

# Split the dataset into independent (X) and dependent (y) features
X = df.drop('Churn', axis='columns')
y = df['Churn']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Display the shapes of X_train and X_test
st.markdown("<h3 style='color: yellow;'>Shapes of X_train and X_test</h3>", unsafe_allow_html=True)

st.write(f"X_train shape: {X_train.shape}")
st.write(f"X_test shape: {X_test.shape}")
st.write("")  # Add an empty line for spacing

# Display a random sample of 3 rows from X_train
st.markdown("<h3 style='color: yellow;'>Random Sample of 3 Rows from X_train</h3>", unsafe_allow_html=True)

st.write(X_train.sample(3))
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the dataset into independent (X) and dependent (y) features
X = df.drop('Churn', axis='columns')
y = df['Churn']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Fit a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display a message indicating that the model has been trained
st.markdown("<h3 style='color: yellow;'>Logistic Regression Model Trained Successfully!</h3>", unsafe_allow_html=True)

st.write("")  # Add an empty line for spacing


multiline_string = """
INFERENCE:
Initializes and trains a logistic regression model using the training data (X_train, y_train). 

This model will be used to predict the 'Churn' variable based on the other features in the dataset.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
# Make predictions on the test data
y_pred = model.predict(X_test)

# Display a message indicating that the predictions have been made
st.markdown("<h3 style='color: yellow;'>Predictions Made Successfully!</h3>", unsafe_allow_html=True)

st.write("Predictions:", y_pred)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Uses the trained logistic regression model to make predictions on the test dataset (X_test). 

The predictions (y_pred) represent the model's assessment of whether each customer in the test set is likely to churn or not based on their features.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
# Extract coefficients and intercept from the model
coeffs = model.coef_.flatten()
model_intercept = model.intercept_[0]

# Display the coefficients and the intercept
st.markdown("<h3 style='color: yellow;'>Model Coefficients:</h3>", unsafe_allow_html=True)

for index, coef in enumerate(coeffs, start=1):
    st.write(f"Feature {index}: {coef:.4f}")

st.write("\n### Model Intercept:")
st.write(f"{model_intercept:.4f}")
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:

Extracts and prints the logistic regression model's coefficients and intercept. Each coefficient corresponds to one feature in the dataset.

Coefficients indicate the impact of each feature on the likelihood of churn.

Positive values increase the probability of churn; negative values decrease it.

The intercept (-0.5380) adjusts the decision boundary of the logistic function.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
from sklearn.metrics import classification_report
# Display the classification report
st.markdown("<h3 style='color: yellow;'>Classification Report:</h3>", unsafe_allow_html=True)

st.write(classification_report(y_test, y_pred))
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Generates a classification report from the predictions, providing metrics such as precision, recall, f1-score, and support for each class.

Model performs better at predicting non-churn (class 0) with higher precision and recall.

Overall accuracy is 79%, indicating reasonable predictive performance.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing

#
#
#
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate the confusion matrix
matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix as a heatmap
st.markdown("<h3 style='color: yellow;'>Confusion Matrix:</h3>", unsafe_allow_html=True)

plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
st.pyplot(plt)
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Creates and displays a confusion matrix using Seaborn's heatmap to show the true vs. predicted values for the churn classification.

The matrix indicates more true negatives (890) and true positives (226) than false negatives (182) and false positives (109).

This visual helps in understanding the model's performance across different classes.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import sys
import subprocess
from setuptools import setup
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Define the neural network model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
st.markdown("<h3 style='color: yellow;'>Model Summary:</h3>", unsafe_allow_html=True)

model.summary()
st.write("")  # Add an empty line for spacing
# Train the model
model.fit(X_train, y_train, epochs=50)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
st.markdown("<h3 style='color: yellow;'>Model Evaluation on Test Data:</h3>", unsafe_allow_html=True)

st.write(f"Loss: {loss}")
st.write(f"Accuracy: {accuracy}")
st.write("")  # Add an empty line for spacing


multiline_string = """
INFERENCE:
Defines a Sequential neural network model with three layers: two hidden layers with 32 and 16 neurons, and an output layer with 1 neuron for binary classification.

The model uses ReLU activation for hidden layers and sigmoid for the output layer, optimized with Adam and binary cross-entropy loss.

The model summary displays the architecture details.

Fits the model on the training data (X_train, y_train) for 50 epochs, which means the model will go through the entire dataset 50 times during training to learn the patterns for churn prediction.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Display the evaluation results
st.write("### Model Evaluation on Test Data:")
st.write(f"Loss: {loss}")
st.write(f"Accuracy: {accuracy}")
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Evaluates the trained neural network model on the test data (X_test, y_test).

Reports accuracy as approximately 77.57% and a loss of 0.4638.

The model has a decent accuracy, slightly lower than the logistic regression model, and a moderate loss, indicating room for improvement.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#


# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/desai-kalash/DS-PROJECT-002835211/main/Dataset---WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Split the dataset into independent (X) and dependent (y) features
X = df.drop('Churn', axis='columns')
y = df['Churn']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Define the neural network model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50)

# Predict churn values
y_predicted = model.predict(X_test)
st.write("### Original predicted array:")
st.write(y_predicted[:5])

# Convert predicted probabilities to binary values (0 or 1)
y_predicted_binary = (y_predicted > 0.5).astype(int).flatten()
st.markdown("<h3 style='color: yellow;'>Binary predicted array:</h3>", unsafe_allow_html=True)

st.write(y_predicted_binary[:5])
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
The model's predictions (y_predicted) for the first five test samples are probabilities ranging from 0 to 1, typical for a sigmoid output layer.

These probabilities indicate the model's confidence in predicting class 1 (churn); closer to 1 suggests higher likelihood of churn.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

# Convert predicted probabilities to binary values (0 or 1) using a threshold of 0.5
y_predicted_0_or_1 = []
for element in y_predicted:
    if element > 0.5:
        y_predicted_0_or_1.append(1)
    else:
        y_predicted_0_or_1.append(0)

st.markdown("<h3 style='color: yellow;'>Binary predicted array:</h3>", unsafe_allow_html=True)

st.write(y_predicted_0_or_1[:10])

st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Processes the predicted probabilities (y_predicted) to assign class labels: above 0.5 becomes class 1 (churn), below becomes class 0 (no churn).

Outputs the first ten binary predictions.

This thresholding step converts probabilities into definite class predictions needed for evaluating classification performance.
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#


# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/desai-kalash/DS-PROJECT-002835211/main/Dataset---WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Split the dataset into independent (X) and dependent (y) features
X = df.drop('Churn', axis='columns')
y = df['Churn']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Define the neural network model
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50)

# Predict churn values
y_predicted = model.predict(X_test)

# Convert predicted probabilities to binary values (0 or 1) using a threshold of 0.5
y_predicted_0_or_1 = []
for element in y_predicted:
    if element > 0.5:
        y_predicted_0_or_1.append(1)
    else:
        y_predicted_0_or_1.append(0)

# Display the first 10 elements of y_test and y_predicted_0_or_1
st.markdown("<h3 style='color: yellow;'>Comparison of y_test and Predicted Values:</h3>", unsafe_allow_html=True)

st.write("y_test (correct answers of the predictions): ", list(y_test)[:10])
st.write("Predicted values: ", y_predicted_0_or_1[:10])
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
The y_test (correct answers of the predictions) list looks like -  [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
The list for predictions made on x_test looks like -  [0, 0, 0, 1, 1, 1, 0, 1, 0, 0]
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#

# Display the classification report
st.markdown("<h3 style='color: yellow;'>Classification Report:</h3>", unsafe_allow_html=True)

st.write(classification_report(y_test, y_predicted_0_or_1))
st.write("")  # Add an empty line for spacing

multiline_string = """
INFERENCE:
Generates a classification report for the neural network's predictions, giving precision, recall, and f1-score for both classes.

The neural network model shows balanced precision and recall across classes, with an overall accuracy matching the evaluation result (approximately 77%).
"""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
st.write("")  # Add an empty line for spacing
#
#
#
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Convert string labels to integers
y_test_int = y_test.astype(int)
y_predicted_int = np.array(y_predicted_0_or_1).astype(int)

# Generate the confusion matrix
cm = confusion_matrix(y_test_int, y_predicted_int)

# Plotting the confusion matrix using matplotlib directly
fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.matshow(cm, cmap=plt.cm.Blues)  # Choose a color map suitable for the data
plt.colorbar(cax)

# Set labels to be more informative
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')

# Adding text annotation
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, f'{val}', ha='center', va='center', color='white')

# Display the plot in the Streamlit app
st.markdown("<h3 style='color: yellow;'>Confusion Matrix:</h3>", unsafe_allow_html=True)
st.pyplot(fig)
st.write("")  # Add an empty line for spacing



multiline_string = """
INFERENCE:
Visualizes the confusion matrix for the neural network model predictions, showing true positives, true negatives, false positives, and false negatives.

The matrix shows a higher number of both true positives and false negatives, suggesting that the model has some limitations distinguishing class 1."""
st.write(multiline_string)
st.write("")  # Add an empty line for spacing
