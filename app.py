import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Social Media Addiction Analysis", layout="wide")
# Load data
df = pd.read_csv("Students Social Media Addiction.csv")
import base64

# Background and sidebar styling
def apply_custom_styles(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    custom_css = f"""
    <style>
    /* Background full screen */
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; color: #f1c40f;'>Student Social Media Addiction List</h1>
    <p style='text-align: center; color: #ecf0f1;'>This data shows you how much of students today are addicted to social media</p>
    <hr style='border: 1px solid #555;' />
""", unsafe_allow_html=True)

# Call function with your background image file
apply_custom_styles("background.jpg")  # Make sure the image file is 1:1 and in your app directory
st.markdown("""
    <h1 style='text-align: center; color: #ffff;'>Dataset</h1>
""",unsafe_allow_html=True)
st.write(df)    
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

# Display in Streamlit
st.text("DataFrame Info:")
st.text(info_str)

st.subheader("Missing Values in Each Column")
st.dataframe(df.isnull().sum().reset_index().rename(columns={0: 'Missing Values', 'index': 'Column'}))

# Create age groups
bins = [15, 20, 25, 30, 35]
labels = ['16-20', '21-25', '26-30', '31-35']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Show Age Group column
st.subheader("Age Groups based on Age Column")
st.dataframe(df[['Age', 'Age_Group']])

usage_bins = [0, 2, 4, 6, 12, 24]
usage_labels = ['Minimal (0-2h)', 'Moderate (2-4h)', 'High (4-6h)', 'Very High (6-12h)', 'Extreme (12h+)']

# Categorize usage
df['Usage_Category'] = pd.cut(df['Avg_Daily_Usage_Hours'], bins=usage_bins, labels=usage_labels)

# Show categorized results
st.subheader("Categorized Social Media Usage")
st.dataframe(df[['Avg_Daily_Usage_Hours', 'Usage_Category']])

# Assume 'df' is already defined and contains 'Sleep_Hours_Per_Night'

# Categorize sleep hours
sleep_bins = [0, 5, 7, 9, 12]
sleep_labels = ['Poor (<5h)', 'Fair (5-7h)', 'Good (7-9h)', 'Excellent (9h+)']
df['Sleep_Category'] = pd.cut(df['Sleep_Hours_Per_Night'], bins=sleep_bins, labels=sleep_labels)

# Display the result
st.subheader("Sleep Quality Categories")
st.dataframe(df[['Sleep_Hours_Per_Night', 'Sleep_Category']])


# Assuming df is already loaded and contains a 'Gender' column
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Show result
st.subheader("Gender Column After Mapping")
st.dataframe(df[['Gender']])

# Assuming 'df' already has the 'Affects_Academic_Performance' column
df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})

# Show result
st.subheader("Mapped 'Affects_Academic_Performance' Column")
st.dataframe(df[['Affects_Academic_Performance']])


# Assuming 'df' is already defined and has 'Academic_Level' column
lab = LabelEncoder()
df['Academic_Level'] = lab.fit_transform(df['Academic_Level'])

# Show result
st.subheader("Label Encoded Academic Level")
st.dataframe(df[['Academic_Level']])

# Independent encoders for each column
lab_academic = LabelEncoder()
df['Academic_Level'] = lab_academic.fit_transform(df['Academic_Level'])

lab_relationship = LabelEncoder()
df['Relationship_Status'] = lab_relationship.fit_transform(df['Relationship_Status'])

# Show results
st.subheader("Label Encoded Academic Level & Relationship Status")
st.dataframe(df[['Academic_Level', 'Relationship_Status']])

cat_cols = ['Academic_Level', 'Relationship_Status', 'Country', 'Most_Used_Platform', 'Age_Group', 'Usage_Category', 'Sleep_Category']

# Assume df is already defined and contains all columns listed
cat_cols = ['Academic_Level', 'Relationship_Status', 'Country', 'Most_Used_Platform',
            'Age_Group', 'Usage_Category', 'Sleep_Category']

# Apply label encoding to each categorical column
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Show encoded dataframe
st.subheader("Label Encoded Categorical Columns")
st.dataframe(df[cat_cols])

# Assuming 'df' is already loaded and 'cat_cols' is defined
cat_cols = ['Academic_Level', 'Relationship_Status', 'Country', 'Most_Used_Platform',
            'Age_Group', 'Usage_Category', 'Sleep_Category']

# One-hot encode categorical columns
data = pd.get_dummies(df[cat_cols], drop_first=True)

# Display the encoded data
st.subheader("One-Hot Encoded Categorical Features")
st.dataframe(data)

# Drop the original categorical columns
df.drop(cat_cols, axis=1, inplace=True)

# Then concatenate the one-hot encoded columns if needed
df = pd.concat([df, data], axis=1)

# Show final DataFrame
st.subheader("Final Processed DataFrame")
st.dataframe(df)

num_cols = ['Avg_Daily_Usage_Hours','Sleep_Hours_Per_Night','Mental_Health_Score',
            'Conflicts_Over_Social_Media','Addicted_Score','Age']

# Assuming df is your full, cleaned DataFrame
st.subheader("Summary Statistics for Numerical Columns")
st.dataframe(df[num_cols].describe().T)

# Assume 'df' and 'num_cols' are already defined
# num_cols = [...]
#Handling Numerical Attributes -->Standardization
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
# Initialize the scaler
std = StandardScaler()

# Apply standardization
df[num_cols] = std.fit_transform(df[num_cols])

# Show result
st.subheader("Standardized Numerical Columns")
st.dataframe(df[num_cols].head())


# Concatenate original df with one-hot encoded data
new_data = pd.concat([df, data], axis=1)
# Remove duplicate columns by name
new_data = new_data.loc[:, ~new_data.columns.duplicated()]
# Remove duplicate columns before display
new_data = new_data.loc[:, ~new_data.columns.duplicated()]

# Now display in Streamlit without error
st.dataframe(new_data.head())

# Display result
st.subheader("Final Combined DataFrame")
st.dataframe(new_data.head())

# Drop the Age column
new_data.drop(['Age'], axis=1, inplace=True)

# Show updated dataframe
st.subheader("Data After Dropping 'Age' Column")
st.dataframe(new_data.head())

# Drop the Student_ID column
new_data.drop(['Student_ID'], axis=1, inplace=True)

# Display the cleaned data
st.subheader("Data After Dropping 'Student_ID'")
st.dataframe(new_data.head())

x = new_data.drop(['Addicted_Score'], axis=1)
y = new_data['Addicted_Score']


st.subheader("Shape of Features and Target")
st.write(f"x shape: {x.shape}")
st.write(f"y shape: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

st.subheader("Train-Test Split")
st.write(f"Training features: {x_train.shape}")
st.write(f"Testing features: {x_test.shape}")
st.write(f"Training labels: {y_train.shape}")
st.write(f"Testing labels: {y_test.shape}")

reg = LinearRegression()
reg.fit(x_train, y_train)

st.subheader("Linear Regression Model Coefficients")
coeff_df = pd.DataFrame(reg.coef_, index=x.columns, columns=["Coefficient"])
st.dataframe(coeff_df)

st.write(f"Intercept: {reg.intercept_}")

pred = reg.predict(x_test)

st.subheader("Actual vs Predicted Addicted Scores (Linear Regression)")
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': pred})
st.dataframe(comparison.head(10))  # Display first 10 for readability

cdf = pd.DataFrame(reg.coef_, index=x_train.columns, columns=['Coefficients'])

st.subheader("Linear Regression Coefficients")
st.dataframe(cdf)


fig = px.scatter(new_data,
                 x='Mental_Health_Score',
                 y='Addicted_Score',
                 title='Mental Health Score vs. Addicted Score with Linear Regression',
                 trendline="ols",
                 color='Addicted_Score',  # Color points based on 'Addicted_Score'
                 trendline_color_override='green') # Change trendline color to green
st.subheader("Mental Health vs Addicted Score (with Regression Line)")
st.plotly_chart(fig, use_container_width=True)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred) * 100

st.subheader("Model Evaluation Metrics")
st.write(f"📉 Mean Squared Error (MSE): {mse:.2f}")
st.write(f"📈 R-Squared Value: {r2:.2f}%")

pred_df = pd.DataFrame({
    'Actual_values': y_test,
    'Predicted_values': pred,
    'Difference': y_test - pred
})

st.subheader("Actual vs Predicted Values with Difference")
st.dataframe(pred_df.head(20))  # Show first 20 rows for readability

from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
import pandas as pd

# Assume x is already defined (your feature set)
poly_features = PolynomialFeatures(degree=2)
x_poly = poly_features.fit_transform(x)

# Create a DataFrame to visualize the polynomial features
poly_feature_names = poly_features.get_feature_names_out(x.columns)
x_poly_df = pd.DataFrame(x_poly, columns=poly_feature_names)

st.subheader("Polynomial Features (Degree = 2)")
st.dataframe(x_poly_df.head())

st.subheader("Original Feature Row (x.iloc[0])")
st.write(x.iloc[0])

st.subheader("Transformed Polynomial Feature Row (x_poly[0])")
st.write(x_poly[0])

poly_lr = LinearRegression()
poly_lr.fit(x_poly, y)

coeff_df = pd.DataFrame(poly_lr.coef_, index=poly_feature_names, columns=['Coefficient'])
st.subheader("Polynomial Regression Coefficients")
st.dataframe(coeff_df)

from sklearn.metrics import mean_squared_error, r2_score

poly_pred = poly_lr.predict(x_poly)

mse_poly = mean_squared_error(y, poly_pred)
r2_poly = r2_score(y, poly_pred) * 100

st.subheader("Polynomial Regression Performance")
st.write(f"📉 Mean Squared Error: {mse_poly:.2f}")
st.write(f"📈 R-Squared Value: {r2_poly:.2f}%")


# Assuming you have your data (x, y) from your previous code

# --- Fix: Create a new PolynomialFeatures object for the single feature ---
# Select the feature you want to plot (e.g., 'Mental_Health_Score')
x_plot_feature = x[['Mental_Health_Score']]

# Create a new PolynomialFeatures object for this single feature
poly_features_plot = PolynomialFeatures(degree=2) # Use the same degree as your original model

# Fit and transform this single feature range
x_plot_feature_poly = poly_features_plot.fit_transform(x_plot_feature)

# Create a Linear Regression model and fit it to the single feature
poly_lr_plot = LinearRegression()
poly_lr_plot.fit(x_plot_feature_poly, y)
# --- End of Fix ---

# 1. Calculate polynomial regression predictions for plotting
# Create a range of x values to plot the smooth polynomial curve for the selected feature
x_range = np.linspace(x_plot_feature['Mental_Health_Score'].min(), x_plot_feature['Mental_Health_Score'].max(), 100).reshape(-1, 1)

# Create polynomial features for the x_range using the new poly_features_plot object
x_range_poly = poly_features_plot.transform(x_range)

# Predict the y values using the new polynomial model
y_poly_pred_range = poly_lr_plot.predict(x_range_poly)

# 2. Create a scatter plot of the original data
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x['Mental_Health_Score'],
    y=y,
    mode='markers',
    name='Original Data'
))

# 3. Add the polynomial trendline as a separate trace
fig.add_trace(go.Scatter(
    x=x_range.flatten(),
    y=y_poly_pred_range,
    mode='lines',
    name='Polynomial Regression',
    line=dict(color='red', width=2) # You can change the color and width here
))

# Update layout for title and labels
fig.update_layout(
    title='Polynomial Regression Fit (Mental Health Score vs. Addicted Score)',
    xaxis_title='Mental_Health_Score',
    yaxis_title='Addicted_Score'
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Polynomial Regression: Mental Health Score vs Addicted Score")

# Age distribution (Histogram)
fig_age = px.histogram(df,
                      x='Age',
                      nbins=15,
                      title='Age Distribution of Students',
                      color_discrete_sequence=['lightblue']) # Change color

# Gender distribution (Pie chart)
gender_counts = df['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count'] # Rename columns for Plotly
fig_gender = px.pie(gender_counts,
                    values='Count',
                    names='Gender',
                    title='Gender Distribution',
                    color='Gender', # Color by gender
                    color_discrete_sequence=px.colors.qualitative.Set1) # Change color palette

# Display the plots
st.subheader("📊 Age Distribution of Students")
st.plotly_chart(fig_age, use_container_width=True,key=fig_age)

st.subheader("🧑‍🤝‍🧑 Gender Distribution")
st.plotly_chart(fig_gender, use_container_width=True)

# Country distribution
country_counts = new_data['Country'].value_counts().head(10).reset_index()
country_counts.columns = ['Country', 'Count']
fig_country = px.bar(
    country_counts,
    x='Country',
    y='Count',
    title='Top 10 Countries',
    color='Country',
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Academic level distribution
academic_counts = new_data['Academic_Level'].value_counts().reset_index()
academic_counts.columns = ['Academic_Level', 'Count']
fig_academic = px.bar(
    academic_counts,
    x='Academic_Level',
    y='Count',
    title='Academic Level Distribution',
    color='Academic_Level',
    color_discrete_sequence=px.colors.qualitative.Set2
)

# Display in Streamlit
st.subheader("🌍 Top 10 Countries")
st.plotly_chart(fig_country, use_container_width=True,key=fig_country)

st.subheader("🎓 Academic Level Distribution")
st.plotly_chart(fig_academic, use_container_width=True,key=fig_academic)

fig = px.scatter(
    new_data,
    x='Mental_Health_Score',
    y='Addicted_Score',
    color='Addicted_Score',
    title='Mental Health Score vs. Addicted Score',
    color_continuous_scale='bluered'
)

st.subheader("🧠 Mental Health Score vs. Addicted Score")
st.plotly_chart(fig, use_container_width=True)


fig = px.bar(
    new_data,
    x='Academic_Level',
    y='Addicted_Score',
    color='Addicted_Score',
    title='Academic Level vs. Addicted Score',
    color_continuous_scale='Viridis'  # Optional: custom color scale
)

st.subheader("🎓 Academic Level vs. Addicted Score")
st.plotly_chart(fig, use_container_width=True)


# Average Daily Usage Hours (Histogram)
fig = px.histogram(df,
                   x='Avg_Daily_Usage_Hours',
                   nbins=20,
                   title='Average Daily Social Media Usage Hours',
                   color_discrete_sequence=['teal'])

# Add a vertical line for the mean
mean_usage = df['Avg_Daily_Usage_Hours'].mean()
fig.add_shape(type="line",
              x0=mean_usage, y0=0, x1=mean_usage, y1=1,
              xref='x', yref='paper',
              line=dict(color="purple", width=2, dash="dot"),
              name=f'Mean: {mean_usage:.2f} hours')

# Add legend line (invisible scatter to create legend entry)
fig.add_trace(go.Scatter(x=[mean_usage], y=[0],
                         mode='lines',
                         line=dict(color="purple", width=2, dash="dot"),
                         name=f'Mean: {mean_usage:.2f} hours'))

# Update layout
fig.update_layout(xaxis_title='Hours',
                  yaxis_title='Count')

# Display in Streamlit
st.subheader("📱 Average Daily Social Media Usage (with Mean Line)")
st.plotly_chart(fig, use_container_width=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create binary classification label using median threshold
threshold = new_data['Addicted_Score'].median()
new_data['Addiction_Category'] = new_data['Addicted_Score'].apply(lambda x: 'High' if x > threshold else 'Low')

# Prepare features and target
x = new_data.drop(['Addicted_Score', 'Addiction_Category'], axis=1)
y_clf = new_data['Addiction_Category']

# Train-test split
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(x, y_clf, test_size=0.25, random_state=42)

# Train Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(x_train_clf, y_train_clf)

# Predict and evaluate
y_pred_clf = log_reg_model.predict(x_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
conf_mat = confusion_matrix(y_test_clf, y_pred_clf)

# Display results in Streamlit
st.subheader("🔍 Addiction Category Classification: Logistic Regression")
st.write(f"✅ **Accuracy:** {acc:.2f}")

st.write("📄 **Classification Report**")
st.dataframe(pd.DataFrame(report).transpose())

st.write("🧮 **Confusion Matrix**")
fig, ax = plt.subplots()
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

import plotly.figure_factory as ff

# 1. Create classification label
threshold = new_data['Addicted_Score'].median()
new_data['Addiction_Category'] = new_data['Addicted_Score'].apply(lambda x: 'High' if x > threshold else 'Low')

# 2. Feature and target setup
x = new_data.drop(['Addicted_Score', 'Addiction_Category'], axis=1)
y_clf = new_data['Addiction_Category']

# 3. Train-test split
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(
    x, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)

# 4. Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(x_train_clf, y_train_clf)

# 5. Predictions and evaluation
y_pred_clf = log_reg_model.predict(x_test_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
cm = confusion_matrix(y_test_clf, y_pred_clf)
class_labels = log_reg_model.classes_

# 6. Streamlit display
st.subheader("🧠 Logistic Regression - Addiction Classification")
st.write(f"✅ **Accuracy:** `{acc:.2f}`")

st.write("📊 **Classification Report**")
st.dataframe(pd.DataFrame(report).transpose())

# 7. Plotly Confusion Matrix
z = cm
x_labels = class_labels
y_labels = class_labels

import plotly.figure_factory as ff

# Example
cm = confusion_matrix(y_test_clf, y_pred_clf)
z = cm.tolist()  # <- Convert from array to list

fig = ff.create_annotated_heatmap(
    z=z,
    x=['Predicted Low', 'Predicted High'],
    y=['Actual Low', 'Actual High'],
    annotation_text=[[str(cell) for cell in row] for row in z],
    colorscale='Viridis'
)


st.plotly_chart(fig, use_container_width=True, key='confusion_matrix')

# --- Class Distribution Bar Chart (Actual vs Predicted) ---

# Convert to pandas Series (ensure correct indexing)
y_test_clf_series = pd.Series(y_test_clf, name='Actual')
y_pred_clf_series = pd.Series(y_pred_clf, name='Predicted', index=y_test_clf_series.index)

# Combine for comparison
comparison_df = pd.concat([y_test_clf_series, y_pred_clf_series], axis=1)

# Actual class counts
actual_counts = comparison_df['Actual'].value_counts().reset_index()
actual_counts.columns = ['Category', 'Count']
actual_counts['Type'] = 'Actual'

# Predicted class counts
predicted_counts = comparison_df['Predicted'].value_counts().reset_index()
predicted_counts.columns = ['Category', 'Count']
predicted_counts['Type'] = 'Predicted'

# Combine both
combined_counts = pd.concat([actual_counts, predicted_counts])

# Bar chart
fig_class_comparison = px.bar(
    combined_counts,
    x='Category',
    y='Count',
    color='Type',
    barmode='group',
    title='🆚 Actual vs. Predicted Class Distribution',
    color_discrete_sequence=px.colors.qualitative.Set2
)

st.subheader("📊 Class Prediction Comparison")
st.plotly_chart(fig_class_comparison, use_container_width=True)

# --- Correctness Scatter Plot ---

# Merge prediction results into original data
new_data_with_predictions = new_data.loc[y_test_clf_series.index].copy()
new_data_with_predictions['Actual_Category'] = y_test_clf_series
new_data_with_predictions['Predicted_Category'] = y_pred_clf_series
new_data_with_predictions['Prediction_Correct'] = (
    new_data_with_predictions['Actual_Category'] == new_data_with_predictions['Predicted_Category']
)

# Choose feature to plot
feature_to_plot = 'Mental_Health_Score'  # You can make this dynamic with a dropdown if needed

# Scatter plot
fig_correctness = px.scatter(
    new_data_with_predictions,
    x=feature_to_plot,
    y='Addicted_Score',
    color='Prediction_Correct',
    title=f'🔍 {feature_to_plot} vs. Addicted Score (Prediction Accuracy)',
    hover_data=['Actual_Category', 'Predicted_Category'],
    color_discrete_map={True: 'green', False: 'red'}
)

st.subheader("🎯 Prediction Correctness Visualization")
st.plotly_chart(fig_correctness, use_container_width=True)

from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)

# Extract class labels from the model
class_labels = log_reg_model.classes_

# Flatten confusion matrix
tn, fp, fn, tp = cm.ravel()

# Create structured data
metrics_data = {
    'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
    'Count': [tn, fp, fn, tp],
    'Description': [
        f'✅ Correctly predicted as {class_labels[0]}',
        f'❌ Predicted {class_labels[1]} but actually {class_labels[0]}',
        f'❌ Predicted {class_labels[0]} but actually {class_labels[1]}',
        f'✅ Correctly predicted as {class_labels[1]}'
    ]
}

df_metrics = pd.DataFrame(metrics_data)

# Create Plotly bar chart
fig = go.Figure(data=[go.Bar(
    x=df_metrics['Metric'],
    y=df_metrics['Count'],
    text=df_metrics['Count'],
    textposition='auto',
    hovertext=df_metrics['Description'],
    marker=dict(color=['#60A5FA', '#F87171', '#FBBF24', '#4ADE80'])  # Blue, Red, Orange, Green
)])

fig.update_layout(
    title_text='🔎 Confusion Matrix Breakdown',
    xaxis_title='Metric',
    yaxis_title='Count',
    plot_bgcolor='rgba(0,0,0,0)'
)

# Streamlit output
st.subheader("📊 Confusion Matrix Component Breakdown")
st.plotly_chart(fig, use_container_width=True)


# Generate the confusion matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)

# Get the class labels from the trained model
class_labels = log_reg_model.classes_

# Dynamically get the index for each label
low_index = class_labels.tolist().index('Low')
high_index = class_labels.tolist().index('High')

positions = {
    'TN': (low_index, low_index),
    'FP': (high_index, low_index),
    'FN': (low_index, high_index),
    'TP': (high_index, high_index)
}

tn, fp, fn, tp = cm.ravel()

# Create the confusion matrix figure
fig = go.Figure()

# Colored rectangles for each quadrant
fig.add_shape(type="rect", x0=positions['TN'][0]-0.5, y0=positions['TN'][1]-0.5,
              x1=positions['TN'][0]+0.5, y1=positions['TN'][1]+0.5,
              line=dict(color="gray"), fillcolor="lightblue", opacity=0.7)

fig.add_shape(type="rect", x0=positions['FP'][0]-0.5, y0=positions['FP'][1]-0.5,
              x1=positions['FP'][0]+0.5, y1=positions['FP'][1]+0.5,
              line=dict(color="gray"), fillcolor="salmon", opacity=0.7)

fig.add_shape(type="rect", x0=positions['FN'][0]-0.5, y0=positions['FN'][1]-0.5,
              x1=positions['FN'][0]+0.5, y1=positions['FN'][1]+0.5,
              line=dict(color="gray"), fillcolor="orange", opacity=0.7)

fig.add_shape(type="rect", x0=positions['TP'][0]-0.5, y0=positions['TP'][1]-0.5,
              x1=positions['TP'][0]+0.5, y1=positions['TP'][1]+0.5,
              line=dict(color="gray"), fillcolor="lightgreen", opacity=0.7)

# Add annotations (counts)
fig.add_annotation(x=positions['TN'][0], y=positions['TN'][1], text=str(tn), showarrow=False, font=dict(size=20))
fig.add_annotation(x=positions['FP'][0], y=positions['FP'][1], text=str(fp), showarrow=False, font=dict(size=20))
fig.add_annotation(x=positions['FN'][0], y=positions['FN'][1], text=str(fn), showarrow=False, font=dict(size=20))
fig.add_annotation(x=positions['TP'][0], y=positions['TP'][1], text=str(tp), showarrow=False, font=dict(size=20))

# Layout and axis styling
fig.update_layout(
    title='🧩 Confusion Matrix Grid Representation',
    xaxis=dict(title='Predicted Label',
               tickvals=[low_index, high_index],
               ticktext=['Low', 'High']),
    yaxis=dict(title='True Label',
               tickvals=[low_index, high_index],
               ticktext=['Low', 'High'],
               autorange='reversed'),
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis_zeroline=False,
    yaxis_zeroline=False,
    margin=dict(t=50, b=50, l=50, r=50),
    height=400,
    width=400
)

# Display in Streamlit
st.subheader("📦 Confusion Matrix (Visual Grid)")
st.plotly_chart(fig, use_container_width=False)

from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42) # Using random_state for reproducibility

# Train the Decision Tree model
dt_model.fit(x_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_dt = dt_model.predict(x_test_clf)

# Evaluate the Decision Tree model
st.write("\n--- Decision Tree Model Evaluation ---")
st.write("Accuracy:", accuracy_score(y_test_clf, y_pred_dt))
st.write("\nClassification Report:\n", classification_report(y_test_clf, y_pred_dt))
st.write("\nConfusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_dt))

# Import the RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
# You can adjust n_estimators (number of trees) and other parameters
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(x_train_clf, y_train_clf)

# Make predictions on the test set
y_pred_rf = rf_model.predict(x_test_clf)

# Evaluate the Random Forest model
st.write("\n--- Random Forest Model Evaluation ---")
st.write("Accuracy:", accuracy_score(y_test_clf, y_pred_rf))
st.write("\nClassification Report:\n", classification_report(y_test_clf, y_pred_rf))
st.write("\nConfusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_rf))


import plotly.express as px
import pandas as pd

# Assuming you have trained your Decision Tree model (dt_model) and have your features (x)

# Get feature importances
feature_importances = dt_model.feature_importances_

# Create a DataFrame to hold feature names and their importances
importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create a bar chart of feature importances
fig = px.bar(importance_df,
             x='Importance',
             y='Feature',
             orientation='h', # Horizontal bars
             title='Decision Tree Feature Importances',
             labels={'Importance': 'Feature Importance Score', 'Feature': 'Feature'},
             color='Importance', # Color bars based on importance
             color_continuous_scale=px.colors.sequential.Viridis) # Choose a color scale

st.plotly_chart(fig, use_container_width=False)

import plotly.graph_objects as go
import pandas as pd

# Assuming you have trained your Random Forest model (rf_model) and have your features (x)

# Get feature importances from the Random Forest model
feature_importances_rf = rf_model.feature_importances_

# Create a DataFrame to hold feature names and their importances
importance_df_rf = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances_rf})

# Sort the DataFrame by importance in ascending order for better visualization in a dot plot
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=True)

# Create a dot plot
scatter_plot= go.Figure()

scatter_plot.add_trace(go.Scatter(
    x=importance_df_rf['Importance'],
    y=importance_df_rf['Feature'],
    mode='markers+lines', # Show points and connect them with lines
    marker=dict(size=10, color=importance_df_rf['Importance'], colorscale='Viridis'), # Color points by importance
    line=dict(color='gray', width=1), # Color and style of the connecting lines
    hovertext=importance_df_rf['Importance'].round(4), # Show importance on hover
    hoverinfo='x+y+text'
))

scatter_plot.update_layout(
    title='Random Forest Feature Importances (Dot Plot)',
    xaxis_title='Feature Importance Score',
    yaxis_title='Feature',
    yaxis=dict(autorange="reversed"), # Reverse y-axis to have the most important at the top
    margin=dict(l=150, r=20, t=50, b=50) # Adjust margins to accommodate long feature names
)

st.plotly_chart(scatter_plot, use_container_width=True)

from sklearn.datasets import make_classification

# 1. Create a synthetic dataset with well-separated classes
X, y = make_classification(n_samples=200, # Number of samples
                           n_features=2,  # Number of features (for easy visualization)
                           n_informative=2, # All features are informative
                           n_redundant=0, # No redundant features
                           n_clusters_per_class=1, # One cluster per class
                           flip_y=0.01, # Small amount of noise
                           random_state=42) # For reproducibility

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 3. Train Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 4. Train Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)

# 5. Make predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_dt = dt_clf.predict(X_test)

# 6. Evaluate Accuracy
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

st.write(f"Logistic Regression Accuracy: {accuracy_log_reg:.4f}")
st.write(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
plt_scat, ax = plt.subplots()
# 7. (Optional) Visualize the dataset and decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.8)

# Plot for Logistic Regression
plot_decision_boundary(log_reg, X_test, y_test, "Logistic Regression Decision Boundary")

# Plot for Decision Tree
plot_decision_boundary(dt_clf, X_test, y_test, "Decision Tree Decision Boundary")

# Create figure and plot

ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
ax.set_title('Plot')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# Show plot in Streamlit
st.pyplot(plt_scat)
import streamlit as st
import pandas as pd
import plotly.express as px


