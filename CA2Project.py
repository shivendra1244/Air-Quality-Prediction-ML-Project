
# AIR QUALITY ANALYSIS PROJECT

# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix
)
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import joblib

sns.set_theme(style="whitegrid")

# Load Dataset 
df = pd.read_csv(
    r"C:\Users\Shiva\OneDrive\Desktop\CA\Air Quality.csv"
)

# Basic Exploration 
print("\nDataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# DATA PREPROCESSING 
df.fillna(df.mean(numeric_only=True), inplace=True)

# Rename columns for consistency
df.rename(columns={
    'PM2.5': 'pollutant_min',
    'PM10': 'pollutant_max',
    'AQI': 'pollutant_avg'
}, inplace=True)

# Encode categorical column
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# HISTOGRAM 
plt.figure(figsize=(8,5))
sns.histplot(df['pollutant_avg'], kde=True)
plt.title("Distribution of Average Pollution")
plt.show()

# CORRELATION HEATMAP 
plt.figure(figsize=(6,4))
sns.heatmap(df[['pollutant_min','pollutant_max','pollutant_avg']].corr(),
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# RADAR CHART: Top 10 Cities 
city_avg = df.groupby("city")["pollutant_avg"].mean().sort_values(ascending=False).head(10)
angles = np.linspace(0, 2*np.pi, len(city_avg), endpoint=False).tolist()
angles += angles[:1]
values = city_avg.tolist() + city_avg.tolist()[:1]

plt.figure(figsize=(8,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
plt.xticks(angles[:-1], city_avg.index)
plt.title("Top 10 Cities by Pollution")
plt.show()


# REGRESSION MODELS

X = df[['pollutant_min', 'pollutant_max', 'city_encoded']]
y = df['pollutant_avg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple Linear Regression
slr = LinearRegression()
slr.fit(X_train[['pollutant_min']], y_train)
y_pred_slr = slr.predict(X_test[['pollutant_min']])
print("\nSimple Linear Regression R2:", r2_score(y_test, y_pred_slr))

# Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)
print("\nMultiple Linear Regression")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df[['pollutant_min']])
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)
poly_reg = LinearRegression()
poly_reg.fit(Xp_train, yp_train)
print("\nPolynomial Regression R2:", r2_score(yp_test, poly_reg.predict(Xp_test)))


# CLASSIFICATION MODELS

def pollution_level(x):
    if x <= 50:
        return 0  # Low
    elif x <= 100:
        return 1  # Moderate
    else:
        return 2  # High

df['pollution_class'] = df['pollutant_avg'].apply(pollution_level)

Xc = df[['pollutant_min','pollutant_max']]
yc = df['pollution_class']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

sc = StandardScaler()
Xc_train_s = sc.fit_transform(Xc_train)
Xc_test_s = sc.transform(Xc_test)

# Logistic Regression
log = LogisticRegression(max_iter=300)
log.fit(Xc_train_s, yc_train)
yc_pred_log = log.predict(Xc_test_s)
print("\nLogistic Regression Accuracy:", accuracy_score(yc_test, yc_pred_log))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(Xc_train, yc_train)
yc_pred_dt = dt.predict(Xc_test)
print("Decision Tree Accuracy:", accuracy_score(yc_test, yc_pred_dt))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(Xc_train, yc_train)
yc_pred_rf = rf.predict(Xc_test)
print("Random Forest Accuracy:", accuracy_score(yc_test, yc_pred_rf))

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(Xc_train, yc_train)
yc_pred_gb = gb.predict(Xc_test)
print("Gradient Boosting Accuracy:", accuracy_score(yc_test, yc_pred_gb))

# Naive Bayes
nb = GaussianNB()
nb.fit(Xc_train, yc_train)
yc_pred_nb = nb.predict(Xc_test)
print("Naive Bayes Accuracy:", accuracy_score(yc_test, yc_pred_nb))

# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(Xc_train_s, yc_train)
yc_pred_knn = knn.predict(Xc_test_s)
print("KNN Accuracy:", accuracy_score(yc_test, yc_pred_knn))


# CONFUSION MATRIX: Logistic Regression

from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(yc_test, yc_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low','Moderate','High'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix – Logistic Regression")
plt.show()


# CUMULATIVE GAIN CHART

y_score = log.predict_proba(Xc_test_s)[:,2]  # High pollution class
fpr, tpr, _ = roc_curve((yc_test==2).astype(int), y_score)

plt.figure(figsize=(6,4))
plt.plot(tpr, label="Cumulative Gain Curve")
plt.plot([0,1],[0,1],'--', color='grey')
plt.xlabel("Percentage of Samples")
plt.ylabel("Gain")
plt.title("Cumulative Gain Chart – Logistic Regression")
plt.legend()
plt.show()

# K-MEANS CLUSTERING

km = KMeans(n_clusters=3, random_state=42)
df['cluster'] = km.fit_predict(sc.fit_transform(Xc))

plt.figure(figsize=(6,4))
sns.scatterplot(
    x=df['pollutant_min'],
    y=df['pollutant_max'],
    hue=df['cluster'],
    palette='Set2'
)
plt.title("K-Means Clustering")
plt.show()

