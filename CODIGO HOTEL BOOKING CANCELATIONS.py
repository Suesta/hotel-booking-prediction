import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Carga de datos
df = pd.read_csv('Hotel Reservations.csv')

# 2. Primer vistazo
print(df.head())
print(df.dtypes)

# 3. Conversión de tipos
categorical_cols = [
    'type_of_meal_plan',
    'market_segment_type',
    'room_type_reserved',
    'booking_status'
]
df[categorical_cols] = df[categorical_cols].astype('category')
df['repeated_guest'] = df['repeated_guest'].astype(bool)
df['required_car_parking_space'] = df['required_car_parking_space'].astype(bool)

numeric_cols = [
    'no_of_adults',
    'no_of_children',
    'no_of_week_nights',
    'no_of_weekend_nights',
    'lead_time',
    'avg_price_per_room',
    'no_of_special_requests',
    'no_of_previous_cancellations',
    'no_of_previous_bookings_not_canceled'
]

# 4. Estadísticas descriptivas
print(df.describe())
print(df.describe(include='category'))
print(df.isnull().sum())

for col in numeric_cols:
    print(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}")

# 5. Visualizaciones
# 5.1 Histogramas
for col in numeric_cols:
    plt.hist(df[col], bins=20, edgecolor='k')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 5.2 Contadores categorías
for col in ['booking_status', 'market_segment_type', 'room_type_reserved']:
    sns.countplot(x=col, data=df)
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45)
    plt.show()

# 5.3 Matriz de correlación
corr = df[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 5.4 Scatterplot lead_time vs avg_price_per_room
sns.scatterplot(x='lead_time', y='avg_price_per_room', hue='booking_status', data=df)
plt.title('Lead Time vs Avg Price per Room')
plt.show()

# 5.5 Detección de outliers
for col in ['lead_time', 'no_of_adults', 'avg_price_per_room', 'no_of_previous_cancellations']:
    sns.boxplot(x=df[col])
    plt.title(f'Outliers in {col}')
    plt.show()

# 6. Escalado de variables
for col in numeric_cols:
    if col != 'avg_price_per_room':
        df[f'{col}_scaled'] = (df[col] - df[col].mean()) / df[col].std()

features = [c for c in df.columns if c.endswith('_scaled')]
target = 'avg_price_per_room'

# 7. División train/test
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

# 8. Modelos

# 8.1 Regresión Lineal
lr = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Linear Regression RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# 8.2 Lasso & Ridge
for alpha in [0.001, 0.01, 0.1, 1.0]:
    lasso = Lasso(alpha=alpha).fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    print(f'Lasso (α={alpha}) RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    print(f'Ridge (α={alpha}) RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# 8.3 K-Nearest Neighbors
for k in [1, 3, 5, 10]:
    knn = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f'KNN (k={k}) RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))

# 8.4 Árbol de Decisión
dt = DecisionTreeRegressor(max_depth=10, min_samples_split=2).fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('Decision Tree RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_dt)))

# 8.5 Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest RMSE:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('Random Forest R²:', r2_score(y_test, y_pred_rf))
