import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import mplcursors
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('./data/ads.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print(df.head())

colors = {'TV': 'green', 'Radio': 'orange', 'Newspaper': 'red'}

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

# TV vs Sales
sns.scatterplot(ax=axes[0], x='TV', y='Sales', data=df, color=colors['TV'])
axes[0].set_title('TV Advertising vs Sales')
axes[0].set_xlabel('TV Advertising Budget')
axes[0].set_ylabel('Sales')
mplcursors.cursor(axes[0], hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"TV: {sel.target[0]:.2f}, Sales: {sel.target[1]:.2f}"))

# Radio vs Sales
sns.scatterplot(ax=axes[1], x='Radio', y='Sales', data=df, color=colors['Radio'])
axes[1].set_title('Radio Advertising vs Sales')
axes[1].set_xlabel('Radio Advertising Budget')
axes[1].set_ylabel('Sales')
mplcursors.cursor(axes[1], hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"Radio: {sel.target[0]:.2f}, Sales: {sel.target[1]:.2f}"))

# Newspaper vs Sales
sns.scatterplot(ax=axes[2], x='Newspaper', y='Sales', data=df, color=colors['Newspaper'])
axes[2].set_title('Newspaper Advertising vs Sales')
axes[2].set_xlabel('Newspaper Advertising Budget')
axes[2].set_ylabel('Sales')
mplcursors.cursor(axes[2], hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"Newspaper: {sel.target[0]:.2f}, Sales: {sel.target[1]:.2f}"))

plt.tight_layout()
plt.show()

# Heatmap of Correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Residual Plot
plt.figure(figsize=(8, 6))
sns.residplot(x=y_test, y=y_pred, lowess=True, line_kws={'color': 'red', 'lw': 1}, color='blue')
plt.title('Residuals vs Fitted')
plt.xlabel('Actual Sales')
plt.ylabel('Residuals')
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"Actual: {sel.target[0]:.2f}, Residual: {sel.target[1]:.2f}"))
plt.show()

# Pairplot
sns.pairplot(df, diag_kind='kde', kind='reg', markers='+', plot_kws={'line_kws': {'color': 'red'}}, palette=colors)
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"x: {sel.target[0]:.2f}, y: {sel.target[1]:.2f}"))
plt.show()

# Regression plot
fig, axes = plt.subplots(1, 3, figsize=(24, 6))

for idx, feature in enumerate(['TV', 'Radio', 'Newspaper']):
    sns.regplot(ax=axes[idx], x=feature, y='Sales', data=df, ci=None, line_kws={'color': 'red'}, scatter_kws={'color': colors[feature]})
    axes[idx].set_title(f'{feature} Advertising vs Sales')
    axes[idx].set_xlabel(f'{feature} Advertising Budget')
    axes[idx].set_ylabel('Sales')
    mplcursors.cursor(axes[idx], hover=True).connect("add", lambda sel, feature=feature: sel.annotation.set_text(
        f"{feature}: {sel.target[0]:.2f}, Sales: {sel.target[1]:.2f}"))

plt.tight_layout()
plt.show()

# Distribution plot
distribution_colors = {'TV': 'purple', 'Radio': 'brown', 'Newspaper': 'pink', 'Sales': 'cyan'}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, feature in enumerate(['TV', 'Radio', 'Newspaper', 'Sales']):
    sns.histplot(ax=axes[idx//2, idx%2], data=df[feature], kde=True, color=distribution_colors[feature])
    axes[idx//2, idx%2].set_title(f'Distribution of {feature}')
    axes[idx//2, idx%2].set_xlabel(f'{feature}')
    axes[idx//2, idx%2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# PairGrid
g = sns.PairGrid(df, vars=['TV', 'Radio', 'Newspaper', 'Sales'])
g.map_upper(plt.scatter, color='blue')
g.map_lower(sns.kdeplot, cmap='Blues_d')
g.map_diag(sns.histplot, kde_kws={'color': 'r', 'lw': 1.5})
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"x: {sel.target[0]:.2f}, y: {sel.target[1]:.2f}"))
plt.show()
