import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Collect Data
data = pd.read_csv('data.csv') 

returns_asset = data['Asset Returns']
returns_market = data['Market Index Returns']

# Step 2: Calculate Returns (if not already available)

# Step 3: Set up the Regression Model
X = sm.add_constant(returns_market)  
model = sm.OLS(returns_asset, X)    

# Step 4: Estimate Parameters
results = model.fit()  
alpha, beta = results.params[0], results.params[1]  

# Step 5: Interpret Beta Coefficient
print("Beta:", beta)

# Plotting

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(returns_market, returns_asset, label='Data Points',s=90)
ax.plot(returns_market, alpha + beta * returns_market, color='red', label='Regression Line')
ax.set_xlabel('Market Index Returns')
ax.set_ylabel('Asset Returns')
ax.set_title('Linear Regression: Asset Returns vs. Market Index Returns')
ax.legend()
plt.show()