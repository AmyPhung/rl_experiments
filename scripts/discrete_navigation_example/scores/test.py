import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('scores.csv')

df['averaged'] = df.iloc[:,0].rolling(window=100).mean()
plt.plot(df['averaged'])
plt.show()
