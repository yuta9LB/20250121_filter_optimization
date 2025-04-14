import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('20250306_filter_opt_psr/20250306_filter_opt_psr.csv')

# Plot the data
plt.plot(df['iter'], df['fitness'])
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Fitness over time')
plt.savefig('20250306_filter_opt_psr/fitness.png')