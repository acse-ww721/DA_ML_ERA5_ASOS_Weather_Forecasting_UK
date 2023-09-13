# Name: Wenqi Wang
# Github username: acse-ww721

import matplotlib.pyplot as plt
import numpy as np

# Simulated data, replace with your own data in real applications
steps = 144
stepwise_acc_values = stepwise_acc_values_1  # For demonstration purposes
stepwise_acc_std_values = stepwise_acc_std_values_1  # For demonstration purposes

# x-axis represents steps, here we assume each step represents an hour
x = np.arange(steps)

# Create a new figure
plt.figure(figsize=(10, 6))

# Plot the line chart of accuracy over steps
plt.plot(x, stepwise_acc_values, label="U-STN12", color="blue")

# # Use fill_between to add shading to display the standard deviation
# plt.fill_between(x, stepwise_acc_values - stepwise_acc_std_values,
#                  stepwise_acc_values + stepwise_acc_std_values, color='blue', alpha=0.2)

# Set the y-axis range
plt.ylim(0.3, 1.0)

# Set the x-axis ticks
xticks = np.arange(0, steps, 12)
# yticks = np.arange(0.1, 0.6, 0.2)
plt.xticks(xticks)
# plt.yticks(yticks)

# Set title and axis labels
plt.title("Accuracy over Steps over 50 initial random conditions with no noise")
plt.xlabel("Hours")
plt.ylabel("Accuracy (ACC)")

# Show the legend
plt.legend()

# Display the plot
plt.tight_layout()
plt.grid(True)
plt.show()
