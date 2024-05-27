import matplotlib.pyplot as plt
import numpy as np

# Data
age_groups = ['13-49', '50-69', '>70']
charades_percentage = [83.3, 16.7, 0]
etri_percentage = [50, 3, 47]
custom_percentage = [41.7, 8.3, 50]

# Bar colors
bar_colors = ["#367be3", "#db3763", "#8ac246"]

# X-axis positions for the groups
x = np.arange(len(age_groups))

# Width of the bars
width = 0.2

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each dataset's bars
ax.bar(x - width, charades_percentage, width, label='Charades', color=bar_colors[0])
ax.bar(x, etri_percentage, width, label='ETRI', color=bar_colors[1])
ax.bar(x + width, custom_percentage, width, label='Question answering', color=bar_colors[2])

# Set the labels and title
ax.set_xlabel('Age Groups')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.set_ylim(0, 90)
ax.legend()

# Show the plot
plt.savefig('age_groups.png')