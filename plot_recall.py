import matplotlib.pyplot as plt
import pandas as pd

# Load results.csv for Y1 and Y2
y1_results = pd.read_csv(r'D:/yolov5-master/runs/train/exp4/results.csv')
y2_results = pd.read_csv(r'D:/yolov5-master/runs/train/exp5/results.csv')

# Create a figure and axis for subplots
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot Precision-Recall curve for Model Y1
ax.plot(y1_results['metrics/recall'], y1_results['metrics/precision'], label='Model Y1', linestyle='-', color='blue')

# Plot Precision-Recall curve for Model Y2
ax.plot(y2_results['metrics/recall'], y2_results['metrics/precision'], label='Model Y2', linestyle='-', color='green')

# Add labels and title
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve for Model Y1 and Y2')

# Add legend to distinguish models
ax.legend()

# Display the plot
plt.show()
