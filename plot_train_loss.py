import matplotlib.pyplot as plt
import pandas as pd

# Load results.csv for Y1 and Y2
y1_results = pd.read_csv(r'D:/yolov5-master/runs/train/exp4/results.csv')
y2_results = pd.read_csv(r'D:/yolov5-master/runs/train/exp5/results.csv')

# Calculate total loss for Model Y1 and Model Y2 (box_loss + cls_loss + obj_loss)
y1_results['total_loss'] = y1_results['train/box_loss'] + y1_results['train/obj_loss'] + y1_results['train/cls_loss']
y2_results['total_loss'] = y2_results['train/box_loss'] + y2_results['train/obj_loss'] + y2_results['train/cls_loss']

# Plot total loss for Model Y1
plt.plot(y1_results['epoch'], y1_results['total_loss'], label='Model Y1 Total Loss', color='blue')

# Plot total loss for Model Y2
plt.plot(y2_results['epoch'], y2_results['total_loss'], label='Model Y2 Total Loss', color='green')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Training Loss for Model Y1 and Y2')

# Add legend
plt.legend()

# Show the plot
plt.show()
