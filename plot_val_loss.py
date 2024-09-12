import matplotlib.pyplot as plt
import pandas as pd

# Load results.csv for Y1 and Y2
y1_results = pd.read_csv(r'D:/yolov5-master/runs/train/exp4/results.csv')
y2_results = pd.read_csv(r'D:/yolov5-master/runs/train/exp5/results.csv')

# Calculate total validation loss for Model Y1 and Model Y2 (val/box_loss + val/cls_loss + val/obj_loss)
y1_results['val_total_loss'] = y1_results['val/box_loss'] + y1_results['val/cls_loss'] + y1_results['val/obj_loss']
y2_results['val_total_loss'] = y2_results['val/box_loss'] + y2_results['val/cls_loss'] + y2_results['val/obj_loss']

# Plot total validation loss for Model Y1
plt.plot(y1_results['epoch'], y1_results['val_total_loss'], label='Model Y1 Validation Loss', color='blue')

# Plot total validation loss for Model Y2
plt.plot(y2_results['epoch'], y2_results['val_total_loss'], label='Model Y2 Validation Loss', color='green')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Total Validation Loss for Model Y1 and Y2')

# Add legend
plt.legend()

# Show the plot
plt.show()
