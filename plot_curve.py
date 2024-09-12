import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def replot_pr_curve_from_csv(csv_path1, csv_path2, save_dir=Path("replot_pr_curve.png")):
    """Replots the precision-recall curve from a saved CSV file."""
    recalls1 = []
    precisions1 = []
    recalls2 = []
    precisions2 = []
    names = []
    mAP = None
    first1 = True
    # Read the data from the CSV file
    with open(csv_path1, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # read the first row (column names)
        names = [header.replace('Precision_', '') for header in headers[1:-1]]  # extract class names
        # mAP = headers[-1]  # last column is mAP
        mAP1 = 0
        for row in reader:
            if first1:
                mAP1 = row[-1]
                first1 =False

            recalls1.append(float(row[0]))  # first column is recall
            precisions1.append([float(p) for p in row[1:-1]])  # middle columns are precisions for each class

    recalls1 = np.array(recalls1)
    precisions1 = np.array(precisions1)
    first2 = True
    with open(csv_path2, mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # read the first row (column names)
        names = [header.replace('Precision_', '') for header in headers[1:-1]]  # extract class names
        # mAP = headers[-1]  # last column is mAP
        mAP2 = 0
        for row in reader:
            if first2:
                mAP2 = row[-1]
                first2 =False

            recalls2.append(float(row[0]))  # first column is recall
            precisions2.append([float(p) for p in row[1:-1]])  # middle columns are precisions for each class
    recalls2 = np.array(recalls2)
    precisions2 = np.array(precisions2)
    # Re-plot the precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    # if len(names) > 0 and len(names) < 21:  # display per-class legend if < 21 classes
    #     for i, name in enumerate(names):
    #         ax.plot(recalls, precisions[:, i], linewidth=1, label=f"{name}")  # plot(recall, precision for each class)
    # else:
    #     ax.plot(recalls, precisions.mean(1), linewidth=1, color="grey")  # plot(recall, precision mean for all classes)
    # Plot the mean precision
    ax.plot(recalls1, precisions1.mean(1), linewidth=3, color="blue", label=f"all classes {float(mAP1):.3f} mAP@0.5")
    ax.plot(recalls2, precisions2.mean(1), linewidth=3, color="red", label=f"all classes {float(mAP2):.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

    print(f"Replot saved to {save_dir}")

# Usage example:
replot_pr_curve_from_csv(csv_path1="pr_curve_y1.csv",csv_path2="pr_curve_y2.csv")
