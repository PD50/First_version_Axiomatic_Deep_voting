# view_exp3.py
import matplotlib.pyplot as plt
from plot_and_visual import exp3_table, plot_exp3_loss

# Replace with your actual folder path
folder = "results/exp3/WEC/exp3_2026-02-16_15-18-03_IC"

# Option A: Table of axiom satisfaction (printed to terminal)
exp3_table(file_WEC=folder)

# Option B: Loss curve plot
plot_exp3_loss(folder)
plt.show()