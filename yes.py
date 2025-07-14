import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Create the main window
root = tk.Tk()
root.title("Double Exponential Decay with Cosine")

# Define the equation in LaTeX format
equation = (
    r"$f(t) = A_1 e^{-(t / T_1)^2} \cos(2 \pi f_1 t + \phi_1)"
    r" + A_2 e^{-(t / T_2)^2} \cos(2 \pi f_2 t + \phi_2) + y_0$"
)

# Create a matplotlib figure to render the equation
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')  # Hide the axes
ax.text(0.5, 0.5, equation, fontsize=16, ha='center', va='center')

# Embed the matplotlib figure into the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.draw()

# Start the GUI event loop
root.mainloop()
