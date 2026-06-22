import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

#Data generation from mixture of 2 exponentials (Eq 1)
def draw_sample(beta, a, n):
    n_1 = np.sum(np.random.binomial(n,0.5))
    x_1 = np.random.exponential(beta,n_1)
    x_2 = np.random.exponential(beta / a, n-n_1)
    x = np.concat((x_1,x_2))
    np.random.shuffle(x)
    return x, x_1, x_2

# Scatter plot of data from mixture
def plot_sep(a, beta, x1, x2) :
    fig, ax = plt.subplots()
    sns.set_theme()
    sns.set_style("white")

    y1 = 0.5 * ( (a * np.exp(- a * x1 / beta) / beta) + np.exp(-x1 / beta) / beta)
    y2 = 0.5 * ( (a * np.exp(- a * x2 / beta) / beta) + np.exp(-x2 / beta) / beta)
    ax.scatter(x1,y1)
    ax.scatter(x2,y2)
    ax.set_ylim(top=5, bottom=-0.5)
    plt.suptitle(f'Exponential Mixture for α = {a}')
    plt.show()

# Plot of densities for different alpha values
def plot_alpha(a,b,x):
    fig, ax = plt.subplots()
    y_b = np.exp(-x / b) / b
    y_a = a * np.exp(- a * x / b) / b
    x_pt = b * math.log(a) / (a-1)
    y_pt = math.exp(- math.log(a) / (a-1)) / b
    ax.plot(x,y_b)
    ax.plot(x,y_a)
    ax.set_title(f'α={j}')
    ax.scatter(x_pt, y_pt, color = 'black', zorder=10)
    ax.set_ylim(top=5, bottom=-0.5)
    plt.show()



#Simulation parameters
n = 1000
k = 2.5
a = k * 10
beta = 2

# Figure 1
x = np.arange(0.05, 5, 0.05)
plot_alpha(a, beta, x)

# Figure 2
x, x1, x2 = draw_sample(beta, a, n)
plot_sep(a, beta, x1, x2)
