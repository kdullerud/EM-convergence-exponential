import numpy as np
from scipy.optimize import minimize
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_sample(beta, a, n, pi):
    n_1 = np.sum(np.random.binomial(n,pi))
    x_1 = np.random.exponential(beta,n_1)
    x_2 = np.random.exponential(beta / a, n-n_1)
    x = np.concat((x_1,x_2))
    np.random.shuffle(x)
    return x

def em_unbalanced(x, a, pi, init, eps, its):
    diff = 100
    current = init
    n = len(x)
    j=0
    iterates = []
    for k in range(its):
        sum = 0
        for i in range(n):
            sum += a * x[i] - x[i] * (a-1) * (pi / ( pi + (1-pi) * a * math.exp((1-a) *  x[i] / current )))
        new = (1/n)* sum
        diff = abs(new-current)
        current = new
        iterates.append(current)
    j = its
    return current,j,iterates

def run_sims_unbalanced(beta, a, n, eps, runs, it, ratio, sample):
    em_df = pd.DataFrame(columns=['beta', 'a','n','em_est','iter','error', 'ratio'])
    index=0
    for i in beta:
        for j in a:
            for k in n:
                for N in range(runs):
                    r = j / i
                    np.random.seed(100+N)
                    x = draw_sample(i, j, k, ratio)
                    #init = (1 + 1/a) / (2*np.mean(x)) + np.random.normal(0.1,0.1)
                    init = np.random.uniform(0,sample)
                    em_est, iters,iterates = em_unbalanced(x,j,ratio,init,eps, it)
                    for iter in range(iters):
                        error = abs(iterates[iter]-i)
                        new_row = [i,j,k,em_est,iter+1,error, r]
                        em_df.loc[index] = new_row
                        index+=1
    em_df['final_error'] = abs(em_df['beta']-em_df['em_est'])
    em_df = em_df.groupby(['beta', 'a', 'n','iter']).mean().reset_index()
    return em_df

def run_plot_unbalanced(em_df):
    df_plot2 = em_df
    sns.set_theme(font='Arial')
    sns.set_style("white")

    fig, ax = plt.subplots()
    sns.lineplot(data=df_plot2, x='n',y='final_error',hue=df_plot2[['a', 'beta']].apply(tuple, axis=1), 
                marker='o', errorbar=None, ax=ax)
    plt.legend(loc=(0.85, 0.7), title = ('(α,β)'))
    ax.set_yscale("log")
    ax.set_xscale("log")
    # ax.set_xticks(em_df['n'].unique().tolist(), labels=em_df['n'].unique().tolist())
    ax.set_ylabel('Abs. Error')
    
    #plt.title('Final Estimate Error vs. Sample Size')
    plt.show()


# simulations
beta = [0.5,5]
a = [2,4]
n = [100, 1000, 10000]
eps=1e-2
runs = 50
it = 32
ratio = 0.7
sample = 10

# run unbalanced simulations
em_df_unbalanced = run_sims_unbalanced(beta, a, n, eps, runs, it, ratio, sample)

# figure 5
run_plot_unbalanced(em_df_unbalanced)
