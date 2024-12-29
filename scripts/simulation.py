import numpy as np
import pandas as pd

import pickle

import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import scipy
import time 

## src

def gillespie_algorithm_step(t, x, s, nu=1.0, mu=0.1, lambda_=0.01, alpha=0.1):

  # Calculate reaction rates
  r_production = nu + mu * s  # Production rate
  r_degradation = lambda_ * x  # Degradation rate
  r_total = r_production + r_degradation

  # Generate random numbers
  r1 = np.random.random()
  r2 = np.random.random()

  # Calculate time to next reaction
  tau = - np.log(r1) / r_total #exponential distribution of the waiting time

  # Update time and cell size
  t += tau
  s *= np.exp(alpha * tau)

  # Determine which reaction occurs
  if r_production > r2 * r_total:
      x += 1  # Production
  else:
      x -= 1  # Degradation

  # Return values
  return t, x, s

def simulation(
    # general simulation parameters
    n_lineages = 50,
    n_divisions = 550,
    burn_in_divisions = 50,
    # dynamics
    x_start = 25, #not so relevant as we go for ergodicity
    s_start = 1,  #not so relevant as we go for ergodicity
    molecular_dynamics = "deterministic",
    dx = 1,
    nu = 0,
    mu = 0.5,
    lambda_ = 0.0,
    alpha = 0.1,
    #division
    x_abs_noise = 5,
    x_thresh = 50,
    reset_x = 0,
    reset_s = 0.5
):


    # Initialise data structures
    lineages_time_series = []
    data_divisions = np.zeros(((n_divisions - burn_in_divisions) * n_lineages, 6))

    for i in range(n_lineages):

        t = 0
        x = x_start
        s = s_start

        t_values = [t]
        x_values = [x]
        s_values = [s]

        _first_division = True

        #set the first threshold
        x_thresh_samp = np.random.normal(loc = x_thresh, scale = x_abs_noise)

        # set the division counter
        j = 0

        while j < n_divisions :

            #toggle to choose between stochastic and deterministic x evolution
            t, x, s = gillespie_algorithm_step(t, x, s, nu=nu, mu=mu, lambda_=lambda_, alpha=alpha)


            t_values.append(t)
            x_values.append(x)
            s_values.append(s)

            if x >= x_thresh_samp:
            #perform division
                # update size and molecules
                x = x * reset_x
                s = s * reset_s

                if j == burn_in_divisions - 1:
                    _first_t = t
                    _first_x = x
                    _first_s = s

                if j >= burn_in_divisions:
                    #register data
                    division_index = i * (n_divisions - burn_in_divisions) + j - burn_in_divisions
                    #complete the current division
                    data_divisions[division_index , 0] = t

                    if j == burn_in_divisions: #check if its the first division
                        data_divisions[division_index, 1] = t - _first_t
                        data_divisions[division_index, 2] = _first_x
                        data_divisions[division_index, 4] = _first_s
                    else:
                        data_divisions[division_index, 1] = t - data_divisions[division_index - 1, 0]

                    data_divisions[division_index, 3] = x_thresh_samp

                    data_divisions[division_index, 5] = s / reset_s

                    #initialize the next division
                    if division_index + 1 < data_divisions.shape[0]:
                        data_divisions[division_index + 1, :] = np.array([0, 0, x, 0, s, 0])

                #update the division index
                j += 1
                #set next threshold
                x_thresh_samp = np.random.normal(loc = x_thresh, scale = x_abs_noise)


        #clean the initialised but not completed division
        if division_index + 1 < data_divisions.shape[0]:
            data_divisions[division_index + 1,:] = np.array([0, 0, 0, 0, 0, 0])

        #save the time series
        lineages_time_series.append({
            't': np.array(t_values),
            'x': np.array(x_values),
            's': np.array(s_values)
        })


    data_divisions = pd.DataFrame(
        data_divisions,
        columns=["t","tau", "x_o", "x_*", "s_o", "s_*"]
    )

    return lineages_time_series, data_divisions


##script 

start_time = time.time()

np.random.seed(42)

reset_x_space = [0.5]
ratio_space = [1]
x_thresh_space  =  list(range(10, 10010, 100)) #list(range(10, 10010, 100))
sigma_thresh_ratio_space =  list(np.arange(0, 20, 0.2))

total_iterations = len(reset_x_space) * len(ratio_space) * len(x_thresh_space) * len(sigma_thresh_ratio_space)

sim_data_dictionary = {}

#simulation
for i, (reset_x, ratio, x_thresh, sigma_thresh_ratio) in enumerate(itertools.product(reset_x_space, ratio_space, x_thresh_space, sigma_thresh_ratio_space)):

    alpha = 1
    nu, mu, lambda_ = 0, alpha * ratio, 0

    lineages_time_series, data_divisions = simulation(
        # general simulation parameters
        n_lineages = 50,
        n_divisions = 750,
        burn_in_divisions = 250,
        # dynamics
        x_start = x_thresh / 2, #not so relevant as we go for ergodicity
        s_start = 1,  #not so relevant as we go for ergodicity
        molecular_dynamics = "gillespie",
        nu = nu,
        mu = mu,
        lambda_ = lambda_,
        alpha = alpha,
        #division
        x_abs_noise = sigma_thresh_ratio * x_thresh,
        x_thresh = x_thresh,
        reset_x = reset_x,
        reset_s = 0.5
    )

    sim_data_dictionary[(reset_x, ratio, x_thresh, sigma_thresh_ratio)] = data_divisions

    with open("simulation_data.pkl", "wb") as file:
        pickle.dump(sim_data_dictionary, file)


end_time = time.time()
elapsed_time = (end_time - start_time) / 3600
print(f"Simulation took {elapsed_time:.4f} hours.")
