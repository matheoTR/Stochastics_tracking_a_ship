import numpy as np
import matplotlib.pyplot as plt
import os


def SIR_particle_filter(N=1000, T=100, dt=0.1):
    """
    N = Number of particles
    T = Number of time steps
    dt = delta time
    """
    # --- 0. PHYSICAL MODEL ---
    # x[n] = A * x[n-1] + B * u[n] + B * w[n]
    # y[n] = s[n] + v[n]
    #
    # x[n] = (s[n], s_dot[n]) = state vector (position (x, y), acceleration (x_dot, y_dot)) at time n
    # u[n] = input of the system, the controllable part of the acceleration
    # w[n] = noise at time n, following a Laplace pdf, the uncontrollable part of the acceleration
    # A = [[1, dt], [0, 1]]
    # B = [1/2 * dt², dt]
    # y[n] = observation vector at time n
    # v[n] = white gaussian noise at time n
    # G(s[n]) = s[n]

    dt = 0.1

    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])

    # distributions du bruit w (Laplace) ~L(1,1)
    mu_w = 1
    b_w = 1

    # distribution du bruit v (WGN) ~N(0,1)
    mu_v = 0
    Sigma_v = 1
    sqrt_Sigma_v = np.sqrt(Sigma_v)

    # returns the probability of v over the gaussian N(0,1)
    def observation_noise_pdf(v):
        return (1 / np.sqrt(2 * np.pi * Sigma_v)) * np.exp(
            -0.5 * (v - mu_v) ** 2 / Sigma_v
        )

    # --- 0.5: REAL DATA ---
    # True_data: (x, y, vx, vy)
    x_true_all = np.loadtxt("True_data.txt")

    # Observation: (obs_x, obs_y)
    x_obs = np.loadtxt("Observation.txt")

    # Input: (u_x, u_y)
    u_input = np.loadtxt("Input.txt")

    # --- 1. SETUP & HYPERPARAMETERS ---
    N = 1000  # Number of particles
    T = 100  # Time steps / number of iterations

    # N = particle number
    # (x, y, w_dot, y_dot) = the 4 values of pos and speed of particle n at time t
    # T = time frame
    S = np.zeros((N, 4, T))  # Resampled particles
    # particule_0 au t=0, t=1, t=2, ... t=T
    # particule_1 au t=0, t=1, t=2, ... t=T
    # ...
    # particule_N au t=0, t=1, t=2, ... t=T

    # prior distribution p(x_0):
    # p(s_0) = N(mu_s, Sigma_s)
    mu_s = 10
    Sigma_s = 50
    sqrt_Sigma_s = np.sqrt(Sigma_s)

    # p(s_dot_0) = N(mu_s_dot, Sigma_s_dot)
    mu_s_dot = 10
    Sigma_s_dot = 10
    sqrt_Sigma_s_dot = np.sqrt(Sigma_s_dot)

    # --- 2. INITIALIZATION ---
    # Particles are drawn from the prior distribution p(x_0)
    S[:, 0, 0] = np.random.normal(mu_s, sqrt_Sigma_s, N)  # x
    S[:, 1, 0] = np.random.normal(mu_s, sqrt_Sigma_s, N)  # y

    S[:, 2, 0] = np.random.normal(mu_s_dot, sqrt_Sigma_s_dot, N)  # x_dot
    S[:, 3, 0] = np.random.normal(mu_s_dot, sqrt_Sigma_s_dot, N)  # y_dot

    # in this project, X0 is determined by averaging the random particles drawn at t=0
    X0 = np.mean(S[:, :, 0], axis=0)

    # --- 3. MAIN RECURSION: for each point in time (all particles at once) ---
    for n in range(T - 1):
        # A. PREDICTION
        u = u_input[n]
        # Generate random noise independently for both directions
        w = np.random.laplace(mu_w, b_w, (N, 2))
        # Move particles according to physical model
        # X_{k+1} = A*Xk + B*uk + B*wk
        # .T pour transpose
        S_tilde = (A @ S[:, :, n].T).T + (B @ u).T + (B @ w.T).T

        # B. WEIGHTING
        # How well does each particle match the real-world observation x_obs?
        # We evaluate the likelihood p(x_obs | S_tilde)
        # la prédiction dévie de l'observation. La différence entre les deux est l'erreur
        # comme x = G(s) + w (noise), la différence correspond exactement au bruit
        # pour attribuer un poids à cette particule, on va simplement l'évaluer comme le bruit
        # et voir selon la distribution du bruit à quel point ce bruit était probable
        Y_obs = x_obs[n + 1]
        # Residuals for x and y components
        residual_x = Y_obs[0] - S_tilde[:, 0]
        residual_y = Y_obs[1] - S_tilde[:, 1]

        # Weight is the product of 1D Gaussian probabilities
        # f is standard normal with mu=0, sigma=1
        weights = observation_noise_pdf(residual_x) * observation_noise_pdf(residual_y)

        # C. NORMALIZATION
        # Turn raw likelihoods into a valid probability distribution (sum = 1)
        weights += 1e-300  # Avoid division by zero if all weights are tiny
        weights /= np.sum(weights)

        # D. RESAMPLING
        # we clone the winners and delete the losers
        indices = np.random.choice(N, size=N, p=weights)

        # The new population for time n+1 is the resampled set
        # Because we resampled, weights are implicitly reset to 1/N
        S[:, :, n + 1] = S_tilde[indices]

    # --- 4. ESTIMATION ---
    # The best estimate of the state is typically the mean of the particle cloud
    state_estimate = np.mean(S, axis=0)

    # --- 5. PERFORMANCE ---
    # we only consider position, and not velocity
    s_hat = state_estimate[:2, :].T
    s_true = x_true_all[:, :2]

    # error vector for each time step
    error_vectors = s_true - s_hat

    # ||s_k - s_hat_k||^2
    squared_norms = np.sum(error_vectors**2, axis=1)

    # MSE: take the mean
    MSE = np.mean(squared_norms)
    error_over_time = np.sqrt(squared_norms)

    return MSE, error_over_time


def main():
    # Saving figures
    RESULTS = "results"
    os.makedirs(RESULTS, exist_ok=True)

    N_sizes = [10, 100, 1000, 5000]  # Number of particles
    M = 100  # Number of experiments per N_size
    T = 100  # Number of time steps
    global_mse_results = []

    for N in N_sizes:
        print(f"Running simulation for N = {N} particles...")

        # results of each of the M experiments
        all_MSE = np.empty(M)
        all_errors_over_time = np.empty((M, T))

        for i in range(M):
            all_MSE[i], all_errors_over_time[i] = SIR_particle_filter(N, T)

        # --- AVERAGING ---
        avg_mse = np.mean(all_MSE)
        global_mse_results.append(avg_mse)

        # Average the error curve
        avg_error_over_time = np.mean(all_errors_over_time, axis=0)

        # --- 6. VISUALIZATION (Error over time) ---
        plt.figure(figsize=(10, 5))
        plt.plot(avg_error_over_time, label=f"N={N}")
        plt.title(f"Mean Position Error (M={M} trials) for N={N}")
        plt.xlabel("Time Step")
        plt.ylabel("Euclidean Distance Error")
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(RESULTS, f"error_plot_{N}.png")
        plt.savefig(save_path)
        plt.close()

    # --- 7. FINAL COMPARISON: MSE vs N ---
    plt.figure(figsize=(8, 6))
    plt.loglog(N_sizes, global_mse_results, "o-r", linewidth=2)
    plt.title("Filter Convergence: Global MSE vs. Number of Particles")
    plt.xlabel("Number of Particles (Np)")
    plt.ylabel("Mean Square Error (MSE)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    save_path = os.path.join(RESULTS, "convergence_analysis.png")
    plt.savefig(save_path)
    plt.show()

    print("Simulations complete. Results saved as PNG files.")


if __name__ == "__main__":
    main()
