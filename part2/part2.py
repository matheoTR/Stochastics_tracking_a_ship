import numpy as np
import matplotlib.pyplot as plt
import os

x_true_all = np.loadtxt("True_data.txt")
x_obs = np.loadtxt("Observation.txt")
u_input = np.loadtxt("Input.txt")
s_true = x_true_all[:, :2]


def kalman_filter(T=100, dt=0.1):
    # Paramètres du projet
    mu_laplace = 1.0
    b_laplace = 1.0
    sigma_v2 = 1.0

    # 2. Définition des matrices du modèle (Espace d'état)
    # État X = [x, y, vx, vy]
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])

    # Matrice G pour le bruit d'accélération (c'est la même que b prcq le bruit est une accélération)
    G = B.copy()

    # Matrice d'observation H
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    # 3. Covariances du bruit
    # Q = 2 * b^2 * G * G^T
    Q = 2 * (b_laplace**2) * (G @ G.T)

    # R = sigma_v^2 * I (2x2)
    R = np.eye(2) * sigma_v2

    # 4. Initialisation du filtre
    X_hat = np.array([10.0, 10.0, 10.0, 10.0])  # Moyennes s0 et v0
    P = np.diag([50.0, 50.0, 10.0, 10.0])  # P0_pos=50, P0_vel=10

    # Stockage des résultats
    estimations = []

    # 5. Boucle du Filtre de Kalman
    for k in range(T):
        u_k = u_input[k]
        z_k = x_obs[k]

        # --- ÉTAPE DE PRÉDICTION ---
        # Modification pour le bruit de Laplace non centré (mu=1)
        shift_term = G @ np.array([mu_laplace, mu_laplace])
        X_pred = A @ X_hat + B @ u_k + shift_term
        P_pred = A @ P @ A.T + Q

        # --- ÉTAPE DE MISE À JOUR (Correction) ---
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        X_hat = X_pred + K @ (z_k - H @ X_pred)
        P = (np.eye(4) - K @ H) @ P_pred

        estimations.append(X_hat[:2])  # On garde (x, y)

    estimations = np.array(estimations)

    # --- 4. PERFORMANCE ---
    error_vectors = s_true - estimations
    squared_norms = np.sum(error_vectors**2, axis=1)
    mse_kalman = np.mean(squared_norms)
    error_over_time = np.sqrt(squared_norms)

    return mse_kalman, error_over_time, estimations


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
    # --- 1. SETUP ---
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])

    # distributions du bruit w (Laplace) ~L(1,1)
    mu_w = 1
    b_w = 1

    # distribution du bruit v (WGN) ~N(0,1)
    mu_v = 0
    Sigma_v = 1

    # returns the probability of v over the gaussian N(0,1)
    def observation_noise_pdf(v):
        return (1 / np.sqrt(2 * np.pi * Sigma_v)) * np.exp(
            -0.5 * (v - mu_v) ** 2 / Sigma_v
        )

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
    # N = particle number
    # (x, y, w_dot, y_dot) = the 4 values of pos and speed of particle n at time t
    # T = time frame
    S = np.zeros((N, 4, T))  # Resampled particles
    # particule_0 au t=0, t=1, t=2, ... t=T
    # particule_1 au t=0, t=1, t=2, ... t=T
    # ...
    # particule_N au t=0, t=1, t=2, ... t=T

    # Particles are drawn from the prior distribution p(x_0)
    S[:, 0, 0] = np.random.normal(mu_s, sqrt_Sigma_s, N)  # x
    S[:, 1, 0] = np.random.normal(mu_s, sqrt_Sigma_s, N)  # y

    S[:, 2, 0] = np.random.normal(mu_s_dot, sqrt_Sigma_s_dot, N)  # x_dot
    S[:, 3, 0] = np.random.normal(mu_s_dot, sqrt_Sigma_s_dot, N)  # y_dot

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
        weights += 1e-300  # Avoids division by zero if all weights are tiny (which happened before)
        weights /= np.sum(weights)

        # D. RESAMPLING
        # we clone the winners and delete the losers
        indices = np.random.choice(N, size=N, p=weights)

        # The new population for time n+1 is the resampled set
        # Because we resampled, weights are implicitly reset to 1/N
        S[:, :, n + 1] = S_tilde[indices]

    # --- 4. ESTIMATION ---
    # We estimate the state using the mean of the particle cloud
    # since we're after the resampling, weights are all equal so we can just take a
    # simple mean of the resampled set (for this unimodal model)
    state_estimate = np.mean(S, axis=0)

    # --- 5. PERFORMANCE ---
    # for performance, we only consider position, and not velocity
    s_hat = state_estimate[:2, :].T

    # error vector for each time step
    error_vectors = s_true - s_hat

    # ||s_k - s_hat_k||^2
    squared_norms = np.sum(error_vectors**2, axis=1)

    # MSE: take the mean
    MSE = np.mean(squared_norms)
    error_over_time = np.sqrt(squared_norms)

    return MSE, error_over_time, s_hat


def main():
    RESULTS = "results"
    os.makedirs(RESULTS, exist_ok=True)

    N_sizes = [10, 100, 1000, 5000]
    M, T, dt = 100, 100, 0.1
    global_mse_results = []

    # --- 1. KALMAN FILTER EXECUTION AND PLOTTING ---
    print("Running Kalman Filter...")
    mse_k, err_k, estimations_k = kalman_filter(T, dt)

    plt.figure(figsize=(10, 6))
    plt.plot(s_true[:, 0], s_true[:, 1], "g-", label="Real Trajectory")
    plt.plot(x_obs[:, 0], x_obs[:, 1], "r.", alpha=0.3, label="Noisy observations")
    plt.plot(estimations_k[:, 0], estimations_k[:, 1], "b--", label="Kalman estimation")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.title("Ship tracking: Kalman Filter")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS, "Kalman_Tracking_Result.png"))
    plt.show()

    # --- 2. SIR PARTICLE FILTERS EXECUTION ---
    fig_error, ax_error = plt.subplots(figsize=(10, 6))
    fig_traj, ax_traj = plt.subplots(figsize=(10, 8))

    ax_traj.plot(
        s_true[:, 0], s_true[:, 1], "g-", linewidth=3, label="Real trajectory", zorder=5
    )
    ax_traj.scatter(
        x_obs[:, 0],
        x_obs[:, 1],
        color="red",
        s=5,
        alpha=0.2,
        label="Noisy observations",
        zorder=1,
    )

    for N in N_sizes:
        print(f"Running simulation for N = {N} particles...")
        all_MSE = np.empty(M)
        all_errors_over_time = np.empty((M, T))
        all_trajectories = np.empty((M, T, 2))

        for i in range(M):
            all_MSE[i], all_errors_over_time[i], all_trajectories[i] = (
                SIR_particle_filter(N, T, dt)
            )

        avg_mse = np.mean(all_MSE)
        global_mse_results.append(avg_mse)
        avg_error_over_time = np.mean(all_errors_over_time, axis=0)
        avg_trajectory = np.mean(all_trajectories, axis=0)

        ax_error.plot(avg_error_over_time, label=f"N = {N}")
        ax_traj.plot(
            avg_trajectory[:, 0],
            avg_trajectory[:, 1],
            "--",
            linewidth=1.5,
            label=f"SIR estimation (N={N})",
        )

    # --- 3. SIR PLOTS ---
    ax_error.set_title(f"Comparison of Mean Position Error (M={M} trials)")
    ax_error.set_xlabel("Time Step")
    ax_error.set_ylabel("Euclidean Distance Error")
    ax_error.legend()
    ax_error.grid(True, alpha=0.3)
    fig_error.savefig(os.path.join(RESULTS, "combined_SIR_error_plot.png"))
    plt.close(fig_error)

    ax_traj.set_title(f"Ship tracking: Particle Filter over {M} experiments")
    ax_traj.set_xlabel("Position X (m)")
    ax_traj.set_ylabel("Position Y (m)")
    ax_traj.legend(loc="best")
    ax_traj.grid(True, linestyle=":", alpha=0.6)
    fig_traj.savefig(os.path.join(RESULTS, "combined_SIR_trajectories.png"))
    plt.show()
    plt.close(fig_traj)

    # --- 4. SIR CONVERGENCE ANALYSIS ---
    plt.figure(figsize=(8, 6))
    plt.loglog(N_sizes, global_mse_results, "o-r", linewidth=2)
    plt.title("SIR Filter Convergence: Global MSE vs. Number of Particles")
    plt.xlabel("Number of Particles (Np)")
    plt.ylabel("Mean Square Error (MSE)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(RESULTS, "SIR_convergence_analysis.png"))

    print("Done.")


if __name__ == "__main__":
    main()
