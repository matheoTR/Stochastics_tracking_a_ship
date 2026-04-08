import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement des données
true_data = np.loadtxt("True_data.txt")
observations = np.loadtxt("Observation.txt")
inputs = np.loadtxt("Input.txt")

# Paramètres du projet
dt = 0.1
N = 100
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
for k in range(N):
    u_k = inputs[k]
    z_k = observations[k]

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

# 6. Calcul de la performance (MSE)
true_pos = true_data[:, :2]
mse = np.mean(np.sum((true_pos - estimations) ** 2, axis=1))
print(f"Mean Square Error (MSE): {mse:.4f}")

# 7. Visualisation
plt.figure(figsize=(10, 6))
plt.plot(true_pos[:, 0], true_pos[:, 1], "g-", label="Trajectoire Réelle")
plt.plot(
    observations[:, 0],
    observations[:, 1],
    "r.",
    alpha=0.3,
    label="Observations Bruyantes",
)
plt.plot(estimations[:, 0], estimations[:, 1], "b--", label="Estimation Kalman")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.title("Suivi de navire : Filtre de Kalman (Bruit de Laplace)")
plt.legend()
plt.grid(True)
plt.savefig("results/Kalman_Tracking_Result.png")
plt.show()

