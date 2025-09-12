import numpy as np

# dados do dataset arsenio
idade = np.array([44, 45, 44, 66, 37, 45, 47, 38, 41, 49, 72, 45, 53, 86, 8, 32, 44, 63, 42, 62, 36])
uso_beber = np.array([5, 4, 5, 3, 2, 5, 5, 4, 3, 4, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5])
uso_cozinhar = np.array([5, 5, 5, 5, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
arsenico_agua = np.array([0.00087, 0.00021, 0.0, 0.00115, 0.0, 0.0, 0.00013, 0.00069, 0.00039, 0.0, 0.0, 0.046, 0.0194, 0.137, 0.0214, 0.0175, 0.0764, 0.0, 0.0165, 0.00012, 0.0041])
arsenico_unhas = np.array([0.119, 0.118, 0.099, 0.118, 0.277, 0.358, 0.08, 0.158, 0.31, 0.105, 0.073, 0.832, 0.517, 2.252, 0.851, 0.269, 0.433, 0.141, 0.275, 0.135, 0.175])

# matriz X: coluna de 1s + regressores (idade, uso_beber, uso_cozinhar, arsenico_agua)
n = len(idade)
X = np.column_stack((np.ones(n), idade, uso_beber, uso_cozinhar, arsenico_agua))

# calculo dos coeficientes beta via minimos quadrados (inv(X^T X) X^T y)
beta = np.linalg.inv(X.T @ X) @ (X.T @ arsenico_unhas)

# saida dos coeficientes
print("coeficientes beta (intercepto, idade, uso_beber, uso_cozinhar, arsenico_agua):")
print(beta)

# equação do modelo
equacao = f"arsenico_unhas = {beta[0]:.6f} + {beta[1]:.6f}*idade + {beta[2]:.6f}*uso_beber + {beta[3]:.6f}*uso_cozinhar + {beta[4]:.6f}*arsenico_agua"
print("\n" + equacao)