import numpy as np
# Letra A
dados = np.loadtxt("C:/Faculdade/Trabalho IA/av1-inteligencia-artificial/questões/arsenio_dataset.csv", delimiter=",", skiprows=1)

idade = dados[:, 0]# Regressores
sexo = dados[:, 1] # Não utilizado na letra A
uso_beber = dados[:, 2]# Regressores
uso_cozinhar = dados[:, 3]# Regressores
arsenio_agua = dados[:, 4] # Regressores
y = dados[:, 5]  # Arsenio_Unhas como Resposta

# print(len(dados)) #quantidade
X = np.column_stack((np.ones(len(dados)), idade, uso_beber, uso_cozinhar, arsenio_agua))

beta = np.linalg.inv(X.T @ X) @ (X.T @y) 
# print(beta)

equacao = f"y = {beta[0]} + {beta[1]} * idade + {beta[2]} * uso_beber + {beta[3]} * uso_cozinhar + {beta[4]} * arsenio_agua"
print("Equação da regressão linear multipla = ",equacao)


# Letra B
x_novo = np.array([1, 30, 5, 5, 0.135])  
y_pred_novo = x_novo @ beta  
print("Previsão com os valores do enunciado = ",y_pred_novo)

# Letra C
y_pred = X @ beta
ybar = np.mean(y)
r_squared = 1 - (np.sum((y - y_pred)**2) / np.sum((y - ybar)**2))
print("R² = ",r_squared)

# Letra D
n = len(y)  
p = X.shape[1] - 1  
r_squared_ajustado = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
print("R² ajustado =", r_squared_ajustado)

# Letra E - Comparar com um modelo alternativo
X_comparar = np.column_stack((np.ones(len(dados)), arsenio_agua))

beta_comparar = np.linalg.inv(X_comparar.T @ X_comparar) @ (X_comparar.T @y) 
# print(beta)
y_pred_comparar = X_comparar @ beta_comparar

equacao_comparar = f"y = {beta_comparar[0]} + {beta_comparar[1]} * arsenio_agua"
r_squared_comparar = 1- np.sum((y - y_pred_comparar)**2)/ np.sum((y - ybar)**2)
print("Equação da regressão linear multipla alternativa = ",equacao_comparar)
print("R² comparar = ",r_squared_comparar)

# Letra F - Analise de resíduos 
# Letra F, A) 
     # y_pred = X @ beta

# Letra F, B)
     # residuos = y - y_pred

# Letra F, C)
     #Criar um tabela com:
        # - Observação
        # - y 
        # - y_pred 
        # - residuos

# Letra G: calcular o R² e o RMSE sem a coluna de 1
X_sem_intercepto = np.column_stack((idade, uso_beber, uso_cozinhar, arsenio_agua))

beta_sem_intercepto = np.linalg.inv(X_sem_intercepto.T @ X_sem_intercepto) @ (X_sem_intercepto.T @y) 

y_pred_sem_intercepto = X_sem_intercepto @ beta_sem_intercepto
r_squared_sem_intercepto = 1 - (np.sum((y - y_pred_sem_intercepto)**2) / np.sum((y - ybar)**2))
print("R² sem intercepto = ",r_squared_sem_intercepto)

#RMSE sem intercepto
rmse_sem_intercepto = np.sqrt(np.mean((y-y_pred_sem_intercepto)**2))
print("RMSE sem intercepto = " , rmse_sem_intercepto)

# Letra H: calcular mais outras duas metricas para ambos os modelos
# RMSE e MAE do modelo completo
rmse_completo = np.sqrt(np.mean((y-y_pred)**2))
mae_completo = np.mean(np.abs(y - y_pred))
print("RMSE modelo completo = " , rmse_completo)
print("MAE modelo completo = ", mae_completo)

# RMSE e MAE do modelo alternativo
rmse_alternativo = np.sqrt(np.mean((y-y_pred_comparar)**2))
mae_alternativo = np.mean(np.abs(y - y_pred_comparar))
print("RMSE modelo alternativo = " , rmse_alternativo)
print("MAE modelo alternativo = ", mae_alternativo)
