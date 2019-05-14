import numpy as np
import matplotlib.pyplot as plt

'''
隠れマルコフモデル
イカサマサイコロ連投問題

観測変数    Y=1, ..., 6
隠れ変数    X=1 イカサマサイコロを使用
            X=0　通常サイコロを使用
        
隠れ変数の遷移モデル
    p(X_(t+1) = 0 |X_t = 0) = 0.9
    p(X_(t+1) = 0 |X_t = 1) = 0.1
    p(X_(t+1) = 1 |X_t = 0) = 0.9
    p(X_(t+1) = 1 |X_t = 1) = 0.1
    
観測過程モデル
    P(Y_t = i | X_t = 0) = 1/6 (i=1, ..., 6)
    P(Y_t = i | X_t = 1) = 1/10 (i=1, ..., 5)
    P(Y_t = 6 | X_t = 1) = 1/2
    
'''

# 隠れ変数の遷移モデル hidden_v[X_(t+1)][X_t]
hidden_v = np.array([[0.9, 0.1], [0.1, 0.9]])

# 観測過程モデル observed[Y_t - 1][X_t]
observed = np.array([[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6], [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2]])
observed = observed.T

data_s = str(4454233645534414536533355356551343126161542415411115666616262616666632666166266566566664366525456544)
data = [int(i) for i in data_s]
trial = len(data)  # 100

ALPHA = [0] * trial
BETA = [0] * trial
GAMMA = [0] * trial

alpha = np.array([1 / 2, 1 / 2])
for t in range(trial):
    Y = data[t] - 1
    alpha = observed[Y] * np.dot(alpha, hidden_v)
    alpha /= np.sum(alpha)
    ALPHA[t] = alpha[1]


beta = np.array([1 / 2, 1 / 2])
for t in range(trial - 1, -1, -1):
    Y = data[t] - 1
    beta = np.dot(beta * observed[Y], hidden_v)
    beta /= np.sum(beta)
    BETA[t] = beta[1]
    GAMMA[t] = ALPHA[t] * beta[1] / (ALPHA[t] * beta[1] + (1 - ALPHA[t]) * beta[0])


x = range(1, trial + 1)

plt.plot(x, ALPHA, label='alpha')
plt.plot(x, BETA, label='beta')
plt.plot(x, GAMMA, label='gamma')

six = []
for i in x:
    if data[i - 1] == 6:
        six.append(i)

y = [-0.01] * len(six)
plt.scatter(six, y, label='dice : 6', color='black', marker='.')

plt.legend()

plt.savefig('fig2.pdf')
plt.show()
