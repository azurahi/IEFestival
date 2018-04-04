import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
df = pd.read_csv('./merged.csv', encoding = 'cp1252')
li =list(df.columns)

print(df)
code = pd.read_csv('./codelist.csv')
code_list = []
for c in code['종목코드']:
    ttt =  'krx' + '0' *( 6-len(str(c))) + str(c)
    code_list.append(ttt)
dd=df.loc[:,code_list]
dd.dropna(axis=1, inplace=True)
dd.to_csv('merged0.csv')


# for c in code['']
# stx
# df[li]
# for c in code['종목코드']

def read_csv():
    df = pd.read_csv('./merged.csv', encoding='cp1252')
    li = list(df.columns)


def read_data(price_data, prob_folder):
    df = pd.read_csv('./merged.csv', encoding = 'cp1252')
    li =list(df.columns)
    li[0] = 'ddate'
    df.columns = li
    df1 = df.iloc[::-1]
    df1.index = df1['ddate']
    del df1['ddate']
    df.index = df['ddate']
    del df['ddate']
    return df1, df

def calculate_numeric():


def random_portfolio():


def markowitz_portfolio():




## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000

return_vec = np.random.randn(n_assets, n_obs)


'''
포트폴리오 구성하기
'''


def stastics(data, date):
    data['diff'] = data.diff(periods = date)/



def random_portfolio(returns):
    '''
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))

    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)

    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


weights, returns, risks = optimal_portfolio(return_vec)

plt.plot(stds, means, 'o')
plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o')












import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

def optimal_portfolio(returns, cov, date ='2015-10-05'):
    n = len(returns.ix[0])
    print()
    # returns = np.asmatrix(returns)

    N = 100

    # Convert to cvxopt matrices
    ret = returns.loc[date].values.astype(np.double)
    mus = ret.mean()

    print(ret.mean())
    S = opt.matrix(cov.loc[date].values.astype(np.double))

    pbar = opt.matrix(ret)


    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    # # Calculate efficient frontier weights using quadratic programming
    portfolios = solvers.qp(mus * S, -pbar, G, h, A, b)['x']
    # ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks

returns = pd.read_csv('./mean_3.csv',index_col=0, encoding = 'cp1252')
cov = pd.read_csv('./cov_3.csv', index_col=[0,1],encoding = 'cp1252')
# print(returns.loc['2015-09-25'])
print(returns.loc['2015-09-25'].values.astype(np.double).shape)
print(cov.loc['2015-09-25'].values.astype(np.double).shape)
wt, returns, risk = optimal_portfolio(returns,cov)
print(wt.argmax(), wt[155])