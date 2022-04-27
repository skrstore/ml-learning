import numpy as np

slope = 0
intercept = 0


def fit(x, y):
    # number of observations
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    print(SS_xy)
    SS_xx = np.sum(x * x) - n * m_x * m_x
    print(SS_xx)

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    globals()["slope"] = b_1
    globals()["intercept"] = b_0

    return (slope, intercept)


def predict(x):
    y_pred = globals()["slope"] * x + globals()["intercept"]
    return y_pred


def plot_linear_reg(x, y, x_new, y_pred):
    import matplotlib.pyplot as plt

    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m", marker="o", s=30)

    # plotting the regression line
    plt.plot(x_new, y_pred, color="g")

    # putting labels
    plt.xlabel("x")
    plt.ylabel("y")

    # function to show plot
    plt.show()

