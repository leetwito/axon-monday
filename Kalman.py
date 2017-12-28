from numpy import *
from numpy.linalg import inv
import matplotlib.pyplot as plt

def kalman_predict(X, P, A, Q, B, U):
    """
    Prediction step in Kalman filter algorithm
    :param X: The mean state estimate of the previous step
    :param P: The state covariance of previous step
    :param A: The transition nxn matrix
    :param Q: The process noise covariance matrix
    :param B: The input effect matrix
    :param U: The control input
    :return:(X,P)
    """
    X = dot(A, X) + dot(B, U)
    P = dot(A, dot(P, A.T)) + Q
    return (X, P)

def kalman_update(X, P, Y, H, R):
    """
    Update step in Kalman filter algorithm
    :param X: The mean state estimate of the previous step
    :param P: The state covariance of previous step
    :param Y: The measurement value
    :param H: The measurement matrix
    :param R: The measurement covariance matrix
    :return: (X,P,K = Kalma gain)
    """
    IM = dot(H, X)
    IS = R + dot(H, dot(P, H.T))
    K = dot(P, dot(H.T, inv(IS)))
    X = X + dot(K, (Y-IM))
    P = P - dot(K, dot(IS, K.T))
    return (X, P, K)


def main():

    # Time step between each state
    dt = 1

    # Number of iterations
    N_iter = 200

    # Create true position and measurement
    measurement_error_true = 5000
    x = array(range(N_iter))
    pos_true = 1.0 + square(x) + 10000*sin(0.05*x)
    Y = pos_true + measurement_error_true * random.randn(1, N_iter)

    # Initialization of matrices
    X = array([[40000], [3]]) # X=state=(position,speed)
    P = diag((1, 3))
    A = array([[1, dt], [0, 1]]) # assumes constant speed movement
    Q = diag((1, 3))

    H = array([[1, 0]])
    R = array([[10000]])

    B = eye(X.shape[0])
    U = zeros((X.shape[0],1))

    X_kal = []
    t = dt * range(N_iter)

    # Applying the Kalman Filter
    for i in range(N_iter):
        (X, P) = kalman_predict(X, P, A, Q, B, U)
        (X, P, K) = kalman_update(X, P, array(Y[0, i]), H, R)
        X_kal.append(X[0,0])

    # Plot results
    plt.title("Kalman filter over position of a particle")
    plt.plot(t, pos_true, 'r', label="True position")
    plt.plot(t, Y[0,0:], 'go', label="Measured position")
    plt.plot(t, X_kal, 'b', label="Kalman filter position")
    plt.legend()
    plt.show()






if __name__ == '__main__':
    main()
