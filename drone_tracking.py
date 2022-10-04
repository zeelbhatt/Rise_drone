import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
import matplotlib.pyplot as plt


class UAV(object):

    def __init__(self, J, e3):
        self.m = 4.34
        self.g = 9.8
        self.J = J
        self.e3 = e3
        self.kR = 8.81  # attitude gains
        self.kW = 2.54  # attitude gains
        self.kx = 16. * self.m  # position gains
        self.kv = 5.6 * self.m  # position gains
        self.xd = None
        self.xd_dot = None
        self.command = None
        print('UAV: initialized')
        self.f = np.array([0, 0, 0])
        self.M = np.array([0, 0, 0])


    def dydt(self, t, X):
        R = np.reshape(X[6:15], (3, 3))
        W = X[15:]
        x = X[:3]
        v = X[3:6]


        xd_dot = np.array([0, 0, 0])
        xd_ddot = np.array([0, 0, 0])
        xd_dddot = np.array([0, 0, 0])
        xd_ddddot = np.array([0, 0, 0])
        b1d_ddot = np.array([0., 0., 0.])
        Wd_dot = np.array([0., 0., 0.])

        xd = np.array([1., 2., 3.])
        ang_d = 2. * np.pi * (t - 4)
        ang_d_dot = 2. * np.pi
        Rd = np.array([[np.cos(ang_d), 0., np.sin(ang_d)], [0., 1., 0.],
                       [-np.sin(ang_d), 0., np.cos(ang_d)]])
        Rd_dot = np.array([[-ang_d_dot * np.sin(ang_d), 0.,
                            ang_d_dot * np.cos(ang_d)], [0., 0., 0.],
                           [-ang_d_dot * np.cos(ang_d), 0., -ang_d_dot * np.sin(ang_d)]])
        Wdhat = Rd.T.dot(Rd_dot)
        Wd = np.array([-Wdhat[1, 2], Wdhat[0, 2], -Wdhat[0, 1]])
        b1d = Rd[:, 0]
        b1d_dot = Rd_dot[:, 0]
        d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
                b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (self.f, self.M) = self.attitude_control(t, R, W, x, v, d_in)

        R_dot = np.dot(R, hat(W))
        W_dot = np.dot(la.inv(self.J), self.M - np.cross(W, np.dot(self.J, W)))
        x_dot = v
        v_dot = self.g * self.e3 - self.f * R.dot(self.e3) / self.m
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot)) # state space dot
        self.xd = xd
        self.xd_dot = xd_dot
        self.command = np.insert(self.M, 0, self.f)
        return X_dot

    def attitude_control(self, t, R, W, x, v, d_in):
        (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot, b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot) = d_in
        (ex, ev) = position_errors(x, xd, v, xd_dot)
        self.f = (self.kx * ex + self.kv * v + self.m * self.g * self.e3).dot(R.dot(self.e3))
        W_hat = hat(W)
        (eR, eW) = attitude_errors(R, Rd, W, Wd)
        self.M = -self.kR * eR - self.kW * eW + np.cross(W, self.J.dot(W)) - self.J.dot(
            W_hat.dot(R.T.dot(Rd.dot(Wd))) - R.T.dot(Rd.dot(Wd_dot)))
        return self.f, self.M


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def attitude_errors(R, Rd, W, Wd):
    eR = 0.5 * vee(Rd.T.dot(R) - R.T.dot(Rd))
    eW = W - R.T.dot(Rd.dot(Wd))
    return eR, eW


def position_errors(x, xd, v, vd):
    ex = x - xd
    ev = v - vd
    return ex, ev


def hat(x):
    hat_x = [0, -x[2], x[1],
             x[2], 0, -x[0],
             -x[1], x[0], 0]
    return np.reshape(hat_x, (3, 3))


if __name__ == "__main__":


    J = np.diag([0.0820, 0.0845, 0.1377])
    e3 = np.array([0., 0., 1.])
    uav_t = UAV(J, e3)

    t_max = 20
    N = 100 * t_max + 1
    t = np.linspace(0, t_max, N)
    xd = np.array([0., 0., 0.])
    # Initial Conditions

    R0 = np.eye(3)
    W0 = [0., 0., 0.]
    x0 = [0., 0., 0.]
    v0 = [0., 0., 0.]

    R0v = np.array(R0).flatten().T
    y0 = np.concatenate((x0, v0, R0v, W0))

    solver = ode(uav_t.dydt)
    solver.set_integrator('dopri5').set_initial_value(y0, 0)
    dt = 1. / 100
    sim = []
    xd = []
    xd_dot = []
    command_val = []

    while solver.successful() and solver.t < t_max:
        solver.integrate(solver.t + dt)
        sim.append(solver.y)
        xd.append(uav_t.xd)
        xd_dot.append(uav_t.xd_dot)
        command_val.append(uav_t.command)

    sim = np.array(sim)
    xd = np.array(xd)
    commands = np.array(command_val)
    print(xd.shape)
    xd_dot = np.array(xd_dot)

    f, ax = plt.subplots(3, 3)
    ax[0][0].plot(xd[:, 0])
    ax[0][0].plot(sim[:, 0])
    ax[0, 0].set_title("X Pose")

    ax[0][1].plot(xd_dot[:, 0])
    ax[0][1].plot(sim[:, 3])
    ax[0, 1].set_title("X Velocity")

    ax[1][0].plot(xd[:, 1])
    ax[1][0].plot(sim[:, 1])
    ax[1, 0].set_title("Y Pose")

    ax[1][1].plot(xd_dot[:, 1])
    ax[1][1].plot(sim[:, 4])
    ax[1, 1].set_title("Y Velocity")

    ax[2][0].plot(xd[:, 2])
    ax[2][0].plot(sim[:, 2])
    ax[2, 0].set_title("Z Pose")

    ax[2][1].plot(sim[:, 5])
    ax[2][1].plot(xd_dot[:, 2])
    ax[2, 1].set_title("Z Velocity")

    ax[0][2].plot(commands[:, 0])
    ax[0, 2].set_title("Moment - M")

    ax[1][2].plot(commands[:, 2])
    ax[1, 2].set_title("Thrust - f")

    plt.show()
