import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# The traffic flow class allows us to model traffic flow. It uses the method
# of lines along with a High-resolution shock-capturing schemes (HRSC).
class TrafficFlow:
    # Initializes the
    def __init__(self, rho_bar, delta_rho, N = 10, x_center = 0.5, lam = 0.1, rho_max = 1, u_max = 1):
        self.u_max = u_max
        self.rho_max = rho_max
        self.delta_x = 1./N
        self.N = N
        self.rho_init_vec = np.zeros(N)
        self.x_vec = np.zeros(N)
        for i in range(N):
            self.x_vec[i] = (i+0.5)/N
            self.rho_init_vec[i] = rho_bar + delta_rho*math.exp(-((self.x_vec[i] - x_center)/lam)**2)

    def compute_rho(self, t, M):
        self.M = M
        self.t_vec = np.linspace(0, t, M)
        self.rho_mat = odeint(self.derivs, self.rho_init_vec, self.t_vec)

    def derivs_HRSC(self, rho_vec, t):
        # one array to store parallel values for
        rho_vec_half_L = np.zeros(self.N)
        rho_vec_half_R = np.zeros(self.N)
        for i in range(self.N):
            rho_vec_half_L[i] = rho_vec[i] + \
                0.5*self.minmod(rho_vec[self.right(i)]-rho_vec[i], \
                rho_vec[i]-rho_vec[self.left(i)])
            rho_vec_half_R[i] = rho_vec[self.right(i)] - \
                0.5*self.minmod(rho_vec[self.right(self.right(i))] - \
                rho_vec[self.right(i)], rho_vec[self.right(i)]- \
                rho_vec[i])
        f_vec_half = np.zeros(self.N)
        for i in range(self.N):
            s = 0
            if (rho_vec_half_L[i] - rho_vec_half_R[i]) != 0:
                s = (self.f(rho_vec_half_L[i]) - self.f(rho_vec_half_R[i]))/ \
                    (rho_vec_half_L[i] - rho_vec_half_R[i])
            if s>0:
                f_vec_half[i] = self.f(rho_vec_half_L[i])
            else:
                f_vec_half[i] = self.f(rho_vec_half_R[i])
        rho_vec_dt = np.zeros(self.N)
        for i in range(self.N):
            rho_vec_dt[i] = -1/(self.delta_x)*(f_vec_half[i] - f_vec_half[self.left(i)])
        return rho_vec_dt

    def derivs(self, rho_vec, t):
        rho_vec_dt = np.zeros(self.N)
        for i in range(self.N):
            rho_vec_dt[i] = -1/(2*self.delta_x)*(self.f(rho_vec[self.right(i)]) - self.f(rho_vec[self.left(i)]))
        return rho_vec_dt


    def f(self, rho):
        return rho*self.u_max*(1-rho/self.rho_max)

    def left(self,i):
        if i == 0:
            return self.N - 1
        return i - 1

    def right(self,i):
        if i == self.N - 1:
            return 0
        return i + 1

    def minmod(self, a, b):
        if a*b > 0:
            if np.abs(a) <= np.abs(b):
                return a
            else:
                return b
        return 0

    def graph_rho(self, num_plots):
        for i in range(len(self.t_vec)):
            if i%int(self.M/num_plots) == 0:
                plt.plot(self.x_vec, self.rho_mat[i], label="t=" + str(self.t_vec[i]))
        plt.xlabel(r'x')
        plt.ylabel(r'$\rho$')
        plt.title(r'Plot of $\rho$ vs. $x$ for $t=0$')
        plt.legend()
        plt.show()

def main():
    flow = TrafficFlow(0.5, 10e-3, 299)
    t = 20
    flow.compute_rho(t, 5*t+1)
    flow.graph_rho(10)

if __name__ == "__main__":
    main()
