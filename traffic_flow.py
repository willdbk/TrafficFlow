import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# The traffic flow class allows us to model traffic flow. It uses the method
# of lines along with a High-resolution shock-capturing schemes (HRSC).
class TrafficFlow:
    # Computes the initial rho values based on the input conditions.
    def __init__(self, rho_bar, delta_rho, N, x_center = 0.5, \
            lam = 0.1, rho_max = 1, u_max = 1):
        self.rho_bar = rho_bar
        self.delta_rho = delta_rho
        self.u_max = u_max
        self.rho_max = rho_max
        self.delta_x = 1./N
        self.N = N
        self.rho_init_vec = np.zeros(N)
        self.x_vec = np.zeros(N)
        for i in range(N):
            self.x_vec[i] = (i+0.5)/N
            self.rho_init_vec[i] = rho_bar + \
            delta_rho*math.exp(-(((self.x_vec[i] - x_center)/lam)**2))

    # Fills the rho matrix using the basic Method of Lines.
    def fill_rho_mat(self, t, M):
        self.M = M
        self.t_vec = np.linspace(0, t, M)
        self.rho_mat = odeint(self.derivs, self.rho_init_vec, self.t_vec)

    # Fills the rho matrix using the Method of Lines along with the
    # High-resolution shock capturing (HRSC) scheme.
    def fill_rho_mat_HRSC(self, t, M):
        self.M = M
        self.t_vec = np.linspace(0, t, M)
        integral = odeint(self.derivs_HRSC, self.rho_init_vec, \
            self.t_vec, mxstep=5000000)
        self.rho_mat = integral

    # Computes the derivatives of rho using the HRSC method.
    def derivs_HRSC(self, rho_vec, t):
        # These two arrays store rho values based on the left and  right slopes,
        # respectively, for the half-indicies between each rho value. The value
        # at index 0 correspondes to rho_(1/2), index 1 to rho_(3/2), etc.
        rho_vec_half_L = np.zeros(self.N)
        rho_vec_half_R = np.zeros(self.N)
        # Uses linear "total variation diminishing" (TVD) scheme to extrapolate.
        for i in range(self.N):
            rho_vec_half_L[i] = rho_vec[i] + \
                0.5*self.minmod(rho_vec[self.right(i)]-rho_vec[i], \
                rho_vec[i]-rho_vec[self.left(i)])
            rho_vec_half_R[i] = rho_vec[self.right(i)] - \
                0.5*self.minmod(rho_vec[self.right(self.right(i))] - \
                rho_vec[self.right(i)], rho_vec[self.right(i)]- \
                rho_vec[i])
        f_vec_half = np.zeros(self.N)
        # This solves the Riemann problem by updating f_(i+1/2) based on sign(s).
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
        #create array of rho derivatives using f values
            rho_vec_dt[i] = -1/(self.delta_x)*(f_vec_half[i] - \
            f_vec_half[self.left(i)])
        return rho_vec_dt

    # Computes rho derivs without HRSC, using center difference method.
    def derivs(self, rho_vec, t):
        rho_vec_dt = np.zeros(self.N)
        for i in range(self.N):
            rho_vec_dt[i] = -1/(2*self.delta_x) * \
                (self.f(rho_vec[self.right(i)]) - self.f(rho_vec[self.left(i)]))
        return rho_vec_dt

    # This function returns the anti-derivative of the characteristic speed.
    def f(self, rho):
        return rho*self.u_max*(1-rho/self.rho_max)

    # Left and right allow for periodic boundary conditions.
    def left(self,i):
        if i == 0:
            return self.N - 1
        return i - 1
    def right(self,i):
        if i == self.N - 1:
            return 0
        return i + 1

    # Minmod diminisher function is necessary for linear TVD scheme.
    def minmod(self, a, b):
        if a*b > 0:
            if np.abs(a) <= np.abs(b):
                return a
            else:
                return b
        return 0

    # This creates a plot of rho as a function of x for various t values.
    def graph_rho(self, num_plots):
        for i in range(len(self.t_vec)):
            if i%int(self.M/num_plots) == 0:
                plt.plot(self.x_vec, self.rho_mat[i], label="t=" + \
                    str(self.t_vec[i]))
        plt.xlabel(r'x')
        plt.ylabel(r'$\rho$')
        title = r'Plot of $\rho$ vs. $x$ for $\bar{\rho}=$' + str(self.rho_bar) \
            + r' and $\delta\rho=$' + str(self.delta_rho)
        plt.title(title)
        plt.legend()
        plt.show()

    def get_characteristic_speed(self):
        analytic_speed = self.u_max*(1-(2*self.rho_bar)/(self.rho_max))
        index_of_last_max = np.where(self.rho_mat[-1] == np.amax(self.rho_mat[-1]))
        index_of_first_max = np.where(self.rho_mat[0] == np.amax(self.rho_mat[0]))
        change_in_x = self.x_vec[index_of_last_max] - self.x_vec[index_of_first_max]
        change_in_t = self.t_vec[-1] - self.t_vec[0]
        num_speed = change_in_x / change_in_t
        return analytic_speed, num_speed



# Creates an instance of the TrafficFlow class, fills the rho matrix and graphs it.
def main():
    flow = TrafficFlow(0.5, 0.1, 99)
    t = 10
    flow.fill_rho_mat_HRSC(t, t + 1)
    analytic_speed, num_speed = flow.get_characteristic_speed()
    print("analytic_speed: " + str(analytic_speed))
    print("num_speed: " + str(num_speed))
    flow.graph_rho(5)


if __name__ == "__main__":
    main()
