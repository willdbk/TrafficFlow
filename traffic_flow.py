import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys

# The traffic flow class allows us to model traffic flow. It uses the method
# of lines along with a High-resolution shock-capturing schemes (HRSC).
class TrafficFlow:
    # Creates the rho init vector based on the initial conditions.
    def __init__(self, rho_bar, delta_rho, N, x_center = 0.5, \
            lam = 0.1, rho_max = 1, u_max = 1):
        self.rho_bar = rho_bar
        self.delta_rho = delta_rho
        self.u_max = u_max
        self.x_center = x_center
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
    def fill_rho_mat(self, max_t, time_steps):
        self.time_steps = time_steps
        self.delta_t = max_t/(time_steps - 1)
        self.t_vec = np.linspace(0, max_t, time_steps)
        self.rho_mat = odeint(self.derivs, self.rho_init_vec, self.t_vec)

    # Fills the rho matrix using the Method of Lines along with the
    # High-resolution shock capturing (HRSC) scheme.
    def fill_rho_mat_HRSC(self, max_t, time_steps):
        self.time_steps = time_steps
        self.delta_t = max_t/(time_steps - 1)
        self.t_vec = np.linspace(0, max_t, time_steps)
        self.rho_mat = odeint(self.derivs_HRSC, self.rho_init_vec, \
            self.t_vec, mxstep=5000000)

    # Computes rho derivatives without HRSC, using center difference method.
    def derivs(self, rho_vec, t):
        rho_vec_dt = np.zeros(self.N)
        for i in range(self.N):
            rho_vec_dt[i] = -1/(2*self.delta_x) * \
                (self.f(rho_vec[self.right(i)]) - self.f(rho_vec[self.left(i)]))
        return rho_vec_dt

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
            if i%int(self.time_steps/num_plots) == 0:
                plt.plot(self.x_vec, self.rho_mat[i], label="t=" + \
                    str(round(100*self.t_vec[i])/100))
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\rho$')
        title = r'Plot of $\rho$ vs. $x$ for $\bar{\rho}=$' + str(self.rho_bar) \
            + r' and $\delta\rho=$' + str(self.delta_rho)
        plt.title(title)
        plt.legend()
        plt.show()

    # Returns the analytic expectation for the characteristic speed as well as
    # the numerical calculation for the characteristic speed.the numerical
    # calculation for the characteristic speed.
    def get_characteristic_speed(self):
        direction = -1
        i = 1
        while(True):
            x_pos_of_next_max = self.x_vec[np.where(self.rho_mat[i] == np.amax(self.rho_mat[i]))[0][0]]
            if x_pos_of_next_max > self.x_center:
                direction = 1
                break
            elif x_pos_of_next_max > self.x_center:
                break
            i+=1
        analytic_speed = self.u_max*(1-(2*self.rho_bar)/(self.rho_max))*direction

        total_change_in_x = 0
        total_change_in_t = 0

        for i in range(len(self.rho_mat) - 1):
            curr_max = np.where(self.rho_mat[i] == np.amax(self.rho_mat[i]))[0][0]
            next_max = np.where(self.rho_mat[i+1] == np.amax(self.rho_mat[i+1]))[0][0]
            if direction == -1:
                # didn't wrap
                if self.x_vec[next_max] <= self.x_vec[curr_max]:
                    total_change_in_x += self.x_vec[next_max] - self.x_vec[curr_max]
                # did wrap
                else:
                    total_change_in_x += self.x_vec[next_max] - (1 + self.x_vec[curr_max])
            else:
                # didn't wrap
                if self.x_vec[next_max] >= self.x_vec[curr_max]:
                    total_change_in_x += self.x_vec[next_max] - self.x_vec[curr_max]
                # did wrap
                else:
                    total_change_in_x += (1 + self.x_vec[next_max]) - self.x_vec[curr_max]

            total_change_in_t += self.t_vec[i+1] - self.t_vec[i]
        num_speed = total_change_in_x / total_change_in_t
        return analytic_speed, num_speed

    # Models the position of a number of cars based on the computed row values
    # for a particular traffic flow.
    def model_cars(self, num_cars, pos_max):
        # each row is a particular car, each column is a time step
        car_pos_mat = np.zeros((num_cars, self.time_steps))
        init_pos = np.linspace(0.0, pos_max, num_cars)

        for i in range(num_cars):
            car_pos_mat[i][0] = init_pos[i]
            u_vec = np.zeros(self.time_steps - 1)
            for j in range(0, self.time_steps - 1):
                car_pos = car_pos_mat[i][j]
                x_index = int(car_pos*(self.N-1))
                rho = self.rho_mat[j][x_index]
                u_vec[j] = self.u_max*(1-(rho/self.rho_max))
                pos = car_pos_mat[i][j] + u_vec[j]*self.delta_t
                if pos >=1 :
                    pos -= 1
                car_pos_mat[i][j + 1] = pos

        self.car_pos_mat = car_pos_mat

    # Generates an t vs. x plot of the position of the cars as created in the
    # model cars method.
    def graph_cars(self):
        for i in range(len(self.car_pos_mat)):
            plt.plot(self.car_pos_mat[i], self.t_vec, label = "car " +  str(i))
        plt.legend()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$t$')
        title = r'Plot of $t$ vs. $x$ for $\bar{\rho}=$' + str(self.rho_bar) \
            + r' and $\delta\rho=$' + str(self.delta_rho)
        plt.title(title)
        plt.show()


# Creates an instance of the TrafficFlow class, fills the rho matrix and graphs
# values of rho for different times t > 0. Computes the analytical and
# numerical characteristic speed. Creates a model of cars moving in this
# traffic flow and graphs the motion of these cars.
def main():
    rho_bar = 0.5 if len(sys.argv) <= 1 else float(sys.argv[1])
    delta_rho = 0.1 if len(sys.argv) <= 2 else float(sys.argv[2])
    N = 99 if len(sys.argv) <= 3 else int(sys.argv[3])
    max_t = 1 if len(sys.argv) <= 4 else float(sys.argv[4])
    timesteps = 101 if len(sys.argv) <= 5 else float(sys.argv[5])

    flow = TrafficFlow(rho_bar, delta_rho, N)
    flow.fill_rho_mat_HRSC(max_t, timesteps)
    analytic_speed, num_speed = flow.get_characteristic_speed()
    print("analytic_speed: " + str(analytic_speed))
    print("num_speed: " + str(num_speed))
    flow.graph_rho(5)
    # flow.model_cars(5, 0.3)
    # flow.graph_cars()


if __name__ == "__main__":
    main()
