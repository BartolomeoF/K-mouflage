import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Function, exp, pi, sqrt, lambdify, solve, Derivative
from scipy.optimize import newton
from scipy.interpolate import InterpolatedUnivariateSpline

class KmouExpansionJordan:
    def __init__(self, lamb_val=2, beta=0.2, n=2, K0=1, Om0=0.3089,  a_ini=3e-5, a_fin=1):
        self.lamb_val = lamb_val
        self.beta = beta
        self.n = n
        self.K0 = K0
        self.a_ini = a_ini
        self.a_fin = a_fin
        self.Om0_val = Om0
        self.H0_hinvMpc = 1/2997.92458 # in h/Mpc units
        
        self._initialize_symbols()
        self._setup_equations()
        
    def _initialize_symbols(self):
        # self.phi, self.phi_p, self.phi_pp = symbols(r'\phi \phi^{\prime} \phi^{\prime\prime}')
        self.X, self.rho_m, self.G = symbols(r'X \rho_m G')
        self.a = symbols('a', positive=True)
        self.phi = Function(r'\phi')(self.a)
        self.phi_a = symbols(r'\phi_a')
        self.E = Function('E')(self.a)
        self.lamb = symbols(r'\lambda')
        
    def _setup_equations(self):
        H = self.H0_hinvMpc*self.E
        self.H_conf = self.a*H
        self.Om= self.Om0_val*self.a**(-3)
        # H_LCDM = self.H0_hinvMpc*sqrt(self.Om0_val*a**(-3) + self.Ol0)

        phi_p = self.a*self.H_conf*self.phi.diff(self.a)
        # phi_pp = self.a*self.H_conf*(phi_p).diff(a)        
        
        A = exp(self.beta * self.phi)
        self.A = A
        rho_m0 = self.Om0_val * self.H0_hinvMpc**2 / (8 * pi * self.G / 3)
        
        K = (-1 + self.X + self.K0 * self.X**self.n)
        K_x = K.diff(self.X)
        X_bar = A**2 * phi_p**2 / (2 * self.lamb**2 * self.a**2 * self.H0_hinvMpc**2)
        K_bar = K.subs(self.X, X_bar)
        K_x_bar = K_x.subs(self.X, X_bar)
        self.K_x_bar = K_x_bar
        
        M_pl = 1 / sqrt(8 * pi * self.G)
        rho_phi = M_pl**2 * self.H0_hinvMpc**2 * self.lamb**2 / A**4 * (2 * X_bar * K_x_bar - K_bar)
        p_phi = M_pl**2 * self.H0_hinvMpc**2 * self.lamb**2 / A**4 * K_bar
        O_phi = rho_phi / (3 * self.H0_hinvMpc**2 * M_pl**2)
        
        eps2 = self.a * self.beta * self.phi.diff(self.a)
        eps1 = 2 * self.beta**2 / K_x_bar
        
        E_kmou_sq = A**2 / (1 - eps2)**2 * (self.Om + O_phi)
        self.E_kmou = solve(E_kmou_sq - self.E**2, self.E)[1] # Only works for n=2
        self.E_kmou_a = -3 / 2 * self.H0_hinvMpc / self.H_conf * (
            A**2 / (1 - eps2) * (self.Om + (rho_phi + p_phi) / (3 * M_pl**2 * self.H0_hinvMpc**2)) + 
            2 * A**2 / (3 * (1 - eps2)**2) * (eps2 - 1 / (1 - eps2) * self.a * eps2.diff(self.a)) * (self.Om + O_phi)
            )
        
        self.dphia_o_da_sym_eq = self.H_conf * (A**(-2) * self.a**3 * self.H_conf * self.phi.diff(self.a) * K_x_bar).diff(self.a) + self.beta * rho_m0 / M_pl**2
        self.dphia_o_da_sym_eq = self.dphia_o_da_sym_eq.subs(self.E.diff(self.a), self.E_kmou_a)
        self.dphia_o_da_sym_eq = solve(self.dphia_o_da_sym_eq.subs(self.phi.diff(self.a), self.phi_a), Derivative(self.phi_a, self.a))[0]
        self.dphia_o_da_sym_eq = self.dphia_o_da_sym_eq.subs(self.E, self.E_kmou).subs(self.phi.diff(self.a), self.phi_a)

    def _enrich_sol(self):
        self.A_fun = lambdify(self.phi, self.A)
        self.E_kmou_fun = lambdify((self.a, self.phi, self.phi_a), self.E_kmou.subs(self.phi.diff(self.a), self.phi_a)
        .subs(self.lamb, self.lamb_val))
        self.E_kmou_a_fun = lambdify((self.a, self.phi, self.phi_a), 
                                self.E_kmou_a.subs(self.phi.diff(self.a), self.phi_a
                                                   ).subs(Derivative(self.phi_a, self.a), self.dphia_o_da_sym_eq 
                                                   ).subs(self.E, self.E_kmou
                                                   ).subs(self.phi.diff(self.a), self.phi_a)
                                                    .subs(self.lamb, self.lamb_val))
        
    def dum_fun_phi(self, t, vec):
        '''Convenience function for solve_ivp'''
        return (self.dphi_o_da_eq(t, vec[0], vec[1]),
                self.dphia_o_da_eq(t, vec[0], vec[1]))
        
    def get_conf_fact(self, a):
        phi_val = self.sol.sol(a)[0]
        return self.A_fun(phi_val)
        
    def get_a_Jor(self, a_Ein):
        a_vals_J = np.linspace(0.1,self.a_fin, 1000)
        A_vals = self.get_conf_fact(a_vals_J)
        a_Jor_fun = InterpolatedUnivariateSpline(a_vals_J/A_vals, a_vals_J)
        a_Jor = a_Jor_fun(a_Ein)
        return a_Jor
    
    def get_E_Ein(self, a_Ein):
        a_Jor = self.get_a_Jor(a_Ein)
        phi_val, phi_a_val_Jor = self.sol.sol(a_Jor)
        A = self.get_conf_fact(a_Jor)
        phi_a_val_Ein = A*phi_a_val_Jor/(1-self.beta*a_Jor*phi_a_val_Jor)
        E_Jor = self.E_kmou_fun(a_Jor, phi_val, phi_a_val_Jor)
        E_Ein = E_Jor * A / (1 + a_Ein*self.beta*phi_a_val_Ein)
        return E_Ein
    
    def get_Delta_E(self, lamb_val, frame):
        self.lamb_val = lamb_val
        self.solve()
        if frame=='Jordan':
            phi_val, phi_a_val = self.sol.sol(self.a_target)
            E_val = self.E_kmou_fun(self.a_target, phi_val, phi_a_val)
        elif frame=='Einstein':
            E_val = self.get_E_Ein(self.a_target)
            
        self.Delta_E = np.array(E_val - self.E_target) / self.E_target
        return self.Delta_E
        
    def tune_lambda(self, E_target=1, a_target=1, frame='Jordan', maxiter=5):
        self.a_target = a_target
        self.E_target = E_target
        self.lamb_val = newton(self.get_Delta_E, self.lamb_val, maxiter=maxiter, args=[frame])
        return None
        
    def eval(self, a_vals=None):
        self.a_vals = np.logspace(-3, 0, 1000) if a_vals is None else a_vals
        phi_vals, phi_a_vals = self.sol.sol(self.a_vals)
        return (self.a_vals, phi_vals, phi_a_vals, self.E_kmou_fun(self.a_vals, phi_vals, phi_a_vals), 
                self.E_kmou_a_fun(self.a_vals, phi_vals, phi_a_vals))
        

    def solve(self):
        self.dphi_o_da_eq = lambdify((self.a, self.phi, self.phi_a), self.phi_a)
        self.dphia_o_da_eq = lambdify((self.a, self.phi, self.phi_a), self.dphia_o_da_sym_eq.subs(self.lamb, self.lamb_val))

        self.sol = solve_ivp(self.dum_fun_phi, t_span=(self.a_ini, self.a_fin), y0=(-1e-15 * self.a_ini, -1), 
                               dense_output=True, rtol=1e-9, atol=1e-9)
        self._enrich_sol()
        return None
        # return self.sol
        
    def get_growth(self):
        D = Function('D')(self.a)
        D_a, E_diffa = symbols('D_a, E_diffa')
        phi_val = lambda dum_a: self.sol.sol(dum_a)[0]
        phi_a_val = lambda dum_a: self.sol.sol(dum_a)[1]
        E_kmou_ofa = lambda dum_a: self.E_kmou_fun(dum_a, phi_val(dum_a), phi_a_val(dum_a))
        E_kmou_a_ofa = lambda dum_a: self.E_kmou_a_fun(dum_a, phi_val(dum_a), phi_a_val(dum_a))
        
        mu_kmou = 1 + 2 * self.beta**2 / (self.K_x_bar)
        diff_eq_kmou = (self.a * self.H_conf * (self.a * self.H_conf * D.diff(self.a)).diff(self.a) 
                        + self.a * self.H_conf * self.H_conf * D.diff(self.a) 
                        - 3 / 2 * self.A**2 * self.Om * self.H0_hinvMpc**2 * self.a**2 * mu_kmou * D).expand()
        # Split 2nd order differential equation in a system of first order differential equations
        D_a_sym_eq = diff_eq_kmou.subs(D.diff(self.a),D_a).subs(self.E.diff(self.a),E_diffa)
        print(solve(D_a_sym_eq, Derivative(D_a,self.a)))

        D_a_eq= lambdify((self.a,D_a,D,self.E,E_diffa,self.phi,self.phi_a), solve(D_a_sym_eq, Derivative(D_a,self.a)
                                                                            )[0].subs(self.lamb,self.lamb_val).subs(
                                                                                self.phi.diff(self.a), self.phi_a))
        D_eq = lambdify((self.a,D_a,D), D_a)
        def dum_fun(t,vec):
            '''Dummy function to adapt the input of solve_ivp'''
            return (D_a_eq(t,vec[0],vec[1], E_kmou_ofa(t),E_kmou_a_ofa(t), phi_val(t), phi_a_val(t)),
                    D_eq(t,vec[0],vec[1]))

        # Compute the solution of the differential equation
        self.D_kmou_J = solve_ivp(dum_fun, t_span=(self.a_ini, self.a_fin), y0=(1,self.a_ini), dense_output=True, rtol=1e-9)
        return self.D_kmou_J
