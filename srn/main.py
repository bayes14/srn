import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.linalg import inv
from sympy import Matrix
import seaborn as sns
import pandas as pd
import graphviz as gr
from typing import List, Tuple, Optional, Set
from cvxopt import solvers, matrix


class SRN:
    def __init__(self, g: nx.DiGraph) -> None:
        self.g = g

    @property
    def Omega(self) -> List[int]:
        Omega = []
        for edge in self.g.edges:
            head = int(edge[0].split("S")[0])
            tail = int(edge[1].split("S")[0])
            Omega.append(tail - head)
        return Omega

    @property
    def Omega_minus(self) -> List[int]:
        return [w for w in self.Omega if w < 0]

    @property
    def Omega_plus(self) -> List[int]:
        return [w for w in self.Omega if w > 0]

    @property
    def omega_ast(self) -> int:
        return math.gcd(*self.Omega)

    @property
    def omega_minus(self) -> int:
        return int(min([w / self.omega_ast for w in self.Omega_minus]))

    @property
    def omega_plus(self) -> int:
        return int(max([w / self.omega_ast for w in self.Omega_plus]))

    def i_omega(self, Omega: List[int]) -> List[int]:
        i_omega_list = []
        for omega in Omega:
            species = []
            for edge in self.g.edges:
                head = int(edge[0].split("S")[0])
                tail = int(edge[1].split("S")[0])
                if tail - head == omega:
                    species.append(head)
            i_omega_list.append(min(species))
        return i_omega_list

    def o_omega(self, Omega: List[int]) -> List[int]:
        return [a + b for a, b in zip(self.i_omega(Omega), Omega)]

    @property
    def i(self) -> int:
        return min(self.i_omega(self.Omega))

    @property
    def i_plus(self) -> int:
        return min(self.i_omega(self.Omega_plus))

    @property
    def o(self) -> int:
        return min(self.o_omega(self.Omega))

    @property
    def o_minus(self) -> int:
        return min(self.o_omega(self.Omega_minus))

    @property
    def N(self) -> Set[int]:
        return set(range(0, min(self.i, self.o)))

    @property
    def T(self) -> Set[int]:
        return set(range(self.o, self.i))

    @property
    def E(self) -> Set[int]:
        return set(range(self.i, max(self.i_plus, self.o_minus)))

    @property
    def P(self) -> Optional[List[str]]:
        if len(self.T) == 0:
            return [
                f"{self.omega_ast}N_0 + {s}"
                for s in range(self.o_minus, self.o_minus + self.omega_ast)
            ]
        elif len(self.T) >= self.omega_ast:
            return []
        else:
            return [
                f"{self.omega_ast}N_0 + {s}"
                for s in range(self.i_plus, self.o_minus + self.omega_ast)
            ]

    @property
    def Q(self) -> Optional[List[str]]:
        if len(self.T) >= self.omega_ast:
            return [
                f"{self.omega_ast}N_0 + {s}"
                for s in range(self.i_plus, self.i_plus + self.omega_ast)
            ]
        elif len(self.T) == 0:
            return []
        else:
            return [
                f"{self.omega_ast}N_0 + {s}"
                for s in range(self.o_minus, self.i_plus + self.omega_ast)
            ]

    @property
    def s_list(self) -> np.ndarray:
        s_list = np.array(
            range(max(self.i_plus, self.o_minus), self.o_minus + self.omega_ast)
        )
        return s_list

    @property
    def L(self) -> List[int]:
        Ls = (
            np.ceil(
                (
                    np.array(self.i_omega([self.omega_ast * self.omega_minus]))
                    - self.s_list
                )
                / self.omega_ast
            )
            + self.omega_minus
        )
        return [int(Ls[i]) for i in range(self.s_list.shape[0])]

    @property
    def U(self) -> List[int]:
        Us = (
            np.ceil(
                (
                    np.array(self.i_omega([self.omega_ast * self.omega_minus]))
                    - self.s_list
                )
                / self.omega_ast
            )
            - 1
        )
        return [int(Us[i]) for i in range(self.s_list.shape[0])]

    @property
    def m_ast(self) -> int:
        return self.omega_plus - self.omega_minus - 1

    @property
    def H(self) -> np.ndarray:
        H = np.zeros((self.L[0], self.U[0]))
        for m in range(self.L[0]):
            sum_lam = 0
            for w in set(self.Omega):
                sum_lam += self.lam(w, m)
            for n in range(self.U[0]):
                if m == n:
                    H[m, n] = 1
                else:
                    H[m, n] = kronecker(m, n) - self.lam(m - n, n) / sum_lam
        return H

    @property
    def kappa(self) -> List[int]:
        kappa = []
        for edge in self.g.edges(data=True):
            kappa.append(edge[2]["weight"])
        return kappa

    @property
    def y(self) -> List[int]:
        y = []
        for edge in self.g.edges:
            y.append(int(edge[0].split("S")[0]))
        return y

    def eta(self, k: int, x: int) -> float:
        if x >= self.y[k]:
            return self.kappa[k] * math.factorial(x) / math.factorial(x - self.y[k])
        else:
            return 0

    def lam(self, w: int, x: int) -> float:
        lam = 0
        for i, omega in enumerate(self.Omega):
            if w == omega:
                lam += self.eta(i, x)
        return lam

    @property
    def s(self):
        return max(self.i_plus, self.o_minus)

    def lam_s(self, w: int, x: int) -> float:
        lam = 0
        for i, omega in enumerate(self.Omega):
            if w == omega:
                lam += self.eta(i, x + self.s)
        return lam

    @property
    def H1_and_H2(self):
        H1 = self.H[: self.L[0], : self.L[0]]
        H2 = self.H[: self.L[0], self.U[0] - self.L[0] + 1 :]
        return H1, H2

    @property
    def G(self) -> np.ndarray:
        H_1, H_2 = self.H1_and_H2
        return np.matmul(-inv(H_1), H_2)

    def B(self, k) -> List[int]:
        B = []
        k_prime = self.omega_minus + k + 0.5
        for w in set(self.Omega):
            if k_prime * (w - k_prime) > 0:
                B.append(w)
        return B

    def c(self, k: int, l: int) -> float:
        lam_sum = 0
        for w in self.B(k):
            lam_sum += self.lam_s(w, l)
        return np.sign(self.omega_minus + k + 0.5) * lam_sum

    def f(self, k: int, l: int) -> float:
        return -self.c(k, l - k) / self.c(0, l)

    def gamma_mat(self, n_rows: int) -> np.ndarray:
        gamma = np.zeros((n_rows, self.U[0] - self.L[0] + 1), dtype=np.longfloat)
        for l in range(n_rows):
            for j in range(self.L[0], self.U[0] + 1):
                if (l <= self.L[0] - 1) and (j == self.U[0]):
                    gamma[l, j - self.L[0]] = 0
                if (l >= self.L[0]) and (l <= self.U[0]):
                    gamma[l, j - self.L[0]] = kronecker(l, j)
                if (l >= 0) and (l <= self.L[0] - 1) and (j < self.U[0]):
                    gamma[l, j - self.L[0]] = self.G[l, j - self.L[0]]
                if l > self.U[0]:
                    sum_gam_f = 0
                    for k in range(1, self.m_ast + 1):
                        sum_gam_f += gamma[l - k, j - self.L[0]] * self.f(k, l)
                    gamma[l, j - self.L[0]] = sum_gam_f

        return gamma

    def rref_solution(self, n_rows: int) -> np.ndarray:
        "Legacy method finding the generating terms by reduced row echelon form"
        gamma_mat = self.gamma_mat(n_rows=n_rows)
        a_mat = gamma_mat[n_rows - (self.U[0] - self.L[0]) :, :]
        v = Matrix(a_mat).rref()
        w = -np.append(np.array(v[0])[:, -1], -1)
        w = w / np.sum(w)
        return w.astype(np.longfloat)

    def quadratic_solution(self, n_rows: int):
        gamma_mat = self.gamma_mat(n_rows=n_rows)
        a_mat = gamma_mat[n_rows - (self.U[0] - self.L[0]) :, :].astype(np.double)

        P = matrix(np.eye(a_mat.shape[1]))
        q = matrix(np.zeros(a_mat.shape[1]))
        G = matrix(-np.eye(a_mat.shape[1]))
        h = matrix(np.zeros(a_mat.shape[1]))
        A = matrix(np.vstack([a_mat, np.ones(a_mat.shape[1])]))
        b = matrix(np.matrix(np.hstack([np.zeros(a_mat.shape[0]), np.ones(1)]))).T

        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = 150
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol["x"]).T

    def stationary_distribution(self, n_max: int) -> np.ndarray:
        gamma = self.gamma_mat(n_max)
        # w = self.rref_solution(500)
        w = self.quadratic_solution(500)
        # pi = exp_log_sum_exp_log(gamma, w)
        pi = np.matmul(w, gamma.T).reshape(-1)
        pi = pi / np.sum(pi)
        return np.hstack([np.zeros(self.s) * np.nan, pi])

    def info(self):
        info = pd.DataFrame(
            {
                "Ω": str(set(self.Omega)),
                "Ω-": str(set(self.Omega_minus)),
                "Ω+": str(set(self.Omega_plus)),
                "ω⁎": self.omega_ast,
                "ω-": self.omega_minus,
                "ω+": self.omega_plus,
                "i+": self.i_plus,
                "i": self.i,
                "o-": self.o_minus,
                "o": self.o,
                "L": str(self.L),
                "U": str(self.U),
                "N": str(self.N),
                "T": str(self.T),
                "E": str(self.E),
                "P": str(self.P),
                "Q": str(self.Q),
            },
            index=["Info"],
        )
        return info.T


def create_network(reactions: List[Tuple[str, str, float]]) -> nx.DiGraph:
    "The reacions must be written in the form 0S and 1S instead of 0 and S respectively"
    g = nx.DiGraph()
    g.add_weighted_edges_from(reactions)
    return g


def plot_network(g: nx.DiGraph) -> gr.Digraph:
    G = gr.Digraph()
    G.attr("graph", ranksep="1", nodesep="1")
    weight_dict = nx.get_edge_attributes(g, "weight")

    for edge in g.edges:
        tail_name = edge[0].split("S")[0]
        head_name = edge[1].split("S")[0]
        if tail_name == "0":
            tail_name = "0"
        elif tail_name == "1":
            tail_name = "S"
        else:
            tail_name = edge[0]
        if head_name == "0":
            head_name = "0"
        elif head_name == "1":
            head_name = "S"
        else:
            head_name = edge[1]
        G.node(tail_name)
        G.edge(tail_name, head_name, str(weight_dict[edge]))
    return G


def plot_distribution(pi: np.ndarray, kind: str = "line") -> None:
    if kind == "line":
        sns.lineplot(pi, marker="o")
    elif kind == "bar":
        sns.barplot(pi)
    else:
        raise Exception("kind must be one of line or bar")
    plt.title("Stationary Distribution")
    plt.xlabel("State x")
    plt.ylabel("Probability")


def plot_generating_terms(s: SRN, n: int) -> None:
    terms_df = pd.DataFrame()
    for n in range(1, n):
        try:
            terms_df = pd.concat(
                [terms_df, pd.DataFrame(s.quadratic_solution(n_rows=n))], axis=0
            )
        except Exception:
            print("Could not find solution to quadratic problem")
    terms_df = terms_df.reset_index(drop=True)
    sns.lineplot(terms_df)
    plt.title("Generating Terms")
    plt.xlabel("Index n")


def plot_gamma(s: SRN, n: int, scale=None) -> None:
    gamma_df = pd.DataFrame(s.gamma_mat(n_rows=n))
    if scale:
        gamma_df = gamma_df.map(
            lambda val: np.log1p(val) if val >= 0 else -np.log1p(-val)
        )  # type: ignore
    sns.lineplot(gamma_df)
    plt.title("Generating Terms")
    plt.xlabel("Index n")


def kronecker(a: float, b: float) -> int:
    if a == b:
        return 1
    else:
        return 0


# def exp_log_sum_exp_log(v, w):
#     sign_v = np.sign(v)
#     log_v = np.nan_to_num(clog(w)) + np.nan_to_num(clog(np.abs(v)))
#     max_val = np.max(sign_v * log_v, axis=1)
#     sum_exp_val = np.sum(np.exp(log_v.T - max_val).T * sign_v, axis=1)
#     log_sum_exp_val = max_val + clog(np.abs(sum_exp_val))
#     s = np.exp(log_sum_exp_val)
#     return s


# def clog(x):
#     x[x == 0] = 10 ** (-16)
#     return np.log(x)


if __name__ == "__main__":
    g = create_network(
        reactions=[("1S", "2S", 10), ("2S", "3S", 10), ("3S", "2S", 1), ("3S", "1S", 2)]
    )
    s = SRN(g)
    plot_gamma(s, n=10)

    # v = np.array([[-0.5, 0.5], [0.4, 0.6]])
    # print(v)
