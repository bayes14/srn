from srn.main import SRN, create_network

g1 = create_network(
    reactions=[("0S", "1S", 4), ("1S", "0S", 3), ("2S", "4S", 2), ("4S", "2S", 1)]
)

g2 = create_network(
    reactions=[("1S", "3S", 50), ("3S", "2S", 1), ("2S", "4S", 5), ("4S", "1S", 1)]
)

s1 = SRN(g1)
s2 = SRN(g2)


def test_Omega():
    assert set(s1.Omega) == set([1, -1, 2, -2])
    assert set(s2.Omega) == set([2, -1, 2, -3])


def test_Omega_minus():
    assert set(s1.Omega_minus) == set([-1, -2])
    assert set(s2.Omega_minus) == set([-1, -3])


def test_Omega_plus():
    assert set(s1.Omega_plus) == set([1, 2])
    assert set(s2.Omega_plus) == set([2])


def test_omega_ast():
    assert s1.omega_ast == 1
    assert s2.omega_ast == 1


def test_omega_minus():
    assert s1.omega_minus == -2
    assert s2.omega_minus == -3


def test_omega_plus():
    assert s1.omega_plus == 2
    assert s2.omega_plus == 2


def test_i_omega():
    assert s1.i_omega(s1.Omega) == [0, 1, 2, 4]
    assert s2.i_omega(s2.Omega) == [1, 3, 1, 4]


def test_0_omega():
    assert s1.o_omega(s1.Omega) == [1, 0, 4, 2]
    assert s2.o_omega(s2.Omega) == [3, 2, 3, 1]


def test_i():
    assert s1.i == 0
    assert s2.i == 1


def test_o_minus():
    assert s1.o_minus == 0
    assert s2.o_minus == 1


def test_i_plus():
    assert s1.i_plus == 0
    assert s2.i_plus == 1


def test_o():
    assert s1.o == 0
    assert s2.o == 1


def test_L():
    assert s1.L[0] == 2
    assert s2.L[0] == 0
    
def test_U():
    assert s1.U[0] == 3
    assert s2.U[0] == 2


def test_m_ast():
    assert s1.m_ast == 3
    assert s2.m_ast == 4


def test_kappa():
    assert s1.kappa == [4, 3, 2, 1]
    assert s2.kappa == [50, 1, 5, 1]


def test_y():
    assert s1.y == [0, 1, 2, 4]
    assert s2.y == [1, 3, 2, 4]


def test_G():
    assert s1.G[0] == 2 * 3**2 / 4**2
    assert s1.G[1] == 2 * 3 / 4


def test_eta():
    assert s1.eta(0, 10) == 4
    assert s1.eta(1, 10) == 30
    assert s1.eta(2, 10) == 2 * 10 * 9
    assert s1.eta(3, 10) == 1 * 10 * 9 * 8 * 7
    assert s2.eta(0, 10) == 500
    assert s2.eta(1, 10) == 1 * 10 * 9 * 8
    assert s2.eta(2, 10) == 5 * 10 * 9
    assert s2.eta(3, 10) == 1 * 10 * 9 * 8 * 7


def test_lam():
    assert s1.lam(-1, 0) == 0
    assert s1.lam(-1, 1) == 3
    assert s1.lam(-1, 2) == 6
    assert s2.lam(2, 10) == 500 + 5 * 10 * 9


def test_B():
    assert s1.B(0) == [-2]
    assert s2.B(0) == [-3]
    
def test_s():
    assert s1.s == 0
    assert s2.s == 1

def test_c():
    assert s1.c(0, s1.U[0] + 1) < 0
    assert s2.c(0, s2.U[0] + 1) < 0
