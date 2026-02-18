import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

start_time = time.time()

# ----------------------- basis --------
ket0 = np.array([[1], [0]], dtype=complex)
ket1 = np.array([[0], [1]], dtype=complex)

# ---------------- identity matrices --------
ii = np.identity(4, dtype=complex)
i1 = np.identity(2, dtype=complex)

# ----------------------------- entropy function --------------------

def entropy(rho, tol=1e-15):
    """
    von Neumann entropy S(rho) = -Tr(rho log2 rho)
    using eigenvalues of a Hermitian matrix (eigvalsh).
    """
    rho = (rho + rho.conj().T) / 2.0
    ev = np.linalg.eigvalsh(rho)
    ev = np.real(ev)
    ev = np.clip(ev, 0.0, 1.0)
    ev = ev[ev > tol]
    if len(ev) == 0:
        return 0.0
    return float(-np.sum(ev * np.log2(ev)))

# ------------------------- partial trace function ----------------

def ptrace(y):
    """
    Partial trace over 2nd qubit (based on your indexing),
    output is 2x2.
    """
    x = np.zeros([2, 2], dtype=complex)
    x[0, 0] = y[0, 0] + y[1, 1]
    x[0, 1] = y[0, 2] + y[1, 3]
    x[1, 0] = y[2, 0] + y[3, 1]
    x[1, 1] = y[2, 2] + y[3, 3]
    return x

# -------------------------- measurement function --------------------

def pmeas(povm, state):
    return np.real_if_close(np.trace(povm @ state))

def conditional_state(povm, state, tol=1e-15):
    p = pmeas(povm, state)
    if np.real(p) < tol:
        return np.identity(2, dtype=complex) / 2.0
    post = povm @ state @ povm.conj().T
    return ptrace(post) / p

# ------------------- Pauli-basis projectors (optimal for Bell-diagonal) -------------------

def pauli_projectors():
    """
    Returns list of (P0, P1) projectors for Z, X, Y measurement bases.
    """
    # Z basis
    Pz0 = ket0 @ ket0.conj().T
    Pz1 = ket1 @ ket1.conj().T

    # X basis: |+>, |->
    ketp = (ket0 + ket1) / np.sqrt(2)
    ketm = (ket0 - ket1) / np.sqrt(2)
    Px0 = ketp @ ketp.conj().T
    Px1 = ketm @ ketm.conj().T

    # Y basis: |+i>, |-i>
    keti = (ket0 + 1j * ket1) / np.sqrt(2)
    ketmi = (ket0 - 1j * ket1) / np.sqrt(2)
    Py0 = keti @ keti.conj().T
    Py1 = ketmi @ ketmi.conj().T

    return [(Pz0, Pz1), (Px0, Px1), (Py0, Py1)]

# --------------------------------------------------------------

eta = 0.5         # reflectivity
p0 = 0.5          # probability of presence of the object
p1 = 1 - p0

l = 40            # range for generating states

holevo = []
c1ss, c2ss, c3ss = [], [], []

S_I4 = entropy(ii / 4)     # 2 bits
S_I2 = entropy(i1 / 2)     # 1 bit

# Precompute POVMs for speed
pauli_meas = pauli_projectors()
I2 = np.identity(2, dtype=complex)
denc=[]
for i11 in range(-l, l + 1):
    for i2 in range(-l, l + 1):
        for i3 in range(-l, l + 1):

            c1 = i11 / l
            c2 = i2 / l
            c3 = i3 / l

            # positivity constraints for Bell-diagonal state
            if (1 - c1 - c2 - c3) >= 0 and (1 - c1 + c2 + c3) >= 0 and (1 + c1 - c2 + c3) >= 0 and (1 + c1 + c2 - c3) >= 0:

                bb = np.zeros([4, 4], dtype=complex)
                bb[0, 0] = 1 + c3
                bb[0, 3] = c1 - c2
                bb[3, 0] = c1 - c2
                bb[1, 1] = 1 - c3
                bb[1, 2] = c1 + c2
                bb[2, 1] = c1 + c2
                bb[2, 2] = 1 - c3
                bb[3, 3] = 1 + c3

                bell = 0.25 * bb

                # returned state
                w2 = (1 - eta) * ii / 4 + eta * bell

                # average ensemble state
                rho_abs = ii / 4
                rhobar = p0 * w2 + p1 * rho_abs

                # entropies needed for Iq
                srhow2 = entropy(w2)
                srhobar = entropy(rhobar)

                # Optimize only over Pauli bases
                r1 = 1e9
                r2 = 1e9

                for (P0, P1) in pauli_meas:

                    povm1 = np.kron(I2, P0)
                    povm2 = np.kron(I2, P1)

                    # ---- Conditional states from w2 (object present hypothesis)
                    rhoc21 = conditional_state(povm1, w2)
                    rhoc22 = conditional_state(povm2, w2)

                    p_out1_0 = pmeas(povm1, w2)
                    p_out2_0 = pmeas(povm2, w2)

                    srhoc2 = p_out1_0 * entropy(rhoc21) + p_out2_0 * entropy(rhoc22)
                    if srhoc2 < r1:
                        r1 = srhoc2

                    # ---- Absent hypothesis conditional states
                    # for rho_abs = I4/4, these are always I2/2, but we keep it explicit
                    p_out1_1 = pmeas(povm1, rho_abs)
                    p_out2_1 = pmeas(povm2, rho_abs)

                    rhoA_abs_y1 = i1 / 2
                    rhoA_abs_y2 = i1 / 2

                    # ---- Total outcome probabilities in the mixture
                    p_y1 = p0 * p_out1_0 + p1 * p_out1_1
                    p_y2 = p0 * p_out2_0 + p1 * p_out2_1

                    # ---- Correct Bayes-weighted conditional states of rhobar
                    rhobarc1 = (p0 * p_out1_0 * rhoc21 + p1 * p_out1_1 * rhoA_abs_y1) / p_y1
                    rhobarc2 = (p0 * p_out2_0 * rhoc22 + p1 * p_out2_1 * rhoA_abs_y2) / p_y2

                    srhobarc = p_y1 * entropy(rhobarc1) + p_y2 * entropy(rhobarc2)

                    if srhobarc < r2:
                        r2 = srhobarc

                # Quantum Holevo term (binary ensemble: w2 vs I/4)
                Iq = srhobar - p0 * srhow2 - p1 * S_I4

                # Classical term (based on optimal Pauli measurements)
                Ic = r2 - p0 * r1 - p1 * S_I2

                ho = Iq - Ic

                holevo.append(np.real_if_close(ho))
                c1ss.append(c1)
                c2ss.append(c2)
                c3ss.append(c3)
                            
#discord after noise----------------------------------------
                c1=eta*c1
                c2=eta*c2
                c3=eta*c3

                ccc1  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                        (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                        + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                        (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
                
                
                sm1 = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
                      + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                      + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
                      + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))
                
                qqq1 = 2 + sm1 - ccc1
                
##discord of average state----------------------------------------
                
                c1=p0*c1
                c2=p0*c2
                c3=p0*c3


                ccc2  = ((( 1 + -1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                        (math.log2(1 + -1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ) +
                        + (( 1 + 1* max(abs(c1),abs(c2),abs(c3)) )/2) * 
                        (math.log2(1 + 1*max(abs(c1),abs(c2),abs(c3)) +10**-15 ) ))
                
                
                sm2 = (0.25*(1-c1-c2-c3)*math.log2(0.25*(1-c1-c2-c3) +10**-15 ) 
                      + 0.25*(1-c1+c2+c3)*math.log2(0.25*(1-c1+c2+c3)+10**-15) 
                      + 0.25*(1+c1-c2+c3)*math.log2(0.25*(1+c1-c2+c3)+10**-15) 
                      + 0.25*(1+c1+c2-c3)*math.log2(0.25*(1+c1+c2-c3)+10**-15))
                
                qqq2 = 2 + sm2 - ccc2
                

                #dis.append(qqq)
                denc.append((p0*qqq1 - qqq2))



plt.plot(denc , holevo,'o' )
plt.xlabel(r'$\delta_{enc}$',fontsize=20)
plt.ylabel(r'$\chi_q - \chi_c$',fontsize=20)
