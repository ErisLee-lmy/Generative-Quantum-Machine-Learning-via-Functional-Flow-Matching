# quantum_ot_forward_backward.py
# Dependencies: numpy, scipy
import numpy as np
from scipy.linalg import expm, logm, eigh, norm, sqrtm
import time, math

# ---------------- basic helpers ----------------
def _make_hermitian(A): return 0.5 * (A + A.conj().T)
def _partial_trace_out(mat, d): return np.einsum('iaja->ij', mat.reshape((d,d,d,d)))
def _partial_trace_in(mat, d): return np.einsum('iaib->ab', mat.reshape((d,d,d,d)))
def _safe_logm(A, reg=1e-12):
    A = _make_hermitian(A)
    from scipy.linalg import logm
    return logm(A + reg * np.eye(A.shape[0]))
def _safe_expm(A):
    from scipy.linalg import expm
    return expm(A)

# ---------------- quantum sinkhorn (entropic) ----------------
def quantum_sinkhorn(rho, sigma, C, eps=0.25, tol=1e-6, max_iter=400, verbose=False):
    d = rho.shape[0]
    C = _make_hermitian(C)
    logK = -C / eps
    logU = np.zeros((d,d), dtype=complex)
    logV = np.zeros((d,d), dtype=complex)
    log_rho = _safe_logm(rho)
    log_sigma = _safe_logm(sigma)
    for it in range(max_iter):
        A = _safe_expm(logK + np.kron(np.eye(d), logV))
        TrA_out = _partial_trace_out(A, d)
        logTrA_out = _safe_logm(TrA_out)
        logU = log_rho - logTrA_out
        B = _safe_expm(logK + np.kron(logU, np.eye(d)))
        TrB_in = _partial_trace_in(B, d)
        logTrB_in = _safe_logm(TrB_in)
        logV = log_sigma - logTrB_in
        Gamma = _safe_expm(logK + np.kron(logU, np.eye(d)) + np.kron(np.eye(d), logV))
        Gamma = _make_hermitian(Gamma)
        err_r = norm(_partial_trace_out(Gamma, d) - rho, ord='fro')
        err_s = norm(_partial_trace_in(Gamma, d) - sigma, ord='fro')
        if verbose and (it % 20 == 0 or it < 5):
            print(f"[Sinkhorn iter {it}] err_r={err_r:.3e}, err_s={err_s:.3e}")
        if max(err_r, err_s) < tol:
            return Gamma, {"iter": it+1, "err":(err_r, err_s)}
    return Gamma, {"iter": max_iter, "err":(err_r, err_s)}

# ---------------- Gamma -> Kraus conditional ----------------
def gamma_to_kraus_from_conditionals(Gamma, d, eps_rank=1e-12):
    M = Gamma.reshape((d,d,d,d))
    p_in = np.array([np.real(M[i,:,i,:].trace()) for i in range(d)])
    kraus_list = []
    for i in range(d):
        S_i = M[i,:,i,:]
        S_i = 0.5*(S_i + S_i.conj().T)
        if p_in[i] < 1e-14:
            continue
        sigma_i = S_i / p_in[i]
        vals, vecs = eigh(sigma_i)
        for val_idx in range(len(vals)-1, -1, -1):
            val = vals[val_idx]
            if val > eps_rank:
                v = vecs[:, val_idx]
                mat = np.zeros((d,d), dtype=complex)
                mat[:, i] = np.sqrt(val) * v
                kraus_list.append(mat)
    # normalize approx TP
    S = sum(K.conj().T @ K for K in kraus_list)
    diagS = np.real(np.diag(S))
    for i in range(d):
        if diagS[i] > 1e-12:
            scale = 1.0/np.sqrt(diagS[i])
            for K in kraus_list:
                K[:, i] *= scale
    return kraus_list, p_in

# ---------------- Kraus -> Stinespring U ----------------
def kraus_to_stinespring_unitary(kraus_list):
    d_out, d_in = kraus_list[0].shape
    r = len(kraus_list)
    d_total = d_out * r
    V = np.zeros((d_total, d_in), dtype=complex)
    for a, K in enumerate(kraus_list):
        V[a*d_out:(a+1)*d_out, :] = K
    Q, R = np.linalg.qr(V)
    if d_total > d_in:
        u, s, vh = np.linalg.svd(Q, full_matrices=True)
        U = u
    else:
        U = Q
    return U, r

# ---------------- truncation & trotter helpers ----------------
def reduced_hamiltonian_on_subset(H, total_qubits, subset):
    full = H
    full_dim = 2**total_qubits
    R = np.zeros((2**len(subset), 2**len(subset)), dtype=complex)
    for a in range(full_dim):
        a_bits = [(a>>q)&1 for q in range(total_qubits)]
        a_sub = sum((a_bits[q] << i) for i,q in enumerate(subset))
        for b in range(full_dim):
            b_bits = [(b>>q)&1 for q in range(total_qubits)]
            b_sub = sum((b_bits[q] << i) for i,q in enumerate(subset))
            ok = True
            for q in range(total_qubits):
                if q not in subset and a_bits[q] != b_bits[q]:
                    ok = False; break
            if ok:
                R[a_sub, b_sub] += full[a, b]
    return R

def expand_on_subset_term(op_sub, subset, total_qubits):
    full_dim = 2**total_qubits
    Full = np.zeros((full_dim, full_dim), dtype=complex)
    for i in range(full_dim):
        for j in range(full_dim):
            ibits = [(i>>q)&1 for q in range(total_qubits)]
            jbits = [(j>>q)&1 for q in range(total_qubits)]
            i_sub = sum((ibits[q] << k) for k, q in enumerate(subset))
            j_sub = sum((jbits[q] << k) for k, q in enumerate(subset))
            Full[i,j] = op_sub[i_sub, j_sub]
    return Full

def truncate_generator_by_partial_traces(H, total_qubits, max_body=2):
    terms = []
    for q in range(total_qubits):
        H_q = reduced_hamiltonian_on_subset(H, total_qubits, [q])
        terms.append(expand_on_subset_term(H_q, [q], total_qubits))
    if max_body >= 2:
        for q1 in range(total_qubits):
            for q2 in range(q1+1, total_qubits):
                H_q = reduced_hamiltonian_on_subset(H, total_qubits, [q1, q2])
                terms.append(expand_on_subset_term(H_q, [q1, q2], total_qubits))
    return terms

def trotterize_from_terms(terms, steps=4):
    dt = 1.0/steps
    U = np.eye(terms[0].shape[0], dtype=complex)
    for _ in range(steps):
        for T in terms:
            U = expm(-1j * dt * T) @ U
    return U

# ---------------- Procrustes (alternate unitary mapping) ----------------
def procrustes_unitary(X_samples, Y_samples):
    # X, Y are lists of vectors (length d). Fit U s.t. U X ≈ Y in least squares sense.
    X = np.stack(X_samples, axis=1)  # d x m
    Y = np.stack(Y_samples, axis=1)
    # Solve U = Y X^H (X X^H)^{-1} but we want unitary -> use SVD of Y X^H
    M = Y @ X.conj().T
    U, s, Vh = np.linalg.svd(M)
    return U @ Vh

# ---------------- Main class with forward/backward ----------------
class QuantumOTSystem:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.d = 2**n_qubits
        self.Gamma = None
        self.kraus = None
        self.U = None
        self.anc_r = None
        self.local_sequence = None

    def _build_cost(self):
        d = self.d
        C = np.zeros((d*d, d*d), dtype=complex)
        for i in range(d):
            for j in range(d):
                C[i*d+j, i*d+j] = bin(i^j).count("1")
        return C

    def design_channel(self, sample_states, direction='backward', method='sinkhorn',
                       eps=0.25, tol=1e-6, max_iter=300, max_body=2, trotter_steps=6, steps_path=8,
                       verbose=False):
        """
        sample_states: list/array of target pure states (length d complex vectors)
        direction:
            'forward'  : target ensemble -> Haar (i.e. input marginal = sigma_emp, output marginal = I/d)
            'backward' : Haar -> target (i.e. input marginal = I/d, output marginal = sigma_emp)
        method: 'sinkhorn' (default) or 'procrustes' (faster approximate)
        returns dict with U_full, kraus_count, anc_dim, metrics, local_sequence
        """
        d = self.d
        # estimate sigma_emp
        sigma_emp = np.zeros((d,d), dtype=complex)
        for psi in sample_states:
            psi = np.asarray(psi, dtype=complex); psi = psi / np.linalg.norm(psi)
            sigma_emp += np.outer(psi, psi.conj())
        sigma_emp /= len(sample_states)

        if direction == 'forward':
            rho = sigma_emp.copy()
            sigma = np.eye(d, dtype=complex) / d
        else:
            rho = np.eye(d, dtype=complex) / d
            sigma = sigma_emp.copy()

        C = self._build_cost()

        if method == 'sinkhorn':
            Gamma, info = quantum_sinkhorn(rho, sigma, C, eps=eps, tol=tol, max_iter=max_iter, verbose=verbose)
        elif method == 'procrustes':
            # Procrustes requires paired samples: we'll pair sample_states with Haar samples
            haar_samples = [self.random_haar_state() for _ in range(len(sample_states))]
            U = procrustes_unitary(sample_states, haar_samples) if direction=='forward' else procrustes_unitary(haar_samples, sample_states)
            # create "trivial" Gamma by using Kraus from U (single-Kraus isometry)
            Gamma = None
            info = {"method":"procrustes"}
        else:
            raise ValueError("method unknown")

        self.Gamma = Gamma

        if method == 'sinkhorn':
            kraus_list, p_in = gamma_to_kraus_from_conditionals(Gamma, d)
            Ufull, r = kraus_to_stinespring_unitary(kraus_list)
        else:
            # from U directly (procrustes)
            Ufull = U
            # treat ancilla r=1
            r = 1
            kraus_list = [Ufull]  # not exact format, but placeholder

        self.kraus = kraus_list
        self.U = Ufull
        self.anc_r = r

        # Decompose to local sequence
        M = self.U @ np.eye(self.U.shape[0], dtype=complex).conj().T
        L = logm(M)
        total_qubits = int(np.log2(self.U.shape[0]))
        terms = truncate_generator_by_partial_traces(L, total_qubits, max_body=max_body)
        seq = []
        for alpha in np.linspace(0,1,steps_path+1):
            scaled = [alpha * T for T in terms]
            U_alpha = trotterize_from_terms(scaled, steps=trotter_steps)
            seq.append(U_alpha)
        self.local_sequence = seq

        # Metrics
        marg_err = (None, None)
        if Gamma is not None:
            marg_err = (norm(_partial_trace_out(Gamma,d) - rho), norm(_partial_trace_in(Gamma,d) - sigma))
        # unitary error between full and trotterized final
        U_trot = seq[-1]
        U_err = norm(self.U - U_trot, ord='fro')
        # avg fidelity between exact channel and approx (random sampling)
        avg_fid = self._avg_channel_fidelity(Ufull, U_trot, n_samples=200)

        results = {
            "Gamma_info": info,
            "kraus_count": len(kraus_list),
            "anc_dim": r,
            "marginal_error": marg_err,
            "unitary_fro_error": U_err,
            "avg_output_fidelity": avg_fid,
            "local_sequence": seq
        }
        return results

    def _avg_channel_fidelity(self, U_full, U_approx, n_samples=200):
        d_sys = 2**self.n_qubits
        r = self.anc_r if self.anc_r is not None else 1
        def apply(Umat, rho_in):
            anc0 = np.zeros((r,r), dtype=complex); anc0[0,0]=1.0
            rho_tot = np.kron(rho_in, anc0)
            out = Umat @ rho_tot @ Umat.conj().T
            rho_out = np.zeros((d_sys,d_sys), dtype=complex)
            for a in range(r):
                rows = slice(a*d_sys, (a+1)*d_sys)
                rho_out += out[rows, rows]
            return rho_out
        tot = 0.0
        for _ in range(n_samples):
            psi = np.random.randn(d_sys) + 1j*np.random.randn(d_sys)
            psi /= np.linalg.norm(psi)
            rho_in = np.outer(psi, psi.conj())
            rho_full = apply(U_full, rho_in)
            rho_approx = apply(U_approx, rho_in)
            S = sqrtm(rho_full)
            M = S @ rho_approx @ S
            val = np.real(np.trace(sqrtm(_make_hermitian(M))))
            tot += val
        return tot / n_samples

    def random_haar_state(self):
        z = (np.random.randn(self.d) + 1j*np.random.randn(self.d))
        return z / np.linalg.norm(z)

# ---------------- Demo usage ----------------
if __name__ == "__main__":
    np.random.seed(42)
    n_qubits = 2
    d = 2**n_qubits
    # create a relatively complex ensemble via random shallow circuit simulation (we simulate by random states)
    num_samples = 200
    complex_samples = []
    for _ in range(num_samples):
        # create correlated product of random single-qubit rotations followed by some entangling
        v = np.random.randn(d) + 1j*np.random.randn(d)
        v /= np.linalg.norm(v)
        complex_samples.append(v)

    qot = QuantumOTSystem(n_qubits=n_qubits)

    print("=== BACKWARD (Haar -> target) using quantum_sinkhorn ===")
    res_back = qot.design_channel(complex_samples, direction='backward',
                                  method='sinkhorn', eps=0.25, tol=1e-6, max_iter=300,
                                  max_body=2, trotter_steps=6, steps_path=8, verbose=True)
    print("Results backward:", {k:res_back[k] for k in ['kraus_count','anc_dim','marginal_error','unitary_fro_error','avg_output_fidelity']})

    print("\n=== FORWARD (target -> Haar) using quantum_sinkhorn ===")
    res_forw = qot.design_channel(complex_samples, direction='forward',
                                  method='sinkhorn', eps=0.25, tol=1e-6, max_iter=300,
                                  max_body=2, trotter_steps=6, steps_path=8, verbose=True)
    print("Results forward:", {k:res_forw[k] for k in ['kraus_count','anc_dim','marginal_error','unitary_fro_error','avg_output_fidelity']})
