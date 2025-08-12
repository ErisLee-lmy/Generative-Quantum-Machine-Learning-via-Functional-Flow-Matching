"""
oqt_flow_end2end.py

End-to-end demo:
1) sample unknown source -> empirical sigma
2) quantum Sinkhorn -> Gamma (d^2 x d^2)
3) Gamma -> conditional outputs -> construct Kraus operators
4) Kraus -> Stinespring isometry -> extend to unitary U
5) Build unitary path U(t)
6) Truncate generator H to 1- and 2-body terms via partial traces and Pauli basis on small subsets
7) Trotterize exp(-i H) into local exponentials

Designed for small systems (2 qubits) — runs quickly on a laptop.
"""

import numpy as np
from scipy.linalg import expm, logm, eigh, norm
import math, time

# ---------------- utilities ----------------
def make_hermitian(A): return 0.5 * (A + A.conj().T)
def safe_logm(A, reg=1e-12):
    A = make_hermitian(A)
    return logm(A + reg * np.eye(A.shape[0]))
def safe_expm(A): return expm(A)

def partial_trace_out(mat, d):
    M = mat.reshape((d, d, d, d))
    return np.einsum('iaja->ij', M)

def partial_trace_in(mat, d):
    M = mat.reshape((d, d, d, d))
    return np.einsum('iaib->ab', M)

# ---------------- Quantum Sinkhorn (entropically regularized) ----------------
def quantum_sinkhorn(rho, sigma, C, eps=0.2, tol=1e-6, max_iter=500, verbose=False):
    d = rho.shape[0]
    C = make_hermitian(C)
    logK = -C / eps
    logU = np.zeros((d, d), dtype=complex)
    logV = np.zeros((d, d), dtype=complex)
    log_rho = safe_logm(rho)
    log_sigma = safe_logm(sigma)
    for it in range(max_iter):
        A = safe_expm(logK + np.kron(np.eye(d), logV))
        TrA_out = partial_trace_out(A, d)
        logTrA_out = safe_logm(TrA_out)
        logU = log_rho - logTrA_out

        B = safe_expm(logK + np.kron(logU, np.eye(d)))
        TrB_in = partial_trace_in(B, d)
        logTrB_in = safe_logm(TrB_in)
        logV = log_sigma - logTrB_in

        Gamma = safe_expm(logK + np.kron(logU, np.eye(d)) + np.kron(np.eye(d), logV))
        Gamma = make_hermitian(Gamma)
        err_r = norm(partial_trace_out(Gamma, d) - rho, ord='fro')
        err_s = norm(partial_trace_in(Gamma, d) - sigma, ord='fro')
        if verbose and (it % 10 == 0 or (it<10)):
            print(f"[iter {it}] err_r={err_r:.3e}, err_s={err_s:.3e}")
        if max(err_r, err_s) < tol:
            return Gamma, {"iter": it+1, "err": (err_r, err_s)}
    return Gamma, {"iter": max_iter, "err": (err_r, err_s)}

# ---------------- Gamma -> Kraus (conditional construction) ----------------
def gamma_to_kraus_from_conditionals(Gamma, d, eps_rank=1e-12):
    # Gamma shape (d^2, d^2) with ordering (in,out),(in,out)
    M = Gamma.reshape((d, d, d, d))  # (i,a,j,b)
    p_in = np.array([np.real(M[i,:,i,:].trace()) for i in range(d)])
    kraus_list = []
    for i in range(d):
        S_i = M[i,:,i,:]
        S_i = 0.5*(S_i + S_i.conj().T)
        if p_in[i] < 1e-14:
            continue
        sigma_i = S_i / p_in[i]
        vals, vecs = eigh(sigma_i)
        for k, val in enumerate(vals[::-1]):  # largest first
            if val > eps_rank:
                v = vecs[:, ::-1][:, k]
                mat = np.zeros((d, d), dtype=complex)
                mat[:, i] = np.sqrt(val) * v
                kraus_list.append(mat)
    # ensure TP approximately; renormalize per input if needed
    S = sum(K.conj().T @ K for K in kraus_list)
    diagS = np.real(np.diag(S))
    for i in range(d):
        if diagS[i] > 1e-12:
            scale = 1.0 / np.sqrt(diagS[i])
            for K in kraus_list:
                K[:, i] *= scale
    return kraus_list, p_in

# ---------------- Kraus -> Stinespring unitary ----------------
def kraus_to_stinespring_unitary(kraus_list):
    d_out, d_in = kraus_list[0].shape
    r = len(kraus_list)
    d_total = d_out * r
    V = np.zeros((d_total, d_in), dtype=complex)
    for a, K in enumerate(kraus_list):
        V[a*d_out:(a+1)*d_out, :] = K
    Q, R = np.linalg.qr(V)
    if d_total > d_in:
        # complete to full unitary via SVD-based completion
        u, s, vh = np.linalg.svd(Q, full_matrices=True)
        U = u
    else:
        U = Q
    return U, r

# ---------------- unitary path ----------------
def unitary_path(U0, U1, n_steps=8):
    M = U1 @ U0.conj().T
    L = logm(M)
    path = []
    for t in np.linspace(0, 1, n_steps+1):
        U_t = expm(t * L) @ U0
        path.append(U_t)
    return path

# ---------------- local truncation via partial traces ----------------
# Efficiently compute 1- and 2-body projected Hamiltonian terms by partial tracing
pauli_mats = [
    np.array([[1,0],[0,1]], dtype=complex),
    np.array([[0,1],[1,0]], dtype=complex),
    np.array([[0,-1j],[1j,0]], dtype=complex),
    np.array([[1,0],[0,-1]], dtype=complex)
]

def kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def reduced_hamiltonian_on_subset(H, total_qubits, subset):
    """
    Partial trace of H over qubits not in 'subset'.
    H acts on 2^n x 2^n. subset is list of qubit indices (0..n-1).
    Returns operator on 2^{len(subset)} space.
    """
    # reshape H to tensor indices and trace out complement
    d = 2**total_qubits
    H4 = H.reshape([2]*2*total_qubits)  # many indices
    # reorder so subset indices first (in,in,out,out)
    # easier approach: use vectorized partial trace by iterating basis -- but for small n it's fine
    # Here do by building computational basis projectors and performing partial trace directly
    keep = subset
    drop = [i for i in range(total_qubits) if i not in keep]
    dim_keep = 2**len(keep)
    R = np.zeros((dim_keep, dim_keep), dtype=complex)
    # iterate over all basis states
    for a in range(2**total_qubits):
        a_bits = [(a>>q)&1 for q in range(total_qubits)]
        a_keep = sum((a_bits[q]<<i) for i,q in enumerate(keep))
        for b in range(2**total_qubits):
            b_bits = [(b>>q)&1 for q in range(total_qubits)]
            b_keep = sum((b_bits[q]<<i) for i,q in enumerate(keep))
            # if drop bits equal, accumulate
            same = True
            for q in drop:
                if a_bits[q] != b_bits[q]:
                    same = False
                    break
            if same:
                R[a_keep, b_keep] += H[a, b]
    return R

def expand_on_subset_term(coeff, op_on_subset, subset, total_qubits):
    # embed op_on_subset into full space by kron with identities at other sites
    mats = []
    for q in range(total_qubits):
        if q in subset:
            # get index into subset order
            idx = subset.index(q)
            # we need op_on_subset written in 2^{|subset|} basis; we'll expand later
            pass
    # simpler: build full operator by tensor factors in natural order
    # we construct using bitwise: for each basis index, fill full matrix
    full_dim = 2**total_qubits
    Full = np.zeros((full_dim, full_dim), dtype=complex)
    dim_sub = 2**len(subset)
    # iterate basis of full and map to sub indices
    for i in range(full_dim):
        for j in range(full_dim):
            # compute sub indices
            ibits = [(i>>q)&1 for q in range(total_qubits)]
            jbits = [(j>>q)&1 for q in range(total_qubits)]
            i_sub = sum((ibits[q] << k) for k, q in enumerate(subset))
            j_sub = sum((jbits[q] << k) for k, q in enumerate(subset))
            Full[i, j] = coeff * op_on_subset[i_sub, j_sub]
    return Full

def truncate_generator_by_partial_traces(H, total_qubits, max_body=2):
    """
    Return list of local terms approximating H: compute 1- and 2-qubit reduced Hamiltonians
    and embed them back into full space.
    """
    terms = []
    # 1-body
    for q in range(total_qubits):
        H_q = reduced_hamiltonian_on_subset(H, total_qubits, [q])
        # expand back
        full = expand_on_subset_term(1.0, H_q, [q], total_qubits)
        terms.append(full)
    # 2-body
    if max_body >= 2:
        for q1 in range(total_qubits):
            for q2 in range(q1+1, total_qubits):
                H_q = reduced_hamiltonian_on_subset(H, total_qubits, [q1, q2])
                full = expand_on_subset_term(1.0, H_q, [q1, q2], total_qubits)
                terms.append(full)
    return terms

def trotterize_from_terms(terms, steps=4):
    dt = 1.0 / steps
    dim = terms[0].shape[0]
    U = np.eye(dim, dtype=complex)
    for _ in range(steps):
        for T in terms:
            U = expm(-1j * dt * T) @ U
    return U

# ---------------- Example run for 2 qubits ----------------
if __name__ == "__main__":
    np.random.seed(0)
    d = 4  # two qubits
    # (A) estimate sigma via samples from unknown source
    def sample_unknown():
        psi = np.random.randn(d) + 1j*np.random.randn(d)
        psi /= np.linalg.norm(psi)
        return psi
    N = 300
    sigma_emp = np.zeros((d,d), dtype=complex)
    for _ in range(N):
        psi = sample_unknown()
        sigma_emp += np.outer(psi, psi.conj())
    sigma_emp /= N

    # (B) choose input rho (we use diag probabilities from some prior)
    rho = np.diag([0.45, 0.25, 0.2, 0.1])

    # (C) cost C (diagonal in computational basis using Hamming distance)
    def hamming(i,j): return bin(i^j).count("1")
    C = np.zeros((d*d, d*d), dtype=complex)
    for i in range(d):
        for j in range(d):
            idx = i*d + j
            C[idx, idx] = hamming(i, j)

    print("Running quantum Sinkhorn (d=4)...")
    t0 = time.time()
    Gamma, info = quantum_sinkhorn(rho, sigma_emp, C, eps=0.25, tol=1e-6, max_iter=300, verbose=True)
    print("Sinkhorn done in", time.time()-t0, "s, iters:", info["iter"], "final errs:", info["err"])

    # (D) Gamma -> Kraus
    kraus_list, p_in = gamma_to_kraus_from_conditionals(Gamma, d)
    print("Kraus count:", len(kraus_list), "input probs:", np.round(p_in,3))

    # (E) Kraus -> Stinespring Unitary
    U, r = kraus_to_stinespring_unitary(kraus_list)
    print("Stinespring U dim:", U.shape, "ancilla r:", r)

    # (F) Unit path
    U0 = np.eye(U.shape[0], dtype=complex)
    path = unitary_path(U0, U, n_steps=8)
    print("Unitary path length:", len(path))

    # (G) compute generator H and truncate to 1- and 2-body and trotterize
    M = U @ U0.conj().T
    H = logm(M)  # generator in matrix log form
    n_total = int(np.log2(U.shape[0]))
    print("Total qubits for unitary:", n_total)
    terms = truncate_generator_by_partial_traces(H, n_total, max_body=2)
    print("Number of local terms from partial traces:", len(terms))
    U_trot = trotterize_from_terms(terms, steps=6)
    print("Trotterized unitary shape:", U_trot.shape)
    print("Trotter error (Frobenius) to U:", norm(U - U_trot))
