"""
Network Stability Analysis Using a Linear Algebra Pipeline
UE24MA241B – Linear Algebra and Its Applications
PES University

Topic: Network Stability Analysis
- Real-world data: Node connectivity matrix of a power/communication network
- Pipeline: Matrix Representation → Simplification → Structure → Orthogonalization
            → Projection → Least Squares → Eigenvalue Analysis → Diagonalization
- Output: Stability score, dominant failure modes, predicted load distribution
"""

import numpy as np
from numpy.linalg import matrix_rank, svd, eig, inv, norm
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=4, suppress=True)

DIVIDER = "=" * 65

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

# ─────────────────────────────────────────────────────────────────
# STEP 1: REAL-WORLD DATA → MATRIX REPRESENTATION
# ─────────────────────────────────────────────────────────────────
section("STEP 1: Matrix Representation")

# Adjacency / Load matrix A: rows = nodes (routers/substations),
# cols = links. Entry A[i,j] = load on link j passing through node i.
# 6 nodes, 7 links in a simplified power/communication network.
A = np.array([
    [4, 2, 0, 1, 3, 0, 2],
    [1, 4, 3, 0, 2, 1, 0],
    [0, 1, 4, 2, 0, 3, 1],
    [3, 0, 1, 4, 1, 2, 3],
    [2, 3, 2, 1, 4, 0, 1],
    [1, 0, 3, 2, 1, 4, 2],
], dtype=float)

nodes = ["Node A", "Node B", "Node C", "Node D", "Node E", "Node F"]
links = ["L1","L2","L3","L4","L5","L6","L7"]

print("\nNetwork Load Matrix A (6 nodes × 7 links):")
print("       " + "  ".join(f"{l:>4}" for l in links))
for i, row in enumerate(A):
    print(f"  {nodes[i]}: " + "  ".join(f"{v:4.0f}" for v in row))
print("\nInterpretation: A[i,j] = traffic/load on link j routed through node i")
print(f"Matrix shape: {A.shape[0]} × {A.shape[1]}")

# ─────────────────────────────────────────────────────────────────
# STEP 2: MATRIX SIMPLIFICATION (Gaussian Elimination / RREF)
# ─────────────────────────────────────────────────────────────────
section("STEP 2: Matrix Simplification (Gaussian Elimination / RREF)")

def rref(M):
    """Compute Row Reduced Echelon Form."""
    mat = M.copy().astype(float)
    rows, cols = mat.shape
    pivot_row = 0
    pivots = []
    for col in range(cols):
        # Find pivot
        max_row = np.argmax(np.abs(mat[pivot_row:, col])) + pivot_row
        if abs(mat[max_row, col]) < 1e-10:
            continue
        mat[[pivot_row, max_row]] = mat[[max_row, pivot_row]]
        mat[pivot_row] /= mat[pivot_row, col]
        for r in range(rows):
            if r != pivot_row:
                mat[r] -= mat[r, col] * mat[pivot_row]
        pivots.append(col)
        pivot_row += 1
        if pivot_row >= rows:
            break
    return mat, pivots

rref_A, pivot_cols = rref(A)
r = matrix_rank(A)

print("\nRREF of Load Matrix A:")
print("       " + "  ".join(f"{l:>7}" for l in links))
for i, row in enumerate(rref_A):
    print(f"  {nodes[i]}: " + "  ".join(f"{v:7.4f}" for v in row))

print(f"\nPivot columns (independent links): {[links[c] for c in pivot_cols]}")
print(f"Number of pivot columns = Rank(A) = {r}")
print("Interpretation: {r} links carry truly independent traffic patterns.")

# ─────────────────────────────────────────────────────────────────
# STEP 3: STRUCTURE OF THE SPACE
# ─────────────────────────────────────────────────────────────────
section("STEP 3: Structure of the Space (Vector Spaces, Rank & Nullity)")

n_rows, n_cols = A.shape
nullity = n_cols - r

print(f"\n  Rank(A)    = {r}   → {r} independent link patterns")
print(f"  Nullity(A) = {nullity} → {nullity} redundant/dependent link(s)")
print(f"  Rank-Nullity Theorem: rank + nullity = {r} + {nullity} = {r+nullity} = n_cols ✓")
print(f"\n  Row Space    : {r}-dimensional  → node influence patterns")
print(f"  Column Space : {r}-dimensional  → reachable load distributions")
print(f"  Null Space   : {nullity}-dimensional → load combinations with zero net effect")

# Null space via SVD
U, S, Vt = svd(A)
null_space = Vt[r:].T
print(f"\n  Null space basis vectors (shape {null_space.shape}):")
for i, v in enumerate(null_space.T):
    print(f"    n{i+1}: [{', '.join(f'{x:.4f}' for x in v)}]")

# ─────────────────────────────────────────────────────────────────
# STEP 4: REMOVE REDUNDANCY (Linear Independence, Basis Selection)
# ─────────────────────────────────────────────────────────────────
section("STEP 4: Remove Redundancy (Linear Independence → Basis Selection)")

# Extract linearly independent columns (the pivot columns)
basis_cols = A[:, pivot_cols]
print(f"\n  Independent link basis — using pivot columns {[links[c] for c in pivot_cols]}:")
print(f"  Basis matrix B shape: {basis_cols.shape}")
print("\n  B =")
for i, row in enumerate(basis_cols):
    print(f"    {nodes[i]}: [{', '.join(f'{v:.1f}' for v in row)}]")

dep_links = [links[j] for j in range(n_cols) if j not in pivot_cols]
print(f"\n  Dependent (redundant) links removed: {dep_links}")
print("  These carry no new information — they are linear combinations of the basis.")

# ─────────────────────────────────────────────────────────────────
# STEP 5: ORTHOGONALIZATION (Gram–Schmidt)
# ─────────────────────────────────────────────────────────────────
section("STEP 5: Orthogonalization (Gram–Schmidt Process)")

def gram_schmidt(cols):
    """Gram-Schmidt orthonormalization."""
    Q = []
    for v in cols.T:
        u = v.copy().astype(float)
        for q in Q:
            u -= np.dot(u, q) * q
        if norm(u) > 1e-10:
            Q.append(u / norm(u))
    return np.column_stack(Q)

Q = gram_schmidt(basis_cols)
print("\n  Orthonormal basis Q (each column is a network mode):")
print(f"  Q shape: {Q.shape}")
for i, row in enumerate(Q):
    print(f"    {nodes[i]}: [{', '.join(f'{v:.4f}' for v in row)}]")

# Verify orthogonality
QtQ = Q.T @ Q
print("\n  Verification — Q^T Q (should be identity):")
for row in QtQ:
    print("   ", np.round(row, 4))
print("  ✓ Orthogonality confirmed" if np.allclose(QtQ, np.eye(Q.shape[1])) else "  ✗ Check failed")

# ─────────────────────────────────────────────────────────────────
# STEP 6: PROJECTION (Predict missing node load)
# ─────────────────────────────────────────────────────────────────
section("STEP 6: Projection (Orthogonal Projection onto Subspace)")

# Suppose Node G (new/partially observed) has known loads on links L1,L2,L3
# We project this partial observation into our network's column space
b_partial = np.array([2.5, 1.8, 3.1, 0, 0, 0])   # known for first 3 links, rest unknown

print("\n  New node observed with partial load vector b:")
print(f"    b = {b_partial}")
print("  (entries 0 = unobserved links)")

# Project b onto the column space of Q
proj = Q @ (Q.T @ b_partial)
print(f"\n  Projection onto network subspace:")
print(f"    proj_b = {np.round(proj, 4)}")

residual = b_partial - proj
print(f"\n  Residual (how far outside the subspace):")
print(f"    residual = {np.round(residual, 4)}")
print(f"    ||residual|| = {norm(residual):.4f}")
print(f"\n  Interpretation: proj_b gives the best consistent load estimate")
print(f"  within the known network behavior space.")

# ─────────────────────────────────────────────────────────────────
# STEP 7: LEAST SQUARES PREDICTION
# ─────────────────────────────────────────────────────────────────
section("STEP 7: Least Squares (Predicting Load Distribution)")

# Observed total throughput for each node (from monitoring sensors)
b_obs = np.array([12.0, 11.0, 11.5, 14.0, 13.0, 13.5])

print("\n  Observed total node throughputs (b_obs):")
for i, (n, v) in enumerate(zip(nodes, b_obs)):
    print(f"    {n}: {v}")

# Least squares: solve A x ≈ b_obs for link weights x
# x_hat = (A^T A)^{-1} A^T b
AtA = A.T @ A
Atb = A.T @ b_obs
x_hat = np.linalg.lstsq(A, b_obs, rcond=None)[0]

print(f"\n  Least Squares link weight estimates x̂ = (AᵀA)⁻¹Aᵀb:")
for i, (l, v) in enumerate(zip(links, x_hat)):
    print(f"    {l}: {v:.4f}")

b_pred = A @ x_hat
residuals = b_obs - b_pred
print(f"\n  Predicted node throughputs (A x̂):")
for i, (n, p, o) in enumerate(zip(nodes, b_pred, b_obs)):
    err = abs(p - o)
    print(f"    {n}: predicted={p:.4f}, observed={o:.1f}, error={err:.4f}")

print(f"\n  Least squares residual norm: {norm(residuals):.6f}")
print(f"  Interpretation: x̂ gives the optimal link weight configuration")
print(f"  that best explains all observed node throughputs simultaneously.")

# ─────────────────────────────────────────────────────────────────
# STEP 8: EIGENVALUE ANALYSIS (Pattern Discovery)
# ─────────────────────────────────────────────────────────────────
section("STEP 8: Pattern Discovery (Eigenvalues & Eigenvectors)")

# Build the covariance/Laplacian-style matrix from A
C = A.T @ A   # 7×7 symmetric matrix capturing link correlations
eigenvalues, eigenvectors = np.linalg.eigh(C)

# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

total_variance = np.sum(eigenvalues)
cumulative = np.cumsum(eigenvalues) / total_variance * 100

print("\n  Covariance matrix C = AᵀA (7×7 link correlation matrix)")
print(f"\n  Eigenvalue spectrum:")
print(f"  {'Mode':<6} {'Link':<6} {'Eigenvalue':>12} {'% Variance':>12} {'Cumulative':>12}")
print("  " + "-"*50)
for i in range(len(eigenvalues)):
    print(f"  {i+1:<6} {links[i]:<6} {eigenvalues[i]:12.4f} {eigenvalues[i]/total_variance*100:11.2f}% {cumulative[i]:11.2f}%")

dominant = np.sum(cumulative <= 95) + 1
print(f"\n  → First {dominant} modes capture ≥ 95% of network variance")
print(f"  → Dominant eigenvector (primary failure mode):")
print(f"    [{', '.join(f'{v:.4f}' for v in eigenvectors[:,0])}]")
print(f"  Interpretation: This mode shows which links move together")
print(f"  during peak load — the most critical congestion pattern.")

# ─────────────────────────────────────────────────────────────────
# STEP 9: DIAGONALIZATION (System Simplification)
# ─────────────────────────────────────────────────────────────────
section("STEP 9: System Simplification (Diagonalization)")

# C is symmetric → orthogonally diagonalizable: C = P D P^T
P = eigenvectors
D = np.diag(eigenvalues)

C_reconstructed = P @ D @ P.T
print("\n  C = PDP^T decomposition")
print(f"\n  P (eigenvector matrix) shape: {P.shape}")
print(f"  D (diagonal eigenvalue matrix) — top-left 3×3 block:")
print(f"    {D[:3,:3]}")
print(f"\n  Reconstruction check ||C - PDP^T|| = {norm(C - C_reconstructed):.8f} ≈ 0 ✓")

# Stability Score: ratio of dominant eigenvalue to trace
stability_index = 1 - (eigenvalues[0] / np.trace(C))
condition_number = eigenvalues[0] / max(eigenvalues[-1], 1e-10)

print(f"\n  ── NETWORK STABILITY METRICS ──────────────────────────────")
print(f"  Dominant eigenvalue   λ₁  = {eigenvalues[0]:.4f}")
print(f"  Trace (total energy)       = {np.trace(C):.4f}")
print(f"  Condition number      κ   = {condition_number:.4f}")
print(f"  Stability Index (SI)       = 1 - λ₁/trace = {stability_index:.4f}")

if stability_index > 0.7:
    status = "HIGHLY STABLE ✓"
elif stability_index > 0.4:
    status = "MODERATELY STABLE ⚠"
else:
    status = "UNSTABLE ✗"
print(f"  Network Status             = {status}")

if condition_number < 50:
    cond_msg = "Well-conditioned (resilient to perturbations)"
elif condition_number < 500:
    cond_msg = "Moderately conditioned"
else:
    cond_msg = "Ill-conditioned (sensitive to small load changes)"
print(f"  Conditioning               = {cond_msg}")

print(f"\n  ── FINAL APPLICATION OUTPUT ───────────────────────────────")
print(f"  1. Network has {r} independent load patterns (rank = {r})")
print(f"  2. {nullity} redundant link(s) can be eliminated without info loss")
print(f"  3. Least squares error on load prediction: {norm(residuals):.6f}")
print(f"  4. Top {dominant} eigenmodes capture ≥ 95% of network behavior")
print(f"  5. Stability Index = {stability_index:.4f} → {status}")
print(f"\n  Recommendation: Monitor links {[links[i] for i in np.argsort(np.abs(eigenvectors[:,0]))[-3:][::-1]]}")
print(f"  — these dominate the primary failure mode eigenvector.")
print(f"\n{DIVIDER}")
print("  Analysis complete.")
print(f"{DIVIDER}\n")
