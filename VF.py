import numpy as np
from numpy.linalg import eigvals, lstsq
from concurrent.futures import ProcessPoolExecutor
import time


def compute_A_chunk(args):
    """
    Worker function executed in a separate process.
    Computes a portion (chunk) of the matrix A for a subset of frequency points.
    
    Args:
        args: tuple (s_chunk, poles, modes)
    
    Returns:
        np.ndarray: Chunk of matrix A (complex)
    """
    s_chunk, poles, modes = args
    chunk_size = len(s_chunk)
    n_cols = len(modes) + 2  # columns for residues + d + h*s

    A_chunk = np.zeros((chunk_size, n_cols), dtype=np.complex128)
    A_chunk[:, -2] = 1.0               # constant term column (d)
    A_chunk[:, -1] = s_chunk           # linear term column (h * s)

    for col_idx, (mode, p) in enumerate(modes):
        if mode == 'real':
            A_chunk[:, col_idx] = 1.0 / (s_chunk - p)
        elif mode == 'conj_real':
            pc = np.conj(p)
            A_chunk[:, col_idx] = 1.0 / (s_chunk - p) + 1.0 / (s_chunk - pc)
        elif mode == 'conj_imag':
            pc = np.conj(p)
            A_chunk[:, col_idx] = 1j * (1.0 / (s_chunk - p) - 1.0 / (s_chunk - pc))

    return A_chunk


def build_A(s, poles, modes, n_processes=4):
    """
    Constructs matrix A in parallel by splitting the frequency vector s
    into chunks and processing them using multiple processes.
    """
    Ns = len(s)
    if Ns < 400:
        n_processes = 1

    chunk_size = (Ns // n_processes) + 1
    chunks = [s[i:i + chunk_size] for i in range(0, Ns, chunk_size)]

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [
            executor.submit(compute_A_chunk, (chunk, poles, modes))
            for chunk in chunks
        ]
        chunk_results = [future.result() for future in futures]

    return np.vstack(chunk_results)


def detect_modes(poles):
    """
    Classifies poles into real poles and complex conjugate pairs.
    
    Returns:
        list of tuples: [('real', pole), ('conj_real', pole), ('conj_imag', pole), ...]
    """
    modes = []
    i = 0
    N = len(poles)
    while i < N:
        p = poles[i]
        if np.isclose(np.imag(p), 0.0):
            modes.append(('real', p))
            i += 1
        else:
            if i + 1 < N and np.isclose(p, np.conj(poles[i + 1])):
                modes.append(('conj_real', p))
                modes.append(('conj_imag', p))
                i += 2
            else:
                raise ValueError(f"Complex pole without conjugate pair at index {i}: {p}")
    return modes


def generate_initial_poles_from_freqs(freqs_hz, damping_factor=0.02):
    """
    Generate initial complex conjugate pole pairs from a list of approximate
    resonant frequencies (in Hz).
    
    Args:
        freqs_hz: list or array of frequencies in Hz
        damping_factor: multiplier for real part (negative damping)
    
    Returns:
        np.ndarray: array of complex initial poles
    """
    poles = []
    for f in freqs_hz:
        omega = 2 * np.pi * f
        real_part = -damping_factor * omega
        poles.append(complex(real_part, +omega))
        poles.append(complex(real_part, -omega))
    return np.array(poles)


def vectfit_step(f, s, poles, n_processes=4):
    """Performs one iteration of the Vector Fitting algorithm."""
    modes = detect_modes(poles)

    A_complex = build_A(s, poles, modes, n_processes)
    A_real = np.vstack([np.real(A_complex), np.imag(A_complex)])
    f_real = np.concatenate([np.real(f), np.imag(f)])

    N = len(poles)
    sigma = np.zeros((len(s), N), dtype=complex)

    for i, (mode, p) in enumerate(modes):
        if mode == 'real':
            sigma[:, i] = 1.0 / (s - p)
        elif mode == 'conj_real':
            pc = np.conj(p)
            sigma[:, i] = 1.0 / (s - p) + 1.0 / (s - pc)
        elif mode == 'conj_imag':
            pc = np.conj(p)
            sigma[:, i] = 1j * (1.0 / (s - p) - 1.0 / (s - pc))

    sigma_real = np.vstack([np.real(sigma), np.imag(sigma)])
    sigma_f_real = sigma_real * f_real[:, np.newaxis]

    A_aug = np.hstack([A_real, -sigma_f_real])

    x, *_ = lstsq(A_aug, f_real, rcond=None)
    sigma_coeff = x[-N:]

    A_mat = np.diag(poles)
    b = np.ones(N, dtype=complex)
    H = A_mat - np.outer(b, sigma_coeff)
    H = np.real(H)

    new_poles = eigvals(H)

    # Basic stability enforcement: flip unstable poles to left half-plane
    real_parts = np.real(new_poles)
    unstable = real_parts > 0
    new_poles[unstable] = -real_parts[unstable] + 1j * np.imag(new_poles[unstable])

    return np.sort_complex(new_poles)


def vectfit_auto_parallel(
    f, s,
    n_poles=10,
    n_iter=8,
    n_processes=4,
    initial_damping=0.02,
    initial_resonant_freqs=None,
):
    """
    Main Vector Fitting routine with parallel matrix construction.
    
    Args:
        f:                      function values at points s (complex)
        s:                      complex frequency points (usually jω)
        n_poles:                desired number of poles (used only if no initial_resonant_freqs)
        n_iter:                 number of refinement iterations
        n_processes:            number of CPU processes for parallel parts
        initial_damping:        damping factor for generated initial poles
        initial_resonant_freqs: list/array of approximate resonant frequencies in Hz
    
    Returns:
        poles, residues, d, h
    """
    if initial_resonant_freqs is not None and len(initial_resonant_freqs) > 0:
        poles = generate_initial_poles_from_freqs(initial_resonant_freqs, initial_damping)
        print(f"Generated {len(poles)} initial poles from {len(initial_resonant_freqs)} user-provided resonant frequencies")
    else:
        # Default: logarithmically spaced poles along imaginary axis
        w = np.sort(np.abs(np.imag(s)))
        if len(w) < 3:
            raise ValueError("Too few frequency points (need at least 3)")
        pole_locs = np.linspace(w[0], w[-1], n_poles + 2)[1:-1]
        poles_list = []
        for loc in pole_locs:
            poles_list.append(complex(-initial_damping * loc, +loc))
            poles_list.append(complex(-initial_damping * loc, -loc))
        poles = np.array(poles_list[:2 * n_poles])
        print(f"Generated {len(poles)} logarithmically spaced initial poles")

    print(f"Starting Vector Fitting with {len(poles)} initial poles")

    # Main iteration loop
    for it in range(1, n_iter + 1):
        print(f"Iteration {it}/{n_iter}")
        poles = vectfit_step(f, s, poles, n_processes=n_processes)

    # Final residue / coefficient fitting
    modes = detect_modes(poles)
    A_complex = build_A(s, poles, modes, n_processes)
    A_real = np.vstack([np.real(A_complex), np.imag(A_complex)])
    f_real = np.concatenate([np.real(f), np.imag(f)])

    x, *_ = lstsq(A_real, f_real, rcond=None)

    N = len(poles)
    residues = x[:N]
    d = float(np.real(x[N]))
    h = float(np.real(x[N + 1])) if len(x) > N + 1 else 0.0

    return poles, residues, d, h


# ────────────────────────────────────────────────
#                    Interactive Main
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Interactive Vector Fitting Tool ===\n")

    # Generate synthetic test data (you can replace this with your own data loading)
    print("Generating synthetic test data...")
    freq = np.logspace(1, 6, 1800)              # frequency vector [Hz]
    s = 1j * freq                               # s = jω
    f = (
        1 / (s +  80  + 250j) +
        1 / (s + 350 - 180j) +
        0.008 * s + 4.2
    ) + 0.004 * (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
    print(f"Data ready: {len(s)} frequency points\n")

    # ── User inputs ──────────────────────────────────────────────────────────────

    freq_input = input("Enter approximate resonant frequencies in Hz (comma separated, e.g. 40,175,500)\n"
                       "or press Enter to use automatic log-spaced poles: ").strip()

    if freq_input:
        try:
            user_freqs = [float(x.strip()) for x in freq_input.split(',')]
            print(f" → Using frequencies: {user_freqs}")
        except Exception as e:
            print(f"Invalid input: {e}")
            print(" → Falling back to automatic poles")
            user_freqs = None
    else:
        user_freqs = None
        print(" → Using automatic log-spaced poles")

    n_iter_str = input("Number of iterations [default: 9]: ").strip()
    n_iter = int(n_iter_str) if n_iter_str else 9

    damping_str = input("Initial damping factor [default: 0.015]: ").strip()
    initial_damping = float(damping_str) if damping_str else 0.015

    proc_str = input("Number of parallel processes [default: 4]: ").strip()
    n_processes = int(proc_str) if proc_str else 4

    print("\n" + "="*50)
    print("Starting Vector Fitting with the following settings:")
    print(f"  Iterations       : {n_iter}")
    print(f"  Damping factor   : {initial_damping}")
    print(f"  Processes        : {n_processes}")
    if user_freqs:
        print(f"  User frequencies : {user_freqs}")
    print("="*50 + "\n")

    t_start = time.perf_counter()

    poles, residues, d, h = vectfit_auto_parallel(
        f=f,
        s=s,
        n_iter=n_iter,
        n_processes=n_processes,
        initial_damping=initial_damping,
        initial_resonant_freqs=user_freqs,
    )

    elapsed = time.perf_counter() - t_start
    print(f"\nFitting completed in {elapsed:.2f} seconds")

    # Evaluate fit quality
    f_fit = d + h * s
    for p, r in zip(poles, residues):
        f_fit += r / (s - p)

    rel_error = np.linalg.norm(f - f_fit) / np.linalg.norm(f)
    print(f"Relative error: {rel_error:.2e}")

    print("\nFinal poles (real part / imag part):")
    for p in poles:
        print(f"  {p.real:12.4e}   {p.imag:12.4e}j")

    print("\nDone.")
