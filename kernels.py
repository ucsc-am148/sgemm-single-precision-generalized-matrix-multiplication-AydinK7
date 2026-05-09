"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE. 
    Be careful which one indexes the column.
    """
    # We are given a 1D array, and we have to reconstruct it to 2D
    # This kernel computes the dot product of one row and one col on one thread

    # Thread id gives you the threads position. In this we only have x since 1D
    # so we have one long row with many col. Gives which thread Im at like [0,1,2,...,n]
    tid = cuda.threadIdx.x 

    # We reconstruct the row by doing integer division, so threads 0-31 // 32
    # Will be row 0, and 32-63 // 32 will be row 1, etc.
    # Mod gives us the column position. Kinda like where am I inside the current row
    # tid of 32 gives row = 32 // 32 = 1, col = 32 % 32 = 0 => (1,0)
    row = tid // BLOCKSIZE
    column = tid % BLOCKSIZE

    # With the reconstructed 2D array, we get the global thread positions like normal
    # What block we're in * block size + thread
    x = cuda.blockIdx.x * BLOCKSIZE + row
    y = cuda.blockIdx.y * BLOCKSIZE + column

    # Check if thread is inside the matrix
    if x < M and y < N:
        tmp = float32(0.0)

        # K is shared dimension
        for i in range(K):
            # Go through row of A and col of B
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp

    

# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """
    # CUDA gives us a flattened 1D thread index
    tid = cuda.threadIdx.x

    local_row = tid // BN3
    local_col = tid % BN3

    # Calculate global row and column positions for this thread
    row = cuda.blockIdx.x * BM3 + local_row
    col = cuda.blockIdx.y * BN3 + local_col
    
    dot_sum = float32(0.0)

    # Create shared memory for the tiles
    A_shared = cuda.shared.array((BM3, BK3), float32)
    B_shared = cuda.shared.array((BK3, BN3), float32)
        
    # The computation that one thread needs to do can be split across
    # multiple tiles
    for tile_offset in range(0, K, BK3):
        a_col = tile_offset + local_col
        b_row = tile_offset + local_row


        # Make sure the thread is in the bounds of the matrix
        # Then add the A and B values at that location to shared memory
        if row < M and a_col < K:
            A_shared[local_row, local_col] = A[row, a_col]
        else:
            A_shared[local_row, local_col] = float32(0.0)

        if b_row < K and col < N:
            B_shared[local_row, local_col] = B[b_row, col]
        else:
            B_shared[local_row, local_col] = float32(0.0)

        # Make sure all threads for the block have loaded their data before
        # we start processing the dot product. Each thread loads one value into
        # shared memory, so we wait until all threads in the block are done before
        cuda.syncthreads()
        
        for i in range(BK3):
            dot_sum += A_shared[local_row, i] * B_shared[i, local_col]

        # Make sure no thread overwrites shared memory before others finish reading it
        cuda.syncthreads()

    # Write new positions into memory after checking if we are not an OOB thread
    if row < M and col < N:
        C[row, col] = dot_sum
        
    
# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
    
    tid = cuda.threadIdx.x

    local_thread_row_group = tid // BN4
    local_thread_col = tid % BN4

    # Calculate global row and column positions for this thread
    row_start = cuda.blockIdx.y * BM4 + local_thread_row_group * TM4 # * TM4 (8) for 8 rows
    col = cuda.blockIdx.x * BN4 + local_thread_col

    # Create shared memory for the tiles
    A_shared = cuda.shared.array((BM4, BK4), float32)
    B_shared = cuda.shared.array((BK4, BN4), float32)

    # We have 8 incdices for one thread
    # Create dot_sums array and intialize everything to zero    
    dot_sums = cuda.local.array(TM4, float32)
    
    for i in range(TM4):
        dot_sums[i] = float32(0.0)
    
    for tile_offset in range(0, K, BK4):
        # Load tile A into shared memory
        
        # We flatten this tile down to 1D into 512 elements so we have to use local row and col again
        # We cant reuse local_thread_row_group since it range from 0-8, but the A tile has 64 rows
        a_local_row = tid // BK4
        a_local_col = tid % BK4

        # We only had the global location output for C, not the A value being loaded
        a_global_row = cuda.blockIdx.y * BM4 + a_local_row
        a_global_col = tile_offset + a_local_col

        # Check if A is in bounds, if it is, put it in shared memory
        if a_global_row < M and a_global_col < K:
            A_shared[a_local_row, a_local_col] = A[a_global_row, a_global_col]
        else:
            A_shared[a_local_row, a_local_col] = float32(0.0)

        # Load tile B into shared memory
        b_local_row = tid // BN4
        b_local_col = tid % BN4

        b_global_row = tile_offset + b_local_row
        b_global_col = cuda.blockIdx.x * BN4 + b_local_col

        # Check if B is in bounds, if it is, put it in shared memory
        if b_global_row < K and b_global_col < N:
            B_shared[b_local_row, b_local_col] = B[b_global_row, b_global_col]
        else:
            B_shared[b_local_row, b_local_col] = float32(0.0)

        # Wait to make sure every thread has finished writing to shared memory before we start reading
        cuda.syncthreads()

        # Loops through the current K tile 
        for k_offset in range(BK4):
            # Take one B value for this threads column
            # This same B value is used for all 8 rows in this threads computation
            b_val = B_shared[k_offset, local_thread_col] 

            # Loop through the 8 rows owned by the thread
            for thread_row in range(TM4):
                # Find the actual row inside the 64 row A tile
                a_row = local_thread_row_group * TM4 + thread_row 
                
                dot_sums[thread_row] += A_shared[a_row, k_offset ] * b_val

        cuda.syncthreads()

    # Write the 8 results of the thread to C
    for i in range(TM4):
        row = row_start + i

        if row < M and col < N:
            C[row, col] = dot_sums[i]
        

# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    tid = cuda.threadIdx.x

    # Each thread computes an 8 x 8 tile this time, and the full block computes 128 x 128
    # BN5 // TN5 is the threads per row, which is 128/8 = 16 in this case
    thread_row_group = tid // (BN5 // TN5) # Also these are local again
    thread_col_group = tid % (BN5 // TN5)

    # Global starting row and col of this threads 8 x 8 output tile
    row_start = cuda.blockIdx.y * BM5 + thread_row_group * TM5
    col_start = cuda.blockIdx.x * BN5 + thread_col_group * TN5

    # Shared memory tiles
    A_shared = cuda.shared.array((BM5, BK5), float32)
    B_shared = cuda.shared.array((BK5, BN5), float32)

    # 8 x 8 local sums for this thread
    dot_sums = cuda.local.array((TM5, TN5), float32)

    # Small local arrays for temporary values during one k_offset later on
    reg_a = cuda.local.array(TM5, float32)
    reg_b = cuda.local.array(TN5, float32)

    # Initialize all 64 dot sums
    for i in range(TM5):
        for j in range(TN5):
            dot_sums[i, j] = float32(0.0)

    # Precompute columns outside of loop
    # Moved this from being computed inside the tile loop to outside to improve performance
    a_local_col = tid % BK5
    b_local_col = tid % BN5
    
    # Move across K in steps of 8 (BK5)
    for tile_offset in range(0, K, BK5):
        
        # Load A into shared memory, A is 128 x 8
        # There are 256 threads, so each thread loads 4 values
        for load_idx in range(tid, BM5 * BK5, cuda.blockDim.x):
            # Reconstruct row from load_idx, but column is precomputed this time
            a_local_row = load_idx // BK5
            # a_local_col = load_idx % BK5 # Computed outside of the loop now

            a_global_row = cuda.blockIdx.y * BM5 + a_local_row
            a_global_col = tile_offset + a_local_col

            # Check if A is in bounds, if it is, put it in shared memory
            if a_global_row < M and a_global_col < K:
                A_shared[a_local_row, a_local_col] = A[a_global_row, a_global_col]
            else:
                A_shared[a_local_row, a_local_col] = float32(0.0)

        # Load B into shared memory, B is 8 x 128
        # Each thread loads 4 values again like in A
        for load_idx in range(tid, BK5 * BN5, cuda.blockDim.x):
            b_local_row = load_idx // BN5
            # b_local_col = load_idx % BN5 # Same as in A, computed outside

            b_global_row = tile_offset + b_local_row
            b_global_col = cuda.blockIdx.x * BN5 + b_local_col

            # Check if B is in bounds, if it is, put it in shared memory
            if b_global_row < K and b_global_col < N:
                B_shared[b_local_row, b_local_col] = B[b_global_row, b_global_col]
            else:
                B_shared[b_local_row, b_local_col] = float32(0.0)

        # Wait to make sure every thread has finished writing to shared memory before we start reading
        cuda.syncthreads()

        # Loops over the 8 values in the current K chunk
        for k_offset in range(BK5):

            # Load this threads 8 A values
            for i in range(TM5):
                a_row = thread_row_group * TM5 + i
                reg_a[i] = A_shared[a_row, k_offset]

            # Load this threads 8 B values
            for j in range(TN5):
                b_col = thread_col_group * TN5 + j
                reg_b[j] = B_shared[k_offset, b_col]

            # Actual matrix multiplication using outer product update
            for i in range(TM5):
                for j in range(TN5):
                    dot_sums[i, j] += reg_a[i] * reg_b[j]                    

        cuda.syncthreads()

    # Write this threads 8 x 8 result tile back to C
    for i in range(TM5):
        for j in range(TN5):
            row = row_start + i
            col = col_start + j

            if row < M and col < N:
                C[row, col] = dot_sums[i, j]
                
# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]
