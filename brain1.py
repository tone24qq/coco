import numpy as np

# Neighbor offsets for 8-directional adjacency
_NEIGHBOR_OFFSETS = [(-1,-1), (-1,0), (-1,1),
                     (0,-1),           (0,1),
                     (1,-1),  (1,0),  (1,1)]

def analyze_neighbors(board_arr, blank_id_map, missing_index_map):
    """
    Analyze neighbor-based features (GM1-GM5, GM16) for masked cells.
    Returns:
        up_list, down_list, bridge_list: lists of (blank_index, cand_index, score)
        known_frac: array of baseline scores per blank (known neighbor fraction)
        blank_frac: array of baseline scores per blank (blank neighbor fraction)
        extraneous_mask: boolean array marking blanks to eliminate (consecutive known neighbors)
    """
    N, M = board_arr.shape
    # Identify known cell positions and values
    known_positions = np.argwhere(board_arr > 0)
    known_values = board_arr[board_arr > 0]
    # Prepare lists for contributions
    up_list = []
    down_list = []
    bridge_list = []
    # Prepare extraneous mask for blanks
    num_blanks = int((board_arr == -1).sum())
    extraneous_mask = np.zeros(num_blanks, dtype=bool)
    # Precompute neighbor known count for each cell using convolution (3x3 window)
    pad_board = np.pad((board_arr != -1).astype(int), pad_width=1, constant_values=0)
    # 3x3 sum centered on each cell gives count of known (non--1) in neighbors (including center if known)
    kernel = np.ones((3,3), dtype=int)
    neighbor_known_counts = (
        np.convolve(pad_board.flatten(), kernel.flatten(), mode='valid')
        .reshape(N, M)
    )
    # For a masked cell, if center was known, we subtract later; center known won't matter here since masked center = 0.
    # Identify masked cells with at least two known neighbors for detailed analysis
    mask_multi_known = (board_arr == -1) & (neighbor_known_counts >= 2)
    multi_positions = np.argwhere(mask_multi_known)
    # Process each masked cell with 2+ known neighbors
    for (ci, cj) in multi_positions:
        # Collect values of all known neighbors
        neighbor_vals = []
        for di, dj in _NEIGHBOR_OFFSETS:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < N and 0 <= nj < M and board_arr[ni, nj] != -1:
                neighbor_vals.append(board_arr[ni, nj])
        neighbor_vals.sort()
        # Check for bridging scenario (diff=2 between known neighbors)
        for k in range(len(neighbor_vals) - 1):
            if neighbor_vals[k+1] - neighbor_vals[k] == 2:
                missing_val = neighbor_vals[k] + 1
                if missing_val in missing_index_map:
                    blank_idx = blank_id_map[ci, cj]
                    cand_idx = missing_index_map[missing_val]
                    bridge_list.append((blank_idx, cand_idx, 1.0))
        # Check for consecutive known neighbors (diff=1 -> cell not on path)
        for k in range(len(neighbor_vals) - 1):
            if neighbor_vals[k+1] - neighbor_vals[k] == 1:
                # Mark this blank cell for elimination
                extraneous_mask[blank_id_map[ci, cj]] = True
                break
    # Process adjacency for all known cells (GM1 & GM2)
    for (ki, kj), kv in zip(known_positions, known_values):
        # If kv+1 is missing, all adjacent masked cells get score for being kv+1
        next_val = kv + 1
        if next_val in missing_index_map:
            cand_idx = missing_index_map[next_val]
            for di, dj in _NEIGHBOR_OFFSETS:
                ni, nj = ki + di, kj + dj
                if 0 <= ni < N and 0 <= nj < M and board_arr[ni, nj] == -1:
                    blank_idx = blank_id_map[ni, nj]
                    up_list.append((blank_idx, cand_idx, 1.0))
        # If kv-1 is missing, adjacent masked cells get score for kv-1
        prev_val = kv - 1
        if prev_val in missing_index_map:
            cand_idx = missing_index_map[prev_val]
            for di, dj in _NEIGHBOR_OFFSETS:
                ni, nj = ki + di, kj + dj
                if 0 <= ni < N and 0 <= nj < M and board_arr[ni, nj] == -1:
                    blank_idx = blank_id_map[ni, nj]
                    down_list.append((blank_idx, cand_idx, 1.0))
    # Compute neighbor fraction baselines (GM4 & GM5) for each blank
    # Count known neighbors and blank neighbors for each masked cell
    # neighbor_known_counts already has counts of known (including center if known, which for blank is 0).
    known_counts_masked = neighbor_known_counts[board_arr == -1]
    # Count total neighbors possible (8 minus out-of-bounds edges)
    # We determine out-of-bound count for each blank cell
    blank_positions = np.argwhere(board_arr == -1)
    out_counts = np.zeros(len(blank_positions), dtype=int)
    for idx, (ci, cj) in enumerate(blank_positions):
        out = 0
        if ci == 0: out += 3
        if ci == N-1: out += 3
        if cj == 0: out += 3
        if cj == M-1: out += 3
        if (ci == 0 or ci == N-1) and (cj == 0 or cj == M-1):
            out -= 1  # corner counted one extra
        out_counts[idx] = out
    # Known fraction = known_neighbors / (8 - out_of_bounds)
    total_neighbors = 8 - out_counts
    known_frac = np.divide(known_counts_masked, total_neighbors, out=np.zeros_like(known_counts_masked, float), where=total_neighbors>0)
    # Blank fraction = blank_neighbors / (8 - out_of_bounds)
    # blank_neighbors = total_neighbors - known_neighbors
    blank_neighbors = total_neighbors - known_counts_masked
    blank_frac = np.divide(blank_neighbors, total_neighbors, out=np.zeros_like(blank_neighbors, float), where=total_neighbors>0)
    # Invert blank fraction for scoring (fewer blank neighbors -> higher score)
    isolation_score = 1.0 - blank_frac
    return up_list, down_list, bridge_list, known_frac.astype(float), isolation_score.astype(float), extraneous_mask