import numpy as np
from brain1 import analyze_neighbors
from brain2 import analyze_clusters
from brain3 import compute_distance_scores

def score_full_board(board):
    """
    Compute normalized scores for each masked cell being each missing number.
    Returns a list of prediction dictionaries sorted by number.
    """
    board_arr = np.array(board, dtype=int)
    N, M = board_arr.shape
    # Identify all masked cell coordinates
    blank_coords = np.argwhere(board_arr == -1)
    num_blanks = len(blank_coords)
    if num_blanks == 0:
        return []  # nothing to predict
    # Map each masked cell to a blank index
    blank_id_map = -np.ones(board_arr.shape, dtype=int)
    for idx, (bi, bj) in enumerate(blank_coords):
        blank_id_map[bi, bj] = idx
    # Determine the set of missing numbers
    present_nums = set(board_arr.flatten()) - {-1}
    if present_nums:
        max_present = max(present_nums)
    else:
        max_present = 0
    # Estimate K (the highest number that was originally on the board)
    total_cells = N * M
    # Assume all numbers 1..K were placed; if some cells never get a number, K could be less than total_cells
    K_val = max(max_present, len(present_nums) + (board_arr == -1).sum())
    # Build missing numbers list
    all_nums = set(range(1, K_val+1))
    missing_numbers = sorted(list(all_nums - present_nums))
    num_missing = len(missing_numbers)
    # Prepare result matrix for scores
    final_scores = np.zeros((num_blanks, num_missing), dtype=np.float32)
    # ------------------ Stage 1: Neighbor-based analysis (Brain1) ------------------
    (up_list, down_list, bridge_list,
     known_frac, isolation_frac, extr_mask_neighbors) = analyze_neighbors(board_arr, blank_id_map,
                                                                          {val: i for i, val in enumerate(missing_numbers)})
    # Apply neighbor contributions
    for (b_idx, c_idx, score) in up_list:
        final_scores[b_idx, c_idx] += score
    for (b_idx, c_idx, score) in down_list:
        final_scores[b_idx, c_idx] += score
    for (b_idx, c_idx, score) in bridge_list:
        final_scores[b_idx, c_idx] += score
    # Add neighbor baseline scores (known neighbor fraction and blank neighbor inverse fraction)
    if num_missing > 0:
        final_scores += known_frac[:, None].astype(np.float32)
        final_scores += isolation_frac[:, None].astype(np.float32)
    # ------------------ Stage 2: Cluster-based analysis (Brain2) ------------------
    (cluster_ratio_scores, cluster_size_scores, cluster_slack_scores, cluster_density_scores,
     cluster_min_vals, cluster_max_vals, cluster_anchor_count, region_map) = analyze_clusters(board_arr, blank_id_map,
                                                                                              missing_numbers,
                                                                                              {val: i for i, val in enumerate(missing_numbers)})
    # Add cluster baseline scores
    if num_missing > 0:
        final_scores += cluster_ratio_scores[:, None].astype(np.float32)
        final_scores += cluster_size_scores[:, None].astype(np.float32)
        final_scores += cluster_slack_scores[:, None].astype(np.float32)
        final_scores += cluster_density_scores[:, None].astype(np.float32)
    # ------------------ Stage 3: Global analysis (Brain3) ------------------
    # Determine global anchor coordinates if known
    one_coord = None; K_coord = None
    if 1 not in missing_numbers:
        # Find position of known 1
        loc = np.argwhere(board_arr == 1)
        if loc.size > 0:
            one_coord = tuple(loc[0])
    if K_val not in missing_numbers:
        loc = np.argwhere(board_arr == K_val)
        if loc.size > 0:
            K_coord = tuple(loc[0])
    # Compute global distance-based score matrices
    dist1_scores, distK_scores, center_scores = compute_distance_scores(blank_coords, missing_numbers,
                                                                       one_coord, K_coord, K_val)
    # Add global correlation scores
    if dist1_scores is not None:
        final_scores += dist1_scores
    if distK_scores is not None:
        final_scores += distK_scores
    if center_scores is not None:
        final_scores += center_scores
    # ------------------ Stage 4: Eliminations (mask out impossible assignments) ------------------
    # Eliminate extraneous blanks from neighbor or path analysis
    extraneous_mask = extr_mask_neighbors.copy()
    # Determine extraneous blanks via cluster path feasibility (GM25)
    # If a cluster has two anchors, check each blank in it
    extr_mask_path = np.zeros(num_blanks, dtype=bool)
    two_anchor_clusters = np.where(cluster_anchor_count == 2)[0]
    if two_anchor_clusters.size > 0:
        # Map anchor values to coordinates for clusters
        anchor_val_to_coord = {}
        for cid in two_anchor_clusters:
            # find coords of anchors by value
            a_val = int(cluster_min_vals[cid]); b_val = int(cluster_max_vals[cid])
            a_pos = np.argwhere(board_arr == a_val)
            b_pos = np.argwhere(board_arr == b_val)
            if a_pos.size and b_pos.size:
                anchor_val_to_coord[a_val] = tuple(a_pos[0])
                anchor_val_to_coord[b_val] = tuple(b_pos[0])
        for idx, (bi, bj) in enumerate(blank_coords):
            cid = region_map[bi, bj]
            if cluster_anchor_count[cid] == 2:
                a_val = int(cluster_min_vals[cid]); b_val = int(cluster_max_vals[cid])
                if a_val in anchor_val_to_coord and b_val in anchor_val_to_coord:
                    (ai, aj) = anchor_val_to_coord[a_val]
                    (bi2, bj2) = anchor_val_to_coord[b_val]
                    # Manhattan distance from cell to each anchor
                    d1 = abs(bi - ai) + abs(bj - aj)
                    d2 = abs(bi - bi2) + abs(bj - bj2)
                    if d1 + d2 > (b_val - a_val):
                        extr_mask_path[idx] = True
    extraneous_mask |= extr_mask_path
    # Apply elimination for extraneous blanks (set all scores to 0)
    for idx in range(num_blanks):
        if extraneous_mask[idx]:
            final_scores[idx, :] = 0.0
    # Anchor range viability (no candidate outside [A+1, B-1] for cluster anchors A, B)
    for idx, (bi, bj) in enumerate(blank_coords):
        cid = region_map[bi, bj]
        if cluster_anchor_count[cid] == 2:
            A = int(cluster_min_vals[cid]); B = int(cluster_max_vals[cid])
            # eliminate candidates <= A or >= B
            if num_missing > 0:
                # find insertion indices (binary search) in missing_numbers
                low_idx = np.searchsorted(missing_numbers, A+1)
                high_idx = np.searchsorted(missing_numbers, B)
                final_scores[idx, :low_idx] = 0.0
                final_scores[idx, high_idx:] = 0.0
        elif cluster_anchor_count[cid] == 1:
            A = int(cluster_min_vals[cid]); B = int(cluster_max_vals[cid])
            if A == 1 and B <= K_val:
                # anchor at high side, eliminate >= B
                high_idx = np.searchsorted(missing_numbers, B)
                final_scores[idx, high_idx:] = 0.0
            elif B == K_val and A >= 1:
                # anchor at low side, eliminate <= A
                low_idx = np.searchsorted(missing_numbers, A+1)
                final_scores[idx, :low_idx] = 0.0
    # Global distance viability (1 and K reachability)
    if one_coord is not None:
        (i1, j1) = one_coord
        for idx, (bi, bj) in enumerate(blank_coords):
            # Chebyshev distance (max of row,col diffs) for 8-dir moves or Manhattan for 4-dir; we use Manhattan:
            dist1 = abs(bi - i1) + abs(bj - j1)
            # Candidate must be at least dist1+1
            low_val = dist1 + 1
            if num_missing > 0:
                low_idx = np.searchsorted(missing_numbers, low_val)
                final_scores[idx, :low_idx] = 0.0
    if K_coord is not None:
        (iK, jK) = K_coord
        for idx, (bi, bj) in enumerate(blank_coords):
            distK = abs(bi - iK) + abs(bj - jK)
            # Candidate must be at most K - distK
            high_val = K_val - distK
            if num_missing > 0:
                high_idx = np.searchsorted(missing_numbers, high_val + 1)
                final_scores[idx, high_idx:] = 0.0
    # ------------------ Stage 5: Normalize and prepare output ------------------
    # Normalize scores for each missing number
    col_sums = final_scores.sum(axis=0, keepdims=True)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1.0
    prob_matrix = final_scores / col_sums
    # Compile predictions: choose the most likely cell for each missing number
    predictions = []
    for j, num in enumerate(missing_numbers):
        # identify best blank for number j
        col = prob_matrix[:, j]
        best_idx = int(np.argmax(col))
        prob = float(col[best_idx])
        if prob <= 0.0:
            # skip numbers that couldn't be placed (unlikely if puzzle is consistent)
            continue
        cell_i, cell_j = blank_coords[best_idx]
        predictions.append({
            "number": int(num),
            "row": int(cell_i),
            "col": int(cell_j),
            "probability": prob
        })
    # Sort predictions by number for readability
    predictions.sort(key=lambda x: x["number"])
    return predictions