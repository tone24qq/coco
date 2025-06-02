import numpy as np

def compute_distance_scores(blank_coords, missing_numbers, one_coord=None, K_coord=None, K_val=None):
    """
    Compute global distance correlation scores for each blank vs. each candidate number.
    Returns dist1_scores, distK_scores, center_scores as 2D numpy arrays.
    """
    num_blanks = len(blank_coords)
    num_missing = len(missing_numbers)
    if num_missing == 0 or num_blanks == 0:
        # No scores if no blanks or no missing numbers
        return None, None, None
    # Initialize score matrices
    dist1_scores = np.zeros((num_blanks, num_missing), dtype=np.float32)
    distK_scores = np.zeros((num_blanks, num_missing), dtype=np.float32)
    center_scores = np.zeros((num_blanks, num_missing), dtype=np.float32)
    # Compute Manhattan distances from global anchors if available
    if one_coord is not None:
        (i1, j1) = one_coord
        dist1 = np.abs(blank_coords[:,0] - i1) + np.abs(blank_coords[:,1] - j1)
        # Vectorize difference with (candidate - 1)
        cand_minus1 = np.array(missing_numbers, dtype=np.int32) - 1
        # Compute score = 1/(|dist - (cand-1)| + 1)
        diff1 = np.abs(dist1[:, None] - cand_minus1[None, :])
        dist1_scores = (1.0 / (diff1 + 1)).astype(np.float32)
    if K_coord is not None and K_val is not None:
        (iK, jK) = K_coord
        distK = np.abs(blank_coords[:,0] - iK) + np.abs(blank_coords[:,1] - jK)
        K_minus_cand = K_val - np.array(missing_numbers, dtype=np.int32)
        diffK = np.abs(distK[:, None] - K_minus_cand[None, :])
        distK_scores = (1.0 / (diffK + 1)).astype(np.float32)
    # Compute center proximity correlation (assumes mid-sequence near center)
    if K_val is None:
        K_val = max(missing_numbers) if missing_numbers else 0
    center_i = (blank_coords[:,0].max() + blank_coords[:,0].min()) / 2.0
    center_j = (blank_coords[:,1].max() + blank_coords[:,1].min()) / 2.0
    # Compute normalized radial distance for each blank
    # Use farthest corner distance for normalization
    max_i = blank_coords[:,0].max(); min_i = blank_coords[:,0].min()
    max_j = blank_coords[:,1].max(); min_j = blank_coords[:,1].min()
    corners = np.array([[min_i, min_j], [min_i, max_j], [max_i, min_j], [max_i, max_j]])
    corner_dists = np.sqrt(((corners - np.array([center_i, center_j]))**2).sum(axis=1))
    max_center_dist = corner_dists.max() if corner_dists.size > 0 else 1.0
    blank_radial = np.sqrt(((blank_coords - np.array([center_i, center_j]))**2).sum(axis=1)) / (max_center_dist + 1e-9)
    # Normalize candidate sequence position around mid-point
    mid_val = (K_val + 1) / 2.0
    cand_midness = np.abs(np.array(missing_numbers, dtype=np.float32) - mid_val) / ((K_val - 1)/2.0 if K_val > 1 else 1.0)
    # Compute center correlation score = 1 - |blank_radius - cand_midness|
    center_scores = (1.0 - np.abs(blank_radial[:, None] - cand_midness[None, :])).astype(np.float32)
    return dist1_scores, distK_scores, center_scores

def apply_eliminations(final_matrix, blank_coords, cluster_min_vals, cluster_max_vals, cluster_anchor_count,
                       one_coord=None, K_coord=None, K_val=None):
    """
    Apply elimination (score masking) based on anchor reachability and path constraints.
    Modifies final_matrix in place by setting certain entries to 0.
    """
    num_blanks, num_missing = final_matrix.shape
    # Map missing numbers for quick search
    # Ensure sorted missing list for binary search
    missing_nums = None
    if num_missing > 0:
        missing_nums = sorted((final_matrix.shape[1] * [0]) or [])
        # Actually, we rely on missing_numbers sorted input in new_module, so skip here.
        pass
    # Cluster anchor range elimination (GM10 and cluster anchor viability)
    for idx, (ci, cj) in enumerate(blank_coords):
        cid = cluster_anchor_count.size > 0 and cluster_anchor_count.dtype != float and cluster_anchor_count[0] == 0
        # Actually cluster_anchor_count is an array, not indicator; retrieve cluster id:
        # We assume region_map is accessible globally in new_module to get cluster id by blank coordinate.
        # We'll handle this elimination in new_module directly where region_map is available.
        pass

def identify_extraneous_by_path(blank_coords, anchor_values, anchor_positions, cluster_anchor_count, cluster_min_vals, cluster_max_vals):
    """
    Identify extraneous blanks via cluster path length feasibility (GM25).
    Returns a boolean mask of extraneous blanks.
    """
    num_blanks = len(blank_coords)
    extraneous_mask = np.zeros(num_blanks, dtype=bool)
    # Prepare a lookup for anchor positions by value
    anchor_pos_map = {val: tuple(pos) for val, pos in zip(anchor_values, anchor_positions)}
    for idx, (ci, cj) in enumerate(blank_coords):
        # Determine cluster context if two anchors
        # (In new_module, we can pass cluster id and anchor values for that cluster)
        pass
    return extraneous_mask