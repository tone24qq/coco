import numpy as np
from scipy.ndimage import label, find_objects

def analyze_clusters(board_arr, blank_id_map, missing_numbers, missing_index_map):
    """
    Analyze connected masked clusters and their anchor constraints (GM6-GM9, GM10).
    Returns:
        cluster_ratio: array of score per blank for fill ratio
        cluster_size_score: array of score per blank for cluster size (inverse)
        cluster_slack_flag: array of score per blank for no-slack clusters
        cluster_density_score: array of score per blank for cluster density
        cluster_min_vals, cluster_max_vals: arrays of cluster min/max anchor values
        cluster_anchor_count: array of anchor counts per cluster
        region_map: labeled regions map of masked cells
    """
    N, M = board_arr.shape
    # Label connected masked regions (8-directional connectivity)
    mask = (board_arr == -1).astype(int)
    structure = np.ones((3,3), dtype=int)  # 8-neighbor connectivity
    region_map, num_clusters = label(mask, structure=structure)
    # Compute cluster sizes (number of blanks per cluster)
    # We use bincount on region_map flattened (ignoring 0 label)
    flat_labels = region_map.flatten()
    cluster_sizes = np.bincount(flat_labels, minlength=num_clusters+1)
    # Initialize arrays for cluster anchor information
    cluster_min_vals = np.full(num_clusters+1, np.inf, dtype=float)
    cluster_max_vals = np.full(num_clusters+1, -np.inf, dtype=float)
    cluster_anchor_count = np.zeros(num_clusters+1, dtype=int)
    # Scan each known cell and update neighboring cluster anchor values
    known_mask = (board_arr > 0)
    known_coords = np.argwhere(known_mask)
    for (ki, kj) in known_coords:
        kv = board_arr[ki, kj]
        # Check all adjacent masked cells to associate this known value with that cluster
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = ki + di, kj + dj
                if 0 <= ni < N and 0 <= nj < M and board_arr[ni, nj] == -1:
                    cid = region_map[ni, nj]
                    # Update cluster's min/max known neighbor values
                    if kv < cluster_min_vals[cid]:
                        cluster_min_vals[cid] = kv
                    if kv > cluster_max_vals[cid]:
                        cluster_max_vals[cid] = kv
    # Replace inf/-inf placeholders with appropriate range endpoints (open clusters)
    total_numbers = max(missing_numbers) if missing_numbers else 0
    K_val = total_numbers if total_numbers > 0 else board_arr.size  # estimated K
    # Use K_val as N×M or largest number present if no missing beyond
    if total_numbers < board_arr.size:
        # If some cells never filled (K < N*M), K_val represents the highest number originally (first K values)
        K_val = max(K_val, board_arr.size)
    for cid in range(1, num_clusters+1):
        if cluster_min_vals[cid] == np.inf and cluster_max_vals[cid] == -np.inf:
            # No known neighbors at all (no anchors)
            cluster_min_vals[cid] = 1.0
            cluster_max_vals[cid] = float(K_val)
            cluster_anchor_count[cid] = 0
        elif cluster_min_vals[cid] == np.inf:
            # No low anchor, high anchor exists
            cluster_min_vals[cid] = 1.0
            cluster_anchor_count[cid] = 1
        elif cluster_max_vals[cid] == -np.inf:
            # No high anchor, low anchor exists
            cluster_max_vals[cid] = float(K_val)
            cluster_anchor_count[cid] = 1
        else:
            cluster_anchor_count[cid] = 2
    # Calculate cluster fill ratio and density metrics
    cluster_ratio = np.zeros(num_clusters+1, dtype=float)
    cluster_slack_flag = np.zeros(num_clusters+1, dtype=float)
    cluster_density = np.zeros(num_clusters+1, dtype=float)
    # Use find_objects to get cluster bounding boxes for density
    cluster_slices = find_objects(region_map)
    for cid in range(1, num_clusters+1):
        size = cluster_sizes[cid]
        # Determine missing numbers count the cluster must fill
        a_val = int(cluster_min_vals[cid]); b_val = int(cluster_max_vals[cid])
        if cluster_anchor_count[cid] == 2:
            missing_count = (b_val - a_val - 1)
        elif cluster_anchor_count[cid] == 1:
            if a_val == 1 and b_val <= K_val:
                # Anchor at high side
                missing_count = (b_val - 1)
            elif b_val == K_val and a_val >= 1:
                # Anchor at low side
                missing_count = (K_val - a_val)
            else:
                # Both 1 and K known (should be anchor_count=2 instead)
                missing_count = max(0, b_val - a_val - 1)
        else:
            missing_count = int(K_val) - 0  # all numbers 1..K
        # Compute ratio and slack flag
        if size > 0:
            cluster_ratio[cid] = missing_count / float(size)
        cluster_slack_flag[cid] = 1.0 if (size - missing_count) == 0 else 0.0
        # Compute density = blanks / bounding-box area
        sl = cluster_slices[cid-1]
        if sl is not None:
            height = sl[0].stop - sl[0].start
            width = sl[1].stop - sl[1].start
            area = height * width
            cluster_density[cid] = (size / area) if area > 0 else 0.0
        else:
            cluster_density[cid] = 0.0
    # Map cluster metrics to each blank cell
    blank_positions = np.argwhere(board_arr == -1)
    # Prepare output arrays for each blank index
    num_blanks = len(blank_positions)
    cluster_ratio_scores = np.zeros(num_blanks, dtype=float)
    cluster_size_scores = np.zeros(num_blanks, dtype=float)
    cluster_slack_scores = np.zeros(num_blanks, dtype=float)
    cluster_density_scores = np.zeros(num_blanks, dtype=float)
    for idx, (bi, bj) in enumerate(blank_positions):
        cid = region_map[bi, bj]
        cluster_ratio_scores[idx] = cluster_ratio[cid] if cluster_sizes[cid] > 0 else 0.0
        # Size score: inverse of cluster size (smaller clusters => higher score)
        cluster_size_scores[idx] = 1.0 / cluster_sizes[cid] if cluster_sizes[cid] > 0 else 0.0
        cluster_slack_scores[idx] = cluster_slack_flag[cid]
        # Density score: inverse of density (sparser cluster => higher score)
        density = cluster_density[cid]
        cluster_density_scores[idx] = 1.0 - density  # lower density yields higher score
    return (cluster_ratio_scores, cluster_size_scores, cluster_slack_scores, cluster_density_scores,
            cluster_min_vals, cluster_max_vals, cluster_anchor_count, region_map)