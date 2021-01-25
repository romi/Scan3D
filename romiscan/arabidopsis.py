"""
romican.arabidopsis
-------------------

This module contains all function related to the segmentation of arabidopsis
from its curve skeleton. The two main functionalities are:

    * Computing a tree  from the curve skeleton
    * Estimating the angle between successive fruits from the tree.
"""
import collections
import operator

import networkx as nx
import numpy as np
from romiscan.log import logger


def get_main_stem_and_nodes(G, root_node):
    """
    Get main stem and branching nodes from graph.

    Parameters
    ----------
    G : networkx.graph
        input graph
    root_node : int
        index of the root node

    Returns
    -------
    nodes
    """
    # Get main stem as shortest path to point furthest from root
    predecessors, distances_to_root = nx.dijkstra_predecessor_and_distance(
        G, root_node)
    i = max(distances_to_root.items(), key=operator.itemgetter(1))[0]
    main_stem = [i]
    current_node = i
    while current_node != root_node:
        current_node = predecessors[current_node][0]
        main_stem.append(current_node)
    main_stem = np.array(main_stem, dtype=int)

    # Get nodes, sorted from closest to furthest to the root
    n_neighbors = np.array([len(list(nx.neighbors(G, g)))
                            for g in main_stem], dtype=int)
    nodes = main_stem[n_neighbors > 2]
    nodes = nodes
    return main_stem, nodes


def compute_mst(G, main_stem, nodes):
    """
    Computes a minimum spanning tree in a graph using some custom
    distance. The distance is some function of the distance of the node
    to the closest node in the stem. This may need to be documented
    a bit more!!!

    Parameters
    ----------
    G : nx.Graph
        input graph
    main_stem : list
        list  of ids of points in the main stem
    nodes : list
        list of branching  points on the main stem.

    Returns
    -------
    nx.Graph
    """
    # Set weights proportional to distance to node
    # (for computing minimum spanning tree)
    G = G.copy()
    distances = {}
    tot = []
    for i in nodes:
        _, distances[i] = nx.dijkstra_predecessor_and_distance(G, i)
        tot.append(list(distances[i].values()))
    total_distances = np.array(tot)
    total_distances.flatten()
    try:
        max_dist = np.amax(total_distances)
    except ValueError:
        max_dist = 10000

    distance_to_node = {}
    for n in G.nodes():
        dist = []
        for i in nodes:
            try:
                dist.append(distances[i][n])
            except:
                logger.warning(f"No distance found for node {i}")
        if len(dist):
            distance_to_node[n] = min(dist)
        else:
            distance_to_node[n] = max_dist
        # distance_to_node[n] = min(distances[i][n] for i in nodes)


    def node_penalty(u, v):
        if u in main_stem or v in main_stem:
            return 0
        if len(list(nx.neighbors(G, u))) > 2 or len(list(nx.neighbors(G, v))) > 2:
            return 10000 + distance_to_node[u] + distance_to_node[v]
        return (distance_to_node[u] + distance_to_node[v])

    for u, v in G.edges():
        G[u][v]['weight'] = node_penalty(u, v)

    # Compute minimum spanning tree
    T = nx.minimum_spanning_tree(G)
    return T


def build_graph(vertices, edges):
    """
    Buils a networkx graph from a list of vertices and edges.

    Parameters
    ----------
    vertices : np.ndarray
        Input Nx3 array of points
    edges : np.ndarray
        Input Mx2 array of lines between points (dtype = int)

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(0, vertices.shape[0]))

    for i in range(edges.shape[0]):
        G.add_edge(edges[i, 0], edges[i, 1],
                   weight=np.linalg.norm(vertices[edges[i, 0], :] - vertices[edges[i, 1], :]))
    return G


def fit_plane(points):
    """
    Fit a plane to a set of points. Points is Nx3
    """
    m = points.mean(axis=0)
    points = points - m[np.newaxis, :]
    u, s, v = np.linalg.svd(points)
    return m, v[0, :], v[1, :]


def nx_to_tx(T, attributes, root_id):
    """
    Converts a networkx graph object which is a tree into
    a treex tree.

    Parameters
    ----------
    T: networkx graph
        input graph (must be a tree).
    attributes: list of dict
        each element of the list is a dict containing the attributes of the corresponding node
        in the graph.
    root_id: int
        id of the root node
    """
    successors = nx.dfs_successors(T, source=root_id)
    TT = tree.Tree()
    for k in attributes[root_id].keys():
        TT.add_attribute_to_id(k, attributes[root_id][k])
    Q = collections.deque()
    Q.append((root_id, TT))
    while len(Q) > 0:
        current_id, current_T = Q.pop()
        if current_id in successors:
            for child_id in successors[current_id]:
                new_T = tree.Tree()
                current_T.add_subtree(new_T)
                for k in attributes[child_id].keys():
                    new_T.add_attribute_to_id(k, attributes[child_id][k])
                Q.append((child_id, new_T))
    return TT


def label_fruit(G, branching_fruit_id, fruit_id):
    """
    Labels fruits in a treex tree object.

    Parameters
    ----------
    T: treex.tree.Tree
        input tree which root is the branching node
    """
    Q = collections.deque()
    Q.append(branching_fruit_id)
    while len(Q) > 0:
        current_id = Q.pop()
        for new_id in G.neighbors(current_id):
            node_data = G.nodes[new_id]
            labels = node_data["labels"]
            if not "stem" in labels and "fruit" not in labels:
                labels.append("fruit")
                node_data["fruit_id"] = fruit_id
                Q.append(new_id)


def compute_tree_graph(points, lines, stem_axis, stem_axis_inverted):
    """
    Returns a networkx tree object from the curve skeleton.
    Labels include segmentation of main stem and organs,as well as position in space
    of the points.

    Parameters
    ----------
    points: np.ndarray
        Nx3 position of points
    lines: np.ndarray
        Nx2 lines between points (dtype=int)
    stem_axis: int
        axis to use for stem orientation to get the root node
    stem_axis_inverted: bool
        direction of the stem along the specified axis inverted or not

    Returns
    -------
    nx.Graph
    """
    points, lines = np.asarray(points), np.asarray(lines)
    G = build_graph(points, lines)

    # Get the root node
    if stem_axis_inverted:
        root_node = np.argmin(points[:, stem_axis])
    else:
        root_node = np.argmax(points[:, stem_axis])

    # Get the main stem and node locations
    main_stem, branching_points = get_main_stem_and_nodes(G, root_node)

    attributes = {}
    for i in range(len(points)):
        label = []
        if i in main_stem:
            label.append("stem")
        if i in branching_points:
            label.append("node")
        attributes[i] = {"position": points[i].tolist(),
                         "labels": label}

    for i, n_i in enumerate(branching_points):
        attributes[n_i]["fruit_id"] = i

    for i, n_i in enumerate(main_stem):
        attributes[n_i]["main_stem_id"] = i

    # Compute the minimum spanning tree
    T = compute_mst(G, main_stem, branching_points)
    nx.set_node_attributes(T, attributes)

    for i, n_i in enumerate(branching_points):
        label_fruit(T, n_i, i)

    return T


def get_nodes_by_label(G, label):
    """
    Get all nodes in a graph which have the given label. The key "labels"
    must exist in the data of each node.

    Parameters
    ----------
    G : nx.Graph
    label: str

    Returns
    -------
    list
    """
    return [i for i in G.nodes if label in G.nodes[i]["labels"]]


def get_fruit(G, i):
    x = get_nodes_by_label(G, "fruit")
    return [j for j in x if G.nodes[j]["fruit_id"] == i]


def get_organ_features(organ_bb, stem_skeleton):
    """
    Compute organ features as its main direction and its node

    Parameters
    ----------
    organ_bb : open3d.geometry.OrientedBoundingBox
        bounding box around the organ
    stem_skeleton: np.array
        Nx3 array

    Returns
    -------
    dict
    """
    # create box around organ (fruit)
    box_points = np.asarray(organ_bb.get_box_points())

    # among the 8 points of the box, take the first one, find the 2 closest points from it
    # then the mean of this 2 points to get the middle of the smallest square of the box
    # then same with the square on the other side of the box in order to have the main direction of the organ
    first_point = box_points[0]
    closest_points = sorted(box_points, key=lambda p: np.linalg.norm(p - first_point))
    first_square_middle = np.add(closest_points[1], closest_points[2]) / 2
    opposite_square_middle = np.add(closest_points[5], closest_points[6]) / 2
    width = np.linalg.norm(first_point - closest_points[1])

    # the node with the closest distance between the 2 centers of the smallest squares is
    # considered as the organ node
    dist_first = np.sum((stem_skeleton - first_square_middle) ** 2, axis=1)
    dist_opposite = np.sum((stem_skeleton - opposite_square_middle) ** 2, axis=1)

    if dist_first[np.argmin(dist_first)] <= dist_opposite[np.argmin(dist_opposite)]:
        node_id = np.argmin(dist_first)
        direction = opposite_square_middle - first_square_middle
        fruit_base = first_square_middle
    else:
        node_id = np.argmin(dist_opposite)
        direction = first_square_middle - opposite_square_middle
        fruit_base = opposite_square_middle

    organ_features = {
        "node_id": node_id,
        "direction": direction,
        "mesh": organ_bb,
        "fruit_base": fruit_base,
        "width": width
    }
    return organ_features


def angles_and_internodes_from_point_cloud(stem_pcd, organ_pcd_list, characteristic_length, stem_axis
                                           , stem_axis_inverted, min_elongation_ratio, min_fruit_size):
    """
    Get angles and internodes from point cloud
    Parameters
    ----------
    stem_pcd : o3d.geometry.PointCloud
        point cloud of the stem
    organ_pcd_list : list
        list of o3d.geometry.PointCloud organs
    characteristic_length : int
        distance between 2 elements for the "stem skeletonization"
    stem_axis : int
        [0,1,2] for the projection of the stem on the x, y or z axis
    stem_axis_inverted : bool
        whether or not the stem is inverted
    min_elongation_ratio : float
        minimum elongation ratio for the organ to be considered for the angles and internodes calculation
    min_fruit_size : float
        minimum fruit size

    Returns
    -------
    {dict}
        list of angles, internodes and fruit points
    """
    import open3d
    from romidata import io

    stem_points = np.asarray(stem_pcd.points)

    idx_min = np.argmin(stem_points[:, stem_axis])
    idx_max = np.argmax(stem_points[:, stem_axis])

    stem_axis_min = stem_points[idx_min, stem_axis]
    stem_axis_max = stem_points[idx_max, stem_axis]

    stem_frame_axis = np.arange(stem_axis_min, stem_axis_max, characteristic_length)

    kdtree = open3d.geometry.KDTreeFlann(stem_pcd)

    point = stem_points[idx_min]
    root = stem_points[idx_min]

    stem_skeleton = np.zeros((len(stem_frame_axis), 3))

    for i, axis in enumerate(stem_frame_axis):
        point[stem_axis] = axis
        k, idx, _ = kdtree.search_knn_vector_3d(point, 300)
        vtx = stem_points[idx]
        mean = vtx.mean(axis=0)
        if i == 0:
            root = mean

        point = mean
        stem_skeleton[i, :] = mean

    unique_stem_skeleton = np.unique(stem_skeleton, axis=0)
    # order the stem skeleton with the distance to the root
    ordered_stem_skeleton = sorted(unique_stem_skeleton, key=lambda p: np.linalg.norm(p - root))
    if stem_axis_inverted:
        ordered_stem_skeleton = ordered_stem_skeleton[::-1]

    # stem_mesh.paint_uniform_color([0.5, 0.5, 0.5])
    # main_f = open3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=root)
    # open3d.visualization.draw_geometries([ls, *gs, stem_mesh, main_f])

    # calculate features as direction and corresponding node id for each organ
    organs_features_list = []
    for i, o in enumerate(organ_pcd_list):
        bb = open3d.geometry.OrientedBoundingBox.create_from_points(o.points)
        organ_features = get_organ_features(bb, ordered_stem_skeleton)
        organ_features["points"] = np.asarray(o.points).tolist()
        elongation_ratio = np.linalg.norm(organ_features["direction"]) / np.linalg.norm(organ_features["width"])
        # if the organ does not fit to the minimal fruit size and elongation ratio it is not taken into account
        if np.linalg.norm(organ_features["direction"]) > min_fruit_size and elongation_ratio > min_elongation_ratio:
            organs_features_list.append(organ_features)

    angles = []
    internodes = []
    fruit_points = []
    ordered_organs = sorted(organs_features_list, key=lambda p: p["node_id"])
    # initialization for the first organ
    current_organ = ordered_organs[0]
    fruit_points.append(np.asarray(current_organ["points"]).tolist())
    for next_organ in ordered_organs[1:]:
        # main stem direction
        node = ordered_stem_skeleton[current_organ["node_id"]]
        next_node = ordered_stem_skeleton[next_organ["node_id"]]
        # ... takes into account organs with the same node at extremities
        if (node == next_node).all():
            if current_organ["node_id"] == (len(ordered_stem_skeleton) - 1):
                n = node - ordered_stem_skeleton[current_organ["node_id"] - 1]
            else:
                n = ordered_stem_skeleton[current_organ["node_id"] + 1] - node
        else:
            n = next_node - node
        n /= np.linalg.norm(n)

        # projection on the plane normal to the main stem direction
        current_organ_projection = current_organ["direction"] - (np.dot(current_organ["direction"], n) * n)
        next_organ_projection = next_organ["direction"] - (np.dot(next_organ["direction"], n) * n)

        n1 = current_organ_projection / np.linalg.norm(current_organ_projection)
        n2 = next_organ_projection / np.linalg.norm(next_organ_projection)

        # angle calculation
        a = np.dot(np.cross(n2, n1), n)
        b = np.dot(n1, n2)
        angle = np.arctan2(a, b)
        if angle < 0:
            angle = 2 * np.pi + angle

        internode = np.linalg.norm(node - next_node)

        angles.append(angle)
        internodes.append(internode)
        fruit_points.append(np.asarray(next_organ["points"]).tolist())

        current_organ = next_organ

    # as the angles are always calculated anticlockwise, this part takes the complementary angle if it is not the case
    # for the plant
    if np.median(angles) > np.pi:
        angles = 2 * np.pi - np.array(angles)
        angles = angles.tolist()

    # open3d.visualization.draw_geometries([ls, *gs, stem_mesh, *lg, fruit_mesh, main_f])
    return {"angles": angles, "internodes": internodes, "fruit_points": fruit_points}


def compute_angles_and_internodes(T, n_nodes_fruit=5, n_nodes_stem=5):
    """
    Get angle and internodes from graph

    Parameters
    ----------
    T : nx.Graph
        input tree as a networkx graph
    n_nodes_fruit : int
        number of nodes to consider as neighbours for a branching point
    n_nodes_stem : int
        number of nodes to consider as neighbours in the stem

    Returns
    -------
    dict
    """

    unordered_main_stem = get_nodes_by_label(T, "stem")
    unordered_branching_points = get_nodes_by_label(T, "node")
    angles = []
    internodes = []
    all_fruit_points = []
    node_info_list = []

    # seems like the branching order is lost in the treegraoh computation task
    # re order nodes
    nodes_dict = {}
    for ubp in unordered_branching_points:
        nodes_dict[ubp] = T.nodes[ubp]["fruit_id"]
    branching_points = [k for k, v in sorted(nodes_dict.items(), key=lambda item: item[1])]

    # re order main stem
    stem_dict = {}
    for umn in unordered_main_stem:
        stem_dict[umn] = T.nodes[umn]["main_stem_id"]
    main_stem = [k for k, v in sorted(stem_dict.items(), key=lambda item: item[1])]

    for i in range(len(branching_points) - 1):
        node_point = np.array(T.nodes[branching_points[i]]["position"])
        node_next_point = np.array(T.nodes[branching_points[i + 1]]["position"])

        node_fruit_points = [np.array(T.nodes[n]["position"]) for n in get_fruit(T, i)]

        if len(node_fruit_points) > 1:
            vertices_fruit_plane_est = node_fruit_points[0:n_nodes_fruit]
            idx = main_stem.index(branching_points[i])
            stem_neighbors_id = main_stem[idx - n_nodes_stem // 2:idx + n_nodes_stem // 2]
            vertices_node_plane_est = [T.nodes[stem_id]["position"] for stem_id in stem_neighbors_id]

            points = np.vstack([vertices_fruit_plane_est, vertices_node_plane_est])
            _, v1, v2 = fit_plane(points)

            fruit_points = np.vstack(node_fruit_points)
            fruit_mean = fruit_points.mean(axis=0)
            all_fruit_points.append(fruit_points.tolist())

            new_v1 = fruit_mean - node_point
            new_v1 = new_v1.dot(v1) * v1 + new_v1.dot(v2) * v2
            new_v1 /= np.linalg.norm(new_v1)

            # Set v1 as the fruit direction and v2 as the stem direction
            v1, v2 = new_v1, v2 - v2.dot(new_v1) * new_v1
            if v2.dot(node_next_point - node_point) < 0:
                v2 = - v2

            node_info_list.append({
                "node_point": node_point,
                "fruit_direction": v1,
                "stem_direction": v2
            })

    for i in range(1, len(node_info_list)):
        n1 = np.cross(node_info_list[i - 1]["fruit_direction"], node_info_list[i - 1]["stem_direction"])
        n2 = np.cross(node_info_list[i]["fruit_direction"], node_info_list[i]["stem_direction"])
        p1 = node_info_list[i - 1]["node_point"]
        p2 = node_info_list[i]["node_point"]
        v1 = node_info_list[i - 1]["fruit_direction"]
        v2 = node_info_list[i]["fruit_direction"]
        v3 = node_info_list[i]["node_point"] - node_info_list[i - 1]["node_point"]

        # Angle between the planes, between 0 and PI
        angle = np.arccos(np.dot(n1, n2))

        # IF basis is direct, then angle is positive (depends on stem axis inversion ?)
        if np.linalg.det([v1, v2, v3]) < 0:
            angle = 2 * np.pi - angle

        angles.append(angle)
        internodes.append(np.linalg.norm(p2 - p1))

    # complement angles if needed
    if np.median(angles) > np.pi:
        angles = 2 * np.pi - np.array(angles)
        angles = angles.tolist()

    return {
        "angles": angles,
        "internodes": internodes,
        "fruit_points": all_fruit_points
    }
