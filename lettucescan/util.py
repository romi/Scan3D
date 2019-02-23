import numpy as np
import open3d
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from open3d.geometry import read_point_cloud, write_point_cloud, write_triangle_mesh, read_triangle_mesh


def index2point(indexes, origin, width):
    return indexes/width + origin[np.newaxis, :]


def point2index(points, origin, width):
    return np.array(np.round((points - origin[np.newaxis, :]) / width), dtype=int)


def pcd2vol(pcd_points, voxel_width, zero_padding=0):
    """
    Voxelize a point cloud. Every voxel value is equal to the number
    of points in the corresponding cube.

    :param pcd_points: Nx3 array of the 3D points
    :param voxel_width: Width of voxels (float)
    :param zero_padding: Space to leave around the volume
    :rtype: 3D numpy array
    """
    origin = np.min(pcd_points, axis=0)
    indices = point2index(pcd_points, origin, voxel_width)
    shape = indices.max(axis=0)

    vol = np.zeros(shape + 2*zero_padding + 1, dtype=np.float)
    indices = indices + zero_padding

    for i in range(pcd_points.shape[0]):
        vol[indices[i, 0], indices[i, 1], indices[i, 2]] += 1.

    return vol, origin


def vol2pcd(volume, origin, width, dist_threshold=0, quiet=False):
    """
    Converts a binary volume into a point cloud with normals.
    :param volume: NxMxP 3D binary numpy array
    :param width: voxel width
    :param dist[=0]: distance of the level set on which the points are sampled
    :rtype: open3D point cloud with normals
    """
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1-volume)
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    gx, gy, gz = np.gradient(dist)
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)

    on_edge = (dist > -dist_threshold) * (dist <= -dist_threshold+np.sqrt(3))
    x, y, z = np.nonzero(on_edge)

    if not quiet:
        print("number of points = %d" % len(x))
    pts = np.zeros((0, 3))
    normals = np.zeros((0,3))
    for i in range(len(x)):
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + dist_threshold - np.sqrt(3)/2
            pts = np.vstack([pts, np.array([x[i] - grad_normalized[0] * val,
                                      y[i] - grad_normalized[1] * val,
                                      z[i] - grad_normalized[2] * val])])
            normals = np.vstack([normals, -np.array([grad_normalized[0],
                                                     grad_normalized[1],
                                                     grad_normalized[2]])])

    pts = index2point(pts, origin, width)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pts)
    pcd.normals = open3d.Vector3dVector(normals)
    pcd.normalize_normals()
    return pcd

def db_read_point_cloud(file):
    path = fsdb._file_path(file, file.filename)
    return read_point_cloud(path)


def db_write_point_cloud(file, pcd):
    tmpdir = tempfile.TemporaryDirectory()
    pcd_path = os.path.join(tmpdir.name, '%s.ply'%file.id)
    write_point_cloud(pcd_path, pcd)
    file.import_file(pcd_path)


def db_read_triangle_mesh(file):
    path = fsdb._file_path(file, file.filename)
    return read_triangle_mesh(path)


def db_write_triangle_mesh(file, mesh):
    tmpdir = tempfile.TemporaryDirectory()
    mesh_path = os.path.join(tmpdir.name, '%s.ply'%file.id)
    write_triangle_mesh(mesh_path, mesh)
    file.import_file(mesh_path)
