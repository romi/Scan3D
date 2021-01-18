import luigi
import numpy as np
import open3d as o3d

from romidata import RomiTask
from romidata import io
from romiscan import proc3d
from romiscan.log import logger
from romiscan.tasks import config
from romiscan.tasks.cl import Voxels
from romiscan.tasks.colmap import Colmap
from romiscan.tasks.proc2d import Segmentation2D


class PointCloud(RomiTask):
    """ Computes a point cloud from volumetric voxel data (either single or multiclass).

    Module: romiscan.tasks.proc3d
    Default upstream tasks: Voxels
    Upstream task format: npz file with as many 3D array as classes
    Output task format: single point cloud in ply. Metadata may include label name if multiclass.

    """
    upstream_task = luigi.TaskParameter(default=Voxels)
    level_set_value = luigi.FloatParameter(default=1.0)
    log = luigi.BoolParameter(default=False)
    background_prior = luigi.FloatParameter(default=1.0)
    min_contrast = luigi.FloatParameter(default=10.0)
    min_score = luigi.FloatParameter(default=0.2)

    def run(self):
        ifile = self.input_file()
        try:
            voxels = io.read_npz(ifile)
            if (len(voxels.keys()) == 1):
                multiclass = False
                voxels = voxels[list(voxels.keys())[0]]
            else:
                multiclass = True
        except:
            voxels = io.read_volume(ifile)
            multiclass = False
        if multiclass:
            l = list(voxels.keys())
            # background_idx = l.index("background")
            # l.remove("background")
            res = np.zeros((*voxels[l[0]].shape, len(l)))
            for i in range(len(l)):
                res[:, :, :, i] = voxels[l[i]]
            for i in range(len(l)):
                if l[i] == 'background':
                    res[:, :, :, i] *= self.background_prior

            # bg = voxels["background"] > voxels["background"].max() - 10

            res_idx = np.argmax(res, axis=3)
            # res_value = np.amax(res, axis=3)

            # threshold= np.quantile(res_value.flatten(), 0.99)
            # res_idx[res_value < threshold] = background_idx # low scores belong to background

            pcd = o3d.geometry.PointCloud()
            origin = np.array(ifile.get_metadata('origin'))

            voxel_size = float(ifile.get_metadata('voxel_size'))
            point_labels = []
            colors = config.PointCloudColorConfig().colors

            for i in range(len(l)):
                logger.debug(f"label = {l[i]}")
                if l[i] != 'background':
                    pred_no_c = np.copy(res)
                    pred_no_c = np.max(np.delete(res, i, axis=3), axis=3)
                    pred_c = res[:, :, :, i]
                    pred_c = (res_idx == i)
                    if self.min_contrast > 1.0:
                        pred_c *= (pred_c > (self.min_contrast * pred_no_c))
                    pred_c *= (pred_c > self.min_score)

                    out = proc3d.vol2pcd(pred_c, origin, voxel_size,
                                         self.level_set_value)
                    color = np.zeros((len(out.points), 3))
                    if l[i] in colors:
                        color[:] = np.asarray(colors[l[i]])
                    else:
                        color[:] = np.random.rand(3)
                    color = o3d.utility.Vector3dVector(color)
                    out.colors = color
                    pcd = pcd + out
                    point_labels = point_labels + [l[i]] * len(out.points)

            io.write_point_cloud(self.output_file(), pcd)
            self.output_file().set_metadata({'labels': point_labels})

        else:
            origin = np.array(ifile.get_metadata('origin'))
            voxel_size = float(ifile.get_metadata('voxel_size'))
            out = proc3d.vol2pcd(voxels, origin, voxel_size,
                                 self.level_set_value)
            io.write_point_cloud(self.output_file(), out)
            self.output_file().set_metadata({'voxel_size': voxel_size})


class SegmentedPointCloud(RomiTask):
    """ Segments an existing point cloud using 2D pictures
    """
    upstream_task = luigi.TaskParameter(default=Colmap)
    upstream_segmentation = luigi.TaskParameter(default=Segmentation2D)
    use_colmap_poses = luigi.BoolParameter(default=True)

    def requires(self):
        return [self.upstream_task(), self.upstream_segmentation()]

    def load_point_cloud(self):
        try:
            x = self.requires()[0].output().get().get_file("dense")
            return io.read_point_cloud(x)
        except:
            x = self.requires()[0].output().get().get_files()[0]
            return io.read_point_cloud(x)

    def is_in_pict(self, px, shape):
        return px[0] >= 0 and px[0] < shape[1] and px[1] >= 0 and px[1] < shape[0]

    def run(self):
        fs = self.upstream_segmentation().output().get()
        pcd = self.load_point_cloud()
        pts = np.asarray(pcd.points)
        ifile = self.input_file()

        labels = set()
        for fi in fs.get_files():
            label = fi.get_metadata('channel')
            if label is not None:
                labels.add(label)
        labels = list(labels)
        labels.remove('background')
        if 'rgb' in labels:
            labels.remove('rgb')

        scores = np.zeros((len(labels), len(pts)))

        for fi in fs.get_files():
            label = fi.get_metadata("channel")
            if label not in labels:
                continue

            if self.use_colmap_poses:
                camera = fi.get_metadata("colmap_camera")
            else:
                camera = fi.get_metadata("camera")

            if camera is None:
                logger.warning(
                    "Could not get camera pose for view, skipping...")
                continue

            rotmat = np.array(camera["rotmat"])
            tvec = np.array(camera["tvec"])

            intrinsics = camera["camera_model"]["params"]
            K = np.array([[intrinsics[0], 0, intrinsics[2]],
                          [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
            pixels = np.asarray(
                proc3d.backproject_points(pts, K, rotmat, tvec) + 0.5,
                dtype=int)

            label_idx = labels.index(label)
            mask = io.read_image(fi)
            for i, px in enumerate(pixels):
                if self.is_in_pict(px, mask.shape):
                    scores[label_idx, i] += mask[px[1], px[0]]

        pts_labels = np.argmax(scores, axis=0).flatten()
        logger.critical(f"Processed following labels: {labels}")

        colors = config.PointCloudColorConfig().colors
        logger.critical(f"Associated colors: {colors}")

        color_array = np.zeros((len(pts), 3))
        point_labels = [""] * len(pts)
        for i in range(len(labels)):
            nlab_pts = (pts_labels == i).sum()
            logger.critical(f"Number of points associated to label '{labels[i]}': {nlab_pts}")
            if labels[i] in colors:
                color_array[pts_labels == i, :] = np.asarray(colors[labels[i]])
            else:
                color_array[pts_labels == i, :] = np.random.rand(3)
            l = np.nonzero(pts_labels == i)[0].tolist()
            for u in l:
                point_labels[u] = labels[i]
        pcd.colors = o3d.utility.Vector3dVector(color_array)
        out = self.output_file()
        io.write_point_cloud(out, pcd)
        out.set_metadata("labels", point_labels)


class TriangleMesh(RomiTask):
    """ Triangulates input point cloud.

    Currently ignores class data and needs only one connected component.

    Module: romiscan.tasks.proc3d
    Default upstream tasks: PointCloud
    Upstream task format: ply file
    Output task format: ply triangle mesh file

    """
    upstream_task = luigi.TaskParameter(default=PointCloud)
    library = luigi.Parameter(default="cgal")  # ["cgal", "open3d"]

    def run(self):
        from romiscan import proc3d
        point_cloud = io.read_point_cloud(self.input_file())
        if self.library == "cgal":
            out = proc3d.pcd2mesh(point_cloud)
        elif self.library == "open3d":
            out = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)
        io.write_triangle_mesh(self.output_file(), out)


class ClusteredMesh(RomiTask):
    upstream_task = luigi.TaskParameter(default=SegmentedPointCloud)
    min_vol = luigi.FloatParameter(default=1.0)
    min_length = luigi.FloatParameter(default=10.0)

    def run(self):
        x = io.read_point_cloud(self.input_file())
        all_points = np.asarray(x.points)
        all_normals = np.asarray(x.normals)
        all_colors = np.asarray(x.colors)

        # Get the list of semantic label ('flower', 'fruit', ...) attached to each points of the point cloud
        labels = self.input_file().get_metadata("labels")
        output_fileset = self.output().get()
        # Loop on the unique set of labels:
        for l in set(labels):
            pcd = o3d.geometry.PointCloud()
            # Get the index of points matching the semantic label
            idx = [i for i in range(len(labels)) if labels[i] == l]
            # Select points, normals & colors for those point (to reconstruct a point cloud)
            points = all_points[idx, :]
            normals = all_normals[idx, :]
            colors = all_colors[idx, :]
            # Skip point cloud reconstruction if no points corresponding to label
            if len(points) == 0:
                logger.critical(f"No points found for label: '{l}'")
                continue
            # Reconstruct colored point cloud with normals:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # Mesh the point cloud (built with the points corresponding to the label)
            t, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
            t.compute_adjacency_list()
            k, cc, _ = t.cluster_connected_triangles()
            k = np.asarray(k)
            tri_np = np.asarray(t.triangles)
            for j in range(len(cc)):
                newt = o3d.geometry.TriangleMesh(t.vertices,
                                                 o3d.utility.Vector3iVector(tri_np[k == j, :]))
                newt.vertex_colors = t.vertex_colors
                newt.remove_unreferenced_vertices()

                f = output_fileset.create_file("%s_%03d" % (l, j))
                io.write_triangle_mesh(f, newt)
                f.set_metadata("label", l)


class OrganSegmentation(RomiTask):
    """Organ detection using DBSCAN clustering on the SegmentedPointCloud.

    This is done for each semantic label ('flower', 'fruit', ...) of the labelled point cloud,
    except for the stem as it is considered to be one organ.
    This task is suitable to detect organs on a point cloud where organs are detached from each other since
    it use the DBSCAN clustering method with a density estimator.

    Attributes
    ----------
    upstream_task : luigi.TaskParameter
        Upstream task.
        `SegmentedPointCloud` by default.
    eps : luigi.FloatParameter
        The maximum euclidean distance between two samples for one to be considered as in the neighborhood of the other.
        This is not a maximum bound on the distances of points within a cluster.
        `2.0` by default.
    min_points : luigi.IntParameter
        The number of points in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
        `5` by default.

    """
    upstream_task = luigi.TaskParameter(default=SegmentedPointCloud)
    eps = luigi.FloatParameter(default=2.0)
    min_points = luigi.IntParameter(default=5)

    def get_label_pointcloud(self, pcd, labels, label):
        """Return a point cloud only for the selected label.

        Parameters
        ----------
        pcd : open3d.geometry.PointCloud
            A PointCloud instance with points.
        labels : list
            List of labels associated to the points.
        label : str
            Label used to select points from pointcloud.

        Returns
        open3d.geometry.PointCloud
            A point cloud containing only the points associated to the selected label.
        """
        # Get the index of points matching the semantic label
        idx_mask = np.where(np.array(labels) == label)[0]
        # Skip point cloud reconstruction if no points corresponding to label
        n_points = sum(idx_mask)
        if n_points == 0:
            print(f"No points found for label: '{label}'!")
        else:
            print(f"Found {n_points} point for, label '{label}'.")
        # Returns point cloud (colored & with normals if any):
        return pcd.select_by_index(list(idx_mask))

    def run(self):
        # Read the pointcloud from the `upstream_task`
        labelled_pcd = io.read_point_cloud(self.input_file())
        # Initialize the output FileSet object.
        output_fileset = self.output().get()
        # Get the list of semantic label ('flower', 'fruit', ...) attached to each points of the point cloud
        labels = self.input_file().get_metadata("labels")
        unique_labels = set(labels)
        # Loop on the unique set of labels:
        for label in unique_labels:
            label_pcd = self.get_label_pointcloud(labelled_pcd, labels, label)
            # Exclude stem from clustering
            if label == 'stem':
                f = output_fileset.create_file("%s_%03d" % (label, 0))
                io.write_point_cloud(f, label_pcd)
                f.set_metadata("label", label)
                continue
            # DBSCAN clustering:
            clustered_arr = np.array(label_pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))

            ids = np.unique(clustered_arr)
            n_ids = len(ids)
            print(f"Found {n_ids} clusters in the point cloud!")
            # For each organ
            for i in ids:
                # Exclude outliers points (-1) from output point clouds
                if i == -1:
                    continue
                cluster_pcd = self.get_label_pointcloud(label_pcd, clustered_arr, i)
                f = output_fileset.create_file("%s_%03d" % (label, i))
                io.write_point_cloud(f, cluster_pcd)
                f.set_metadata("label", label)


class CurveSkeleton(RomiTask):
    """ Creates a 3D curve skeleton.

    Module: romiscan.tasks.proc3d
    Default upstream tasks: TriangleMesh
    Upstream task format: ply triangle mesh
    Output task format: json with two entries "points" and "lines" (TODO: precise)

    """
    upstream_task = luigi.TaskParameter(default=TriangleMesh)

    def run(self):
        from romiscan import proc3d
        mesh = io.read_triangle_mesh(self.input_file())
        out = proc3d.skeletonize(mesh)
        io.write_json(self.output_file(), out)


class VoxelsWithPrior(RomiTask):
    """
    Assign class to voxel adjusting for the possibility that
    projection can be wrongly labeled.
    """
    upstream_task = luigi.TaskParameter(default=Voxels)
    recall = luigi.DictParameter(default={})
    specificity = luigi.DictParameter(default={})
    n_views = luigi.IntParameter()

    def run(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        voxels = io.read_npz(prediction_file)
        out = {}
        l = list(voxels.keys())
        for c in l:
            if c in self.recall:
                recall = self.recall[c]
            else:
                continue
            if c in self.specificity:
                specificity = self.specificity[c]
            else:
                continue
            l0 = (self.n_views - voxels[c]) * np.log(specificity) + voxels[
                c] * np.log(1 - specificity)
            l1 = (self.n_views - voxels[c]) * np.log(1 - recall) + voxels[
                c] * np.log(recall)
            out[c] = l1 - l0

        outfs = self.output().get()
        outfile = self.output_file()
        io.write_npz(outfile, out)
        outfile.set_metadata(prediction_file.get_metadata())
