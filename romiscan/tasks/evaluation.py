import os
import tempfile

import luigi
import numpy as np
import open3d as o3d
from romidata import DatabaseConfig
from romidata import RomiTask
from romidata import io
from romidata.task import FilesetTarget
from romidata.task import ImagesFilesetExists
from romiscan.log import logger
from romiscan.tasks import cl
from romiscan.tasks import config
from romiscan.tasks import proc2d
from romiscan.tasks import proc3d
from romiscanner.tasks.lpy import VirtualPlant
from romiscan.tasks.proc3d import PointCloud


class EvaluationTask(RomiTask):
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter()

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def output(self):
        fileset_id = self.task_family  # self.upstream_task().task_id + "Evaluation"
        return FilesetTarget(DatabaseConfig().scan, fileset_id)

    def evaluate(self):
        raise NotImplementedError

    def run(self):
        res = self.evaluate()
        io.write_json(self.output_file(), res)


class VoxelGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)

    def run(self):
        import pywavefront
        import trimesh
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = {}
            min = np.min(x.vertices, axis=0)
            max = np.max(x.vertices, axis=0)
            arr_size = np.asarray((max - min) / cl.Voxels().voxel_size + 1,
                                  dtype=np.int) + 1
            for k in x.meshes.keys():
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(
                    np.asarray(x.meshes[k].faces))
                t.vertices = o3d.utility.Vector3dVector(
                    np.asarray(x.vertices))
                t.compute_triangle_normals()
                o3d.io.write_triangle_mesh(os.path.join(tmpdir, "tmp.stl"),
                                              t)
                m = trimesh.load(os.path.join(tmpdir, "tmp.stl"))
                v = m.voxelized(cl.Voxels().voxel_size)

                class_name = x.meshes[k].materials[0].name
                arr = np.zeros(arr_size)
                voxel_size = cl.Voxels().voxel_size
                origin_idx = np.asarray((v.origin - min) / voxel_size,
                                        dtype=np.int)
                arr[origin_idx[0]:origin_idx[0] + v.matrix.shape[0],
                origin_idx[1]:origin_idx[1] + v.matrix.shape[1],
                origin_idx[2]:origin_idx[2] + v.matrix.shape[2]] = v.matrix

                # The following is needed because there are rotation in lpy's output...
                arr = np.swapaxes(arr, 2, 1)
                arr = np.flip(arr, 1)

                res[class_name] = arr  # gaussian_filter(arr, voxel_size)

            bg = np.ones(arr.shape)
            for k in res.keys():
                bg = np.minimum(bg, 1 - res[k])
            res["background"] = bg
            io.write_npz(self.output_file(), res)


class PointCloudGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)
    pcd_size = luigi.IntParameter(default=100000)

    def run(self):
        import pywavefront
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        colors = config.PointCloudColorConfig().colors
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = o3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(
                    np.asarray(x.meshes[k].faces))
                t.vertices = o3d.utility.Vector3dVector(
                    np.asarray(x.vertices))
                t.compute_triangle_normals()

                class_name = x.meshes[k].materials[0].name

                # The following is needed because there are rotation in lpy's output...
                pcd = t.sample_points_poisson_disk(self.pcd_size)
                pcd_pts = np.asarray(pcd.points)
                pcd_pts = pcd_pts[:, [0, 2, 1]]
                pcd_pts[:, 1] *= -1
                pcd.points = o3d.utility.Vector3dVector(pcd_pts)

                color = np.zeros((len(pcd.points), 3))
                if class_name in colors:
                    color[:] = np.asarray(colors[class_name])
                else:
                    color[:] = np.random.rand(3)
                color = o3d.utility.Vector3dVector(color)
                pcd.colors = color

                res = res + pcd
                point_labels += [class_name] * len(pcd.points)

            io.write_point_cloud(self.output_file(), res)
            self.output_file().set_metadata({'labels': point_labels})


class ClusteredMeshGroundTruth(RomiTask):
    upstream_task = luigi.TaskParameter(default=VirtualPlant)

    def run(self):
        import pywavefront
        x = self.input_file()
        mtl_file = self.input().get().get_file(x.id + "_mtl")
        outfs = self.output().get()
        colors = config.PointCloudColorConfig().colors
        output_fileset = self.output().get()
        with tempfile.TemporaryDirectory() as tmpdir:
            io.to_file(x, os.path.join(tmpdir, "plant.obj"))
            io.to_file(x, os.path.join(tmpdir, "plant.mtl"))
            x = pywavefront.Wavefront(os.path.join(tmpdir, "plant.obj"),
                                      collect_faces=True, create_materials=True)
            res = o3d.geometry.PointCloud()
            point_labels = []

            for i, k in enumerate(x.meshes.keys()):
                t = o3d.geometry.TriangleMesh()
                t.triangles = o3d.utility.Vector3iVector(
                    np.asarray(x.meshes[k].faces))

                pts = np.asarray(x.vertices)
                pts = pts[:, [0, 2, 1]]
                pts[:, 1] *= -1

                t.vertices = o3d.utility.Vector3dVector(pts)
                t.compute_triangle_normals()

                class_name = x.meshes[k].materials[0].name

                k, cc, _ = t.cluster_connected_triangles()
                k = np.asarray(k)
                tri_np = np.asarray(t.triangles)
                for j in range(len(cc)):
                    newt = o3d.geometry.TriangleMesh(t.vertices,
                                                     o3d.utility.Vector3iVector(tri_np[k == j, :]))
                    newt.vertex_colors = t.vertex_colors
                    newt.remove_unreferenced_vertices()

                    f = output_fileset.create_file("%s_%03d" % (class_name, j))
                    io.write_triangle_mesh(f, newt)
                    f.set_metadata("label", class_name)


class PointCloudSegmentationEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc3d.PointCloud)
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)

    def evaluate(self):
        source = io.read_point_cloud(self.upstream_task().output_file())
        target = io.read_point_cloud(self.ground_truth().output_file())
        labels_gt = self.ground_truth().output_file().get_metadata('labels')
        labels = self.upstream_task().output_file().get_metadata('labels')
        pcd_tree = o3d.geometry.KDTreeFlann(target)
        res = {}
        unique_labels = set(labels_gt)
        for l in unique_labels:
            res[l] = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }
        for i, p in enumerate(source.points):
            li = labels[i]
            [k, idx, _] = pcd_tree.search_knn_vector_3d(p, 1)
            for l in unique_labels:
                if li == l:  # Positive cases
                    if li == labels_gt[idx[0]]:
                        res[l]["tp"] += 1
                    else:
                        res[l]["fp"] += 1
                else:  # Negative cases
                    if li == labels_gt[idx[0]]:
                        res[l]["tn"] += 1
                    else:
                        res[l]["fn"] += 1

        for l in unique_labels:
            res[l]["precision"] = res[l]["tp"] / (res[l]["tp"] + res[l]["fp"])
            res[l]["recall"] = res[l]["tp"] / (res[l]["tp"] + res[l]["fn"])
        return res


class PointCloudEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter()
    ground_truth = luigi.TaskParameter(default=PointCloudGroundTruth)
    max_distance = luigi.FloatParameter(default=2)

    def evaluate(self):
        source = io.read_point_cloud(self.upstream_task().output_file())
        target = io.read_point_cloud(self.ground_truth().output_file())
        labels = self.upstream_task().output_file().get_metadata('labels')
        labels_gt = self.ground_truth().output_file().get_metadata('labels')
        if labels is not None:
            eval = {"id": self.upstream_task().task_id}
            for l in set(labels_gt):
                idx = [i for i in range(len(labels)) if labels[i] == l]
                idx_gt = [i for i in range(len(labels_gt)) if labels_gt[i] == l]

                subpcd_source = o3d.geometry.PointCloud()
                subpcd_source.points = o3d.utility.Vector3dVector(
                    np.asarray(source.points)[idx])
                subpcd_target = o3d.geometry.PointCloud()
                subpcd_target.points = o3d.utility.Vector3dVector(
                    np.asarray(target.points)[idx_gt])

                if len(subpcd_target.points) == 0:
                    continue

                logger.debug("label : %s" % l)
                logger.debug("gt points: %i" % len(subpcd_target.points))
                logger.debug("pcd points: %i" % len(subpcd_source.points))
                res = o3d.registration.evaluate_registration(subpcd_source,
                                                                subpcd_target,
                                                                self.max_distance)
                eval[l] = {
                    "fitness": res.fitness,
                    "inlier_rmse": res.inlier_rmse
                }
        res = o3d.registration.evaluate_registration(source, target,
                                                        self.max_distance)
        eval["all"] = {
            "fitness": res.fitness,
            "inlier_rmse": res.inlier_rmse
        }
        return eval


class Segmentation2DEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=proc2d.Segmentation2D)
    ground_truth = luigi.TaskParameter(default=ImagesFilesetExists)
    hist_bins = luigi.IntParameter(default=100)
    tol_px = luigi.IntParameter()

    def evaluate(self):
        from scipy.ndimage.morphology import binary_dilation

        prediction_fileset = self.upstream_task().output().get()
        gt_fileset = self.ground_truth().output().get()
        channels = gt_fileset.get_metadata('channels')
        channels.remove('rgb')

        histograms = {}
        res = {}

        for c in channels:

            if (c == "background"):
                continue

            preds = prediction_fileset.get_files(query={'channel': c})
            # hist_high, disc = np.histogram(np.array([]), self.hist_bins, range=(0,1))
            # hist_low, disc = np.histogram(np.array([]), self.hist_bins, range=(0,1))
            res[c] = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }

            for pred in preds:
                ground_truth = gt_fileset.get_files(query={'channel': c,
                                                           'shot_id': pred.get_metadata(
                                                               'shot_id')})[0]
                im_pred = io.read_image(pred)
                im_gt = io.read_image(ground_truth)
                for i in range(self.tol_px):
                    im_gt_tol = binary_dilation(im_gt > 0)

                tp = int(np.sum(im_gt_tol * (im_pred > 0)))
                fn = int(np.sum(im_gt_tol * (im_pred == 0)))
                tn = int(np.sum((im_gt == 0) * (im_pred == 0)))
                fp = int(np.sum((im_gt == 0) * (im_pred > 0)))

                res[c]["tp"] += tp
                res[c]["fp"] += fp
                res[c]["tn"] += tn
                res[c]["fn"] += fn

            #         if c == "background":
            #             threshold = 254
            #         else:
            #             threshold = 0

            #         res[c]["tp"] += int(np.sum((im_gt > threshold) * (im_pred > 128)))
            #         res[c]["fp"] += int(np.sum((im_gt <= threshold) * (im_pred > 128)))
            #         res[c]["tn"] += int(np.sum((im_gt <= threshold) * (im_pred <= 128)))
            #         res[c]["fn"] += int(np.sum((im_gt > threshold) * (im_pred <= 128)))

            #         # im_gt_high = im_pred[im_gt > threshold] / 255
            #         # im_gt_low = im_pred[1-im_gt < threshold] / 255

            #         # hist_high_pred, bins_high = np.histogram(im_gt_high, self.hist_bins, range=(0,1))
            #         # hist_low_pred, bins_low = np.histogram(im_gt_low, self.hist_bins, range=(0,1))
            #         # hist_high += hist_high_pred
            #         # hist_low += hist_low_pred
            #     if res[c]["tp"] + res[c]["fp"] == 0:
            #         res.pop(c, None)
            #         continue

            res[c]["precision"] = res[c]["tp"] / (res[c]["tp"] + res[c]["fp"])

            if res[c]["tp"] + res[c]["fn"] == 0:
                res.pop(c, None)
                continue

            res[c]["recall"] = res[c]["tp"] / (res[c]["tp"] + res[c]["fn"])
        #     # histograms[c] = {"hist_high": hist_high.tolist(), "bins_high": bins_high.tolist(), "hist_low": hist_low.tolist(), "bins_low": bins_low.tolist()}
        # # return histograms
        return res


class VoxelsEvaluation(EvaluationTask):
    upstream_task = luigi.TaskParameter(default=cl.Voxels)
    ground_truth = luigi.TaskParameter(default=VoxelGroundTruth)
    hist_bins = luigi.IntParameter(default=100)

    def requires(self):
        return [self.upstream_task(), self.ground_truth()]

    def evaluate(self):
        prediction_file = self.upstream_task().output().get().get_files()[0]
        gt_file = self.ground_truth().output().get().get_files()[0]

        voxels = io.read_npz(prediction_file)
        gts = io.read_npz(gt_file)

        histograms = {}
        from matplotlib import pyplot as plt

        l = list(gts.keys())
        res = np.zeros((*voxels[l[0]].shape, len(l)))
        for i in range(len(l)):
            res[:, :, :, i] = voxels[l[i]]
        res_idx = np.argmax(res, axis=3)

        for i, c in enumerate(l):
            if c == "background":
                continue

            prediction_c = res_idx == i

            pred_no_c = np.copy(res)
            pred_no_c = np.max(np.delete(res, i, axis=3), axis=3)
            pred_c = res[:, :, :, i]

            prediction_c = prediction_c * (pred_c > (10 * pred_no_c))

            gt_c = gts[c]
            gt_c = gt_c[0:prediction_c.shape[0], 0:prediction_c.shape[1],
                   0:prediction_c.shape[2]]

            fig = plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(prediction_c.max(0))

            plt.subplot(2, 1, 2)
            plt.imshow(gt_c.max(0))

            fig.canvas.draw()
            x = fig.canvas.renderer.buffer_rgba()

            outfs = self.output().get()
            f = outfs.create_file("img_" + c)
            io.write_image(f, x, "png")

            # maxval = np.max(prediction_c)
            # minval = np.min(prediction_c)

            tp = np.sum(prediction_c[gt_c > 0.5])
            fn = np.sum(1 - prediction_c[gt_c > 0.5])
            fp = np.sum(prediction_c[gt_c < 0.5])
            tn = np.sum(1 - prediction_c[gt_c < 0.5])

            histograms[c] = {"tp": tp.tolist(), "fp": fp.tolist(),
                             "tn": tn.tolist(), "fn": fn.tolist()}
        return histograms


class CalibrationTest(RomiTask):
    """
    Extract specific features of a certain shape in a point cloud

    Module: romiscan.tasks.calibration_test
    Default upstream tasks: PointCloud
    Upstream task format: ply
    Output task format: json

    """
    upstream_task = luigi.TaskParameter(default=PointCloud)

    object_shape = luigi.Parameter(default="cylinder")

    def run(self):
        point_cloud = io.read_point_cloud(self.input_file())
        if self.object_shape == "cylinder":
            out = proc3d.get_radius(point_cloud)
            io.write_json(self.output_file(), out)
