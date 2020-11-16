import os
import tempfile

import luigi
import numpy as np
import open3d as o3d
import math
import random
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

NUM_POINTS = 10000


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


class CylinderRadiusGroundTruth(RomiTask):
    """
    Provide a point cloud with a cylindrical shape and a known radius
    """
    upstream_task = luigi.TaskParameter(ImagesFilesetExists)

    def run(self):
        total_number_of_resampled_pts = 10000
        radius = random.uniform(1, 100)
        # cylinder must (for the moment ...) be higher than large
        height = random.uniform(1, 100) + (radius * 2)
        mesh_cylinder = o3d.geometry.TriangleMesh()
        mesh_cylinder = mesh_cylinder.create_cylinder(radius=radius, height=height)
        triangle_id_list = np.array(mesh_cylinder.triangles)
        triangle_vertex_list = np.array(mesh_cylinder.vertices)

        v1 = []
        v2 = []
        v3 = []
        total_triangles = len(triangle_id_list)
        for i in range(total_triangles):
            current_triangle = triangle_id_list[i]
            v1_local_id = current_triangle[0]
            v2_local_id = current_triangle[1]
            v3_local_id = current_triangle[2]
            v1_coordinate = triangle_vertex_list[v1_local_id]
            v2_coordinate = triangle_vertex_list[v2_local_id]
            v3_coordinate = triangle_vertex_list[v3_local_id]
            v1.append(v1_coordinate)
            v2.append(v2_coordinate)
            v3.append(v3_coordinate)
        v1np = np.asarray(v1)  # just a data structure conversion for the ease of calculations
        v2np = np.asarray(v2)
        v3np = np.asarray(v3)
        all_triangle_areas = 0.5 * np.linalg.norm(np.cross(v2np - v1np, v3np - v1np), axis=1)
        weighted_areas = all_triangle_areas / all_triangle_areas.sum()
        random_triangle_list = np.random.choice(range(total_triangles), size=total_number_of_resampled_pts, p=weighted_areas)

        all_sampled_points = []
        total_number_of_resampled_pts = random_triangle_list.shape[0]  # new
        for i in range(total_number_of_resampled_pts):
            current_random_triangle = random_triangle_list[i]
            # currentLabel = globalTriangleLabelList[currentRandomTriangle][0]
            current_triangle_vertex_ids = triangle_id_list[current_random_triangle]  # a list
            v1_id = current_triangle_vertex_ids[0]
            v2_id = current_triangle_vertex_ids[1]
            v3_id = current_triangle_vertex_ids[2]
            v1_corrd = triangle_vertex_list[v1_id]  # a list
            v2_corrd = triangle_vertex_list[v2_id]
            v3_corrd = triangle_vertex_list[v3_id]

            alpha = np.random.rand(1)
            beta = np.random.rand(1)
            if (alpha + beta) > 1:
                alpha = 1 - alpha
                beta = 1 - beta
            gamma = 1 - (alpha + beta)

            sampled_point = [sum(x) for x in zip([i * alpha for i in v1_corrd], [i * beta for i in v2_corrd],
                                                 [i * gamma for i in v3_corrd])]
            sampled_point_x = sampled_point[0][0]  # because these are in the form of list of list
            sampled_point_y = sampled_point[1][0]
            sampled_point_z = sampled_point[2][0]
            new_point = [sampled_point_x, sampled_point_y, sampled_point_z]
            all_sampled_points.append(new_point)

        cylinder = np.array(all_sampled_points)
        gt_cyl = o3d.geometry.PointCloud()
        gt_cyl.points = o3d.utility.Vector3dVector(cylinder)
        o3d.visualization.draw_geometries([gt_cyl])
        io.write_point_cloud(self.output_file(), gt_cyl)
        self.output().get().set_metadata({'radius': radius})

        return gt_cyl


class CylinderRadiusEvaluation(RomiTask):
    """
    Extract specific features of a certain shape in a point cloud

    Module: romiscan.tasks.calibration_test
    Default upstream tasks: PointCloud
    Upstream task format: ply
    Output task format: json

    """
    upstream_task = luigi.TaskParameter(CylinderRadiusGroundTruth)

    def run(self):
        gt = io.read_point_cloud(self.input_file())
        cylinder_gt_fileset = self.input().get()
        pcd_points = np.array(gt.points)
        gt_radius = cylinder_gt_fileset.get_metadata("radius")
        print(gt_radius)

        center = np.mean(pcd_points, axis=0)

        bb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd_points))
        box_points = np.asarray(bb.get_box_points())

        # among the 8 points of the box, take the first one, find the 2 closest points from it
        # then the mean of this 2 points to get the middle of the smallest square of the box
        # then same with the square on the other side of the box in order to have the main direction of the organ
        first_point = box_points[0]
        closest_points = sorted(box_points, key=lambda p: np.linalg.norm(p - first_point))
        first_square_middle = np.add(closest_points[1], closest_points[2]) / 2
        opposite_square_middle = np.add(closest_points[5], closest_points[6]) / 2

        orientation_vector = opposite_square_middle - first_square_middle

        v = pcd_points - center
        normal_vec = orientation_vector/ np.linalg.norm(orientation_vector)
        dist = np.dot(v, normal_vec)
        yo = normal_vec * dist[:, np.newaxis]
        projected_points = pcd_points - yo

        # visualization
        pts_cyl = o3d.geometry.PointCloud()
        pts_cyl.points = o3d.utility.Vector3dVector(pcd_points)
        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(projected_points)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=center)
        cyl_orientation_open3d = o3d.geometry.LineSet()
        ev1_pts = [center, np.add(orientation_vector, center)]
        ev1_lines = [[0, 1]]
        cyl_orientation_open3d.points = o3d.utility.Vector3dVector(ev1_pts)
        cyl_orientation_open3d.lines = o3d.utility.Vector2iVector(ev1_lines)
        # o3d.visualization.draw_geometries([pts, cyl_orientation_open3d, mesh_frame, pts_cyl]) #

        a = 2 * projected_points
        a[:, 2] = 1
        pp_2d = np.delete(projected_points, 2, axis=1)
        b = np.linalg.norm(pp_2d, axis=1) ** 2
        cx, cy, comp = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
        radius = math.sqrt(comp + cx ** 2 + cy ** 2)

        io.write_json(self.output_file(), {"calculated_radius": radius, "gt_radius": gt_radius})
