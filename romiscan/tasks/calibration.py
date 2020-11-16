import luigi
import numpy as np
import open3d as o3d
import math
from romidata import io
from romidata import RomiTask
from romiscan.tasks.proc3d import PointCloud


class CylinderRadius(RomiTask):
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
        scan = self.input().get().scan
        measures = scan.get_measures()
        point_cloud = io.read_point_cloud(self.input_file())
        if self.object_shape == "cylinder":
            pcd_points = np.array(point_cloud.points)
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
            o3d.visualization.draw_geometries([pts, cyl_orientation_open3d, mesh_frame, pts_cyl]) #

            a = 2 * projected_points
            a[:, 2] = 1
            pp_2d = np.delete(projected_points, 2, axis=1)
            b = np.linalg.norm(pp_2d, axis=1) ** 2
            cx, cy, comp = np.dot(np.linalg.inv(np.dot(a.T, a)), np.dot(a.T, b))
            radius = math.sqrt(comp + cx ** 2 + cy ** 2)
            print("radius ", radius)
            # a = 1/0
            io.write_json(self.output_file(), {"calculated_radius": radius, "measured_radius": measures["radius"]})
