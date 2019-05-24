
// Self Header
#include <fileio/result_writer.h>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

// Original
#include <common/container_util.h>

namespace simple_sfm {
namespace fileio {
namespace {

pcl::PointXYZRGB EigenToPclPointXYZRGB(const Eigen::Vector3d &translation,
                                       const Eigen::Vector3d &color) {
  pcl::PointXYZRGB pnt(color(0), color(1), color(2));
  pnt.x = translation(0);
  pnt.y = translation(1);
  pnt.z = translation(2);
  return pnt;
}

pcl::PointXYZ EigenToPclPointXYZ(const Eigen::Vector3d &translation) {
  pcl::PointXYZ pnt;
  pnt.x = translation(0);
  pnt.y = translation(1);
  pnt.z = translation(2);
  return pnt;
}

std::vector<Eigen::Vector3d> CameraFrustumVertices(double base_length,
                                                   double height_base_ratio) {
  float half_base = base_length / 2.0;
  float height = base_length * height_base_ratio;
  std::vector<Eigen::Vector3d> vertices_in_cam = {
      Eigen::Vector3d(0.0, 0.0, 0.0),
      Eigen::Vector3d(half_base, -half_base, height),
      Eigen::Vector3d(half_base, half_base, height),
      Eigen::Vector3d(-half_base, half_base, height),
      Eigen::Vector3d(-half_base, -half_base, height)};
  return vertices_in_cam;
}

std::vector<std::vector<uint32_t> > CameraFrustomVertexConnections(
    uint32_t cam_idx) {
  static const uint32_t VERTEX_NUM = 5;
  std::vector<uint32_t> v(VERTEX_NUM);
  std::iota(v.begin(), v.end(), VERTEX_NUM * cam_idx);

  return {{v[0], v[1], v[2]}, {v[0], v[1], v[4]}, {v[0], v[4], v[3]},
          {v[0], v[3], v[2]}, {v[1], v[2], v[4]}, {v[3], v[2], v[4]}};
}

template <typename RTYPE, typename TTYPE, typename PNTTYPE>
std::vector<PNTTYPE> TransformPoints(const RTYPE &R, const TTYPE &T,
                                     const std::vector<PNTTYPE> &pnts) {
  std::vector<PNTTYPE> transformed(pnts.size());

  std::transform(pnts.begin(), pnts.end(), transformed.begin(),
                 [&](const Eigen::Vector3d &pnt) -> Eigen::Vector3d {
                   return R * pnt + T;
                 });

  return transformed;
}

std::pair<pcl::PointXYZ, pcl::PointXYZ> ComposeCameraViewingRay(
    const Eigen::Matrix3d &cam_rotation, const Eigen::Vector3d &cam_translation,
    const double length) {
  std::vector<Eigen::Vector3d> pnts;
  {
    Eigen::Matrix3d R_cam_2_glob = cam_rotation.transpose();
    Eigen::Vector3d T_cam_2_glob = R_cam_2_glob * -cam_translation;
    Eigen::Vector3d start_pnt_in_cam(0, 0, 2);
    Eigen::Vector3d end_pnt_in_cam(0, 0, 2 + length);
    pnts = TransformPoints(
        R_cam_2_glob, T_cam_2_glob,
        std::vector<Eigen::Vector3d>({start_pnt_in_cam, end_pnt_in_cam}));
  }
  std::pair<pcl::PointXYZ, pcl::PointXYZ> line;
  line.first = EigenToPclPointXYZ(pnts[0]);
  line.second = EigenToPclPointXYZ(pnts[1]);
  return line;
}

void ComposeCameraFrustomAsLines(
    const Eigen::Matrix3d &cam_rotation, const Eigen::Vector3d &cam_translation,
    std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ> > &lines) {
  // Calculate Mesh Position in Global Coord.
  std::vector<Eigen::Vector3d> vertices_in_global;
  {
    std::vector<Eigen::Vector3d> vertices_in_cam =
        CameraFrustumVertices(1.0, 1.0);
    Eigen::Matrix3d R_cam_2_glob = cam_rotation.transpose();
    Eigen::Vector3d T_cam_2_glob = R_cam_2_glob * -cam_translation;
    vertices_in_global =
        TransformPoints(R_cam_2_glob, T_cam_2_glob, vertices_in_cam);
  }

  {
    std::vector<pcl::PointXYZ> vertices_as_pclpnt;
    for (const auto &pnt : vertices_in_global) {
      vertices_as_pclpnt.push_back(EigenToPclPointXYZ(pnt));
    }

    lines.resize(8);
    lines[0] = std::make_pair(vertices_as_pclpnt[0], vertices_as_pclpnt[1]);
    lines[1] = std::make_pair(vertices_as_pclpnt[0], vertices_as_pclpnt[2]);
    lines[2] = std::make_pair(vertices_as_pclpnt[0], vertices_as_pclpnt[3]);
    lines[3] = std::make_pair(vertices_as_pclpnt[0], vertices_as_pclpnt[4]);
    lines[4] = std::make_pair(vertices_as_pclpnt[1], vertices_as_pclpnt[4]);
    lines[5] = std::make_pair(vertices_as_pclpnt[1], vertices_as_pclpnt[2]);
    lines[6] = std::make_pair(vertices_as_pclpnt[2], vertices_as_pclpnt[3]);
    lines[7] = std::make_pair(vertices_as_pclpnt[3], vertices_as_pclpnt[4]);
  }
}

void ComposeCameraFrustumAsPolygonMesh(const Eigen::Matrix3d &cam_rotation,
                                       const Eigen::Vector3d &cam_translation,
                                       const Eigen::Vector3d &color,
                                       double rep_length,
                                       pcl::PolygonMesh &cam_mesh) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PolygonMesh mesh;
  mesh.polygons.reserve(CameraFrustomVertexConnections(0).size());

  // Calculate Mesh Position in Global Coord.
  std::vector<Eigen::Vector3d> vertices_in_global;
  {
    std::vector<Eigen::Vector3d> vertices_in_cam =
        CameraFrustumVertices(rep_length / 2.0, 1.0);
    Eigen::Matrix3d R_cam_2_glob = cam_rotation.transpose();
    Eigen::Vector3d T_cam_2_glob = R_cam_2_glob * -cam_translation;
    vertices_in_global =
        TransformPoints(R_cam_2_glob, T_cam_2_glob, vertices_in_cam);
  }

  // Generate Mesh.
  {
    for (auto &pnt : vertices_in_global) {
      cloud->push_back(EigenToPclPointXYZRGB(pnt, color));
    }
  }

  // Generate Connections between Vertices.
  {
    std::vector<std::vector<uint32_t> > vertex_connections =
        CameraFrustomVertexConnections(0);
    for (auto &connection : vertex_connections) {
      pcl::Vertices v;
      v.vertices = connection;
      mesh.polygons.push_back(v);
    }
  }

  pcl::toPCLPointCloud2(*cloud, cam_mesh.cloud);
  cam_mesh.polygons = mesh.polygons;
}

void MergePolygonMesh(const pcl::PolygonMesh &mesh1,
                      const pcl::PolygonMesh &mesh2, pcl::PolygonMesh &merged) {
  pcl::PolygonMesh tmp_merged;
  size_t size1 = mesh1.cloud.width * mesh1.cloud.height;
  size_t size2 = mesh2.cloud.width * mesh2.cloud.height;
  pcl::concatenatePointCloud(mesh1.cloud, mesh2.cloud, tmp_merged.cloud);

  std::vector<pcl::Vertices> vertices2(mesh2.polygons.size());
  std::transform(mesh2.polygons.begin(), mesh2.polygons.end(),
                 vertices2.begin(),
                 [&](const pcl::Vertices &v_old) -> pcl::Vertices {
                   pcl::Vertices v_new;
                   v_new.vertices = v_old.vertices;
                   for_each(v_new.vertices.begin(), v_new.vertices.end(),
                            [&](uint32_t &num) { num += size1; });
                   return v_new;
                 });

  tmp_merged.polygons.reserve(size1 + size2);
  tmp_merged.polygons.insert(tmp_merged.polygons.end(), mesh1.polygons.begin(),
                             mesh1.polygons.end());
  tmp_merged.polygons.insert(tmp_merged.polygons.end(), vertices2.begin(),
                             vertices2.end());
  merged = tmp_merged;
}

}  // namespace

bool SavePointCloudToPCDFile(const std::string &filepath,
                             const common::vec2d<cv::KeyPoint> &key_points,
                             const common::vec1d<cv::Mat> &original_imgs,
                             const common::vec1d<common::CloudPoint> &points) {
  common::vec1d<pcl::PointXYZRGBA> pcl_pnts;
  common::container_util::ConvertCloudPointListToPCLPointXYZRGBAList(
      key_points, original_imgs, points, pcl_pnts);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p_cloud(
      new pcl::PointCloud<pcl::PointXYZRGBA>);
  p_cloud->reserve(pcl_pnts.size());
  p_cloud->insert(p_cloud->begin(), pcl_pnts.begin(), pcl_pnts.end());

  pcl::io::savePCDFileBinary(filepath, *p_cloud);

  return true;
}

bool SaveCameraPosesToPLYFile(const std::string &ply_filepath,
                              const common::vec1d<cv::Matx34d> &poses) {
  pcl::PolygonMesh cam_meshes;

  Eigen::Vector3d T0(poses[0](0, 3), poses[0](1, 3), poses[0](2, 3));
  Eigen::Vector3d T1(poses[1](0, 3), poses[1](1, 3), poses[1](2, 3));
  double rep_length = (T0 - T1).norm();

  for (const auto &pose : poses) {
    Eigen::Matrix3d R;
    R << pose(0, 0), pose(0, 1), pose(0, 2), pose(1, 0), pose(1, 1), pose(1, 2),
        pose(2, 0), pose(2, 1), pose(2, 2);
    Eigen::Vector3d T;
    T << pose(0, 3), pose(1, 3), pose(2, 3);

    pcl::PolygonMesh cam_mesh;
    ComposeCameraFrustumAsPolygonMesh(R, T, Eigen::Vector3d(255, 0, 0),
                                      rep_length, cam_mesh);
    MergePolygonMesh(cam_meshes, cam_mesh, cam_meshes);
  }

  pcl::io::savePLYFileBinary(ply_filepath, cam_meshes);

  return true;
}

}  // namespace fileio
}  // namespace simple_sfm
