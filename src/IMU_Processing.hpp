#include <Eigen/Eigen>
#include <cmath>
#include <common_lib.h>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <eigen_conversions/eigen_msg.h>
#include <fstream>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>
#include <thread>
#include <voxel_map/States.h>

/// *************Preconfiguration

#define MAX_INI_COUNT (200)

const bool time_list(PointType &x, PointType &y) {
  return (x.curvature < y.curvature);
};

/// *************IMU Process and undistortion
class ImuProcess {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4, 4) & T);
  void set_gyr_cov_scale(const V3D &scaler);
  void set_acc_cov_scale(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void Process(const MeasureGroup &meas, StatesGroup &state,
               PointCloudXYZI::Ptr &pcl_un_);

  ros::NodeHandle nh;
  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  M3D Lid_rot_to_IMU;
  V3D Lid_offset_to_IMU;
  double first_lidar_time;
  bool imu_en;

private:
  void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
  void UndistortPcl(const MeasureGroup &meas, StatesGroup &state_inout,
                    PointCloudXYZI &pcl_in_out);
  void only_propag(const MeasureGroup &meas, StatesGroup &state_inout,
                   PointCloudXYZI::Ptr &pcl_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D> v_rot_pcl_;

  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double time_last_scan_;
  int init_iter_num = 1;
  bool b_first_frame_ = true;
  bool imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1) {
  imu_en = true;
  init_iter_num = 1;
  cov_acc = V3D(0.1, 0.1, 0.1);
  cov_gyr = V3D(0.1, 0.1, 0.1);
  // old
  cov_acc_scale = V3D(1, 1, 1);
  cov_gyr_scale = V3D(1, 1, 1);

  cov_bias_gyr = V3D(0.1, 0.1, 0.1);
  cov_bias_acc = V3D(0.1, 0.1, 0.1);
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  Lid_offset_to_IMU = Zero3d;
  Lid_rot_to_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
  ROS_WARN("Reset ImuProcess");
  mean_acc = V3D(0, 0, -1.0);
  mean_gyr = V3D(0, 0, 0);
  angvel_last = Zero3d;
  imu_need_init_ = true;
  start_timestamp_ = -1;
  init_iter_num = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T) {
  Lid_offset_to_IMU = T.block<3, 1>(0, 3);
  Lid_rot_to_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl) {
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot) {
  Lid_offset_to_IMU = transl;
  Lid_rot_to_IMU = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler) {
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov_scale(const V3D &scaler) {
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout,
                          int &N) {
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
  V3D cur_acc, cur_gyr;

  if (b_first_frame_) {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
    // cout<<"init acc norm: "<<mean_acc.norm()<<endl;
  }

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc += (cur_acc - mean_acc) / N;
    mean_gyr += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N +
              (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) *
                  (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N +
              (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) *
                  (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N++;
  }

  state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;

  state_inout.rot_end =
      Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  state_inout.bias_g = mean_gyr;

  last_imu_ = meas.imu.back();
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas,
                              StatesGroup &state_inout,
                              PointCloudXYZI &pcl_out) {
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;

  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  const double &pcl_end_time =
      pcl_beg_time + pcl_out.points.back().curvature / double(1000);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  IMUpose.clear();
  // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end,
  // state.pos_end, state.rot_end));
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last,
                               state_inout.vel_end, state_inout.pos_end,
                               state_inout.rot_end));

  /*** forward propagation at each imu point ***/
  V3D acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end),
      pos_imu(state_inout.pos_end);
  M3D R_imu(state_inout.rot_end);
  MD(DIM_STATE, DIM_STATE) F_x, cov_w;

  double dt = 0;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < pcl_beg_time)
      continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 *
                   (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // #ifdef DEBUG_PRINT
    fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " "
             << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // #endif

    angvel_avr -= state_inout.bias_g;
    acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

    if (head->header.stamp.toSec() < pcl_beg_time) {
      dt = tail->header.stamp.toSec() - pcl_beg_time;
    } else {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    /* covariance propagation */
    M3D acc_avr_skew;
    M3D Exp_f = Exp(angvel_avr, dt);
    acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

    F_x.setIdentity();
    cov_w.setZero();

    F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
    F_x.block<3, 3>(0, 9) = -Eye3d * dt;
    // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
    F_x.block<3, 3>(3, 6) = Eye3d * dt;
    F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
    F_x.block<3, 3>(6, 12) = -R_imu * dt;
    F_x.block<3, 3>(6, 15) = Eye3d * dt;

    cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
    cov_w.block<3, 3>(6, 6) =
        R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
    cov_w.block<3, 3>(9, 9).diagonal() =
        cov_bias_gyr * dt * dt; // bias gyro covariance
    cov_w.block<3, 3>(12, 12).diagonal() =
        cov_bias_acc * dt * dt; // bias acc covariance

    state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

    /* propogation of IMU attitude */
    R_imu = R_imu * Exp_f;

    /* Specific acceleration (global frame) of IMU */
    acc_imu = R_imu * acc_avr + state_inout.gravity;

    /* propogation of IMU */
    pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

    /* velocity of IMU */
    vel_imu = vel_imu + acc_imu * dt;

    /* save the poses at each IMU measurements */
    angvel_last = angvel_avr;
    acc_s_last = acc_imu;
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(
        set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  state_inout.vel_end = vel_imu + note * acc_imu * dt;
  state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
  state_inout.pos_end =
      pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;

  auto pos_liD_e =
      state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;

  /*** undistort each lidar point (backward propagation) ***/
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    acc_imu << VEC_FROM_ARRAY(head->acc);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    angvel_avr << VEC_FROM_ARRAY(head->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
       * represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt +
               R_i * Lid_offset_to_IMU - pos_liD_e);

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

      /// save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin())
        break;
    }
  }
}

// constant velocity model
void ImuProcess::only_propag(const MeasureGroup &meas, StatesGroup &state_inout,
                             PointCloudXYZI::Ptr &pcl_out) {
  const double &pcl_beg_time = meas.lidar_beg_time;

  /*** sort point clouds by offset time ***/
  pcl_out = meas.lidar;
  const double &pcl_end_time =
      pcl_beg_time + pcl_out->points.back().curvature / double(1000);

  MD(DIM_STATE, DIM_STATE) F_x, cov_w;
  double dt = 0;

  if (b_first_frame_) {
    dt = 0.1;
    b_first_frame_ = false;
    time_last_scan_ = pcl_beg_time;
  } else {
    dt = pcl_beg_time - time_last_scan_;
    time_last_scan_ = pcl_beg_time;
  }

  /* covariance propagation */
  // M3D acc_avr_skew;
  M3D Exp_f = Exp(state_inout.bias_g, dt);

  F_x.setIdentity();
  cov_w.setZero();

  F_x.block<3, 3>(0, 0) = Exp(state_inout.bias_g, -dt);
  F_x.block<3, 3>(0, 9) = Eye3d * dt;
  F_x.block<3, 3>(3, 6) = Eye3d * dt;
  cov_w.block<3, 3>(9, 9).diagonal() =
      cov_gyr * dt * dt; // for omega in constant model
  cov_w.block<3, 3>(6, 6).diagonal() =
      cov_acc * dt * dt; // for velocity in constant model

  state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;
  state_inout.rot_end = state_inout.rot_end * Exp_f;
  state_inout.pos_end = state_inout.pos_end + state_inout.vel_end * dt;
}

void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat,
                         PointCloudXYZI::Ptr &cur_pcl_un_) {
  double t1, t2, t3;

  if (meas.imu.empty() && imu_en) {
    return;
  }
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_ && imu_en) {
    /// The very first lidar frame
    IMU_init(meas, stat, init_iter_num);

    imu_need_init_ = true;

    last_imu_ = meas.imu.back();

    if (init_iter_num > MAX_INI_COUNT) {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f "
               "%.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: "
               "%.8f %.8f %.8f",
               stat.gravity[0], stat.gravity[1], stat.gravity[2],
               mean_acc.norm(), cov_acc_scale[0], cov_acc_scale[1],
               cov_acc_scale[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0],
               cov_gyr[1], cov_gyr[2]);
      cov_acc = Eye3d * cov_acc_scale;
      cov_gyr = Eye3d * cov_gyr_scale;
      // cout<<"mean acc: "<<mean_acc<<" acc measures in word
      // frame:"<<state.rot_end.transpose()*mean_acc<<endl;
      ROS_INFO("IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f "
               "%.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: "
               "%.8f %.8f %.8f",
               stat.gravity[0], stat.gravity[1], stat.gravity[2],
               mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1],
               cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0],
               cov_gyr[1], cov_gyr[2]);
    }

    return;
  }

  /// Undistort points： the first point is assummed as the base frame
  /// Compensate lidar points with IMU rotation (with only rotation now)
  if (imu_en) {
    cout << "Use IMU" << endl;
    UndistortPcl(meas, stat, *cur_pcl_un_);
    last_imu_ = meas.imu.back();
  } else {
    cout << "No IMU, use constant velocity model" << endl;
    cov_acc = Eye3d * cov_acc_scale;
    cov_gyr = Eye3d * cov_gyr_scale;
    only_propag(meas, stat, cur_pcl_un_);
  }
}