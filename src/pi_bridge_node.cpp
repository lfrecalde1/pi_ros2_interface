#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <quadrotor_msgs/msg/so3_command.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <eigen3/Eigen/Geometry>
#include <mutex>
#include <cmath>

extern "C" {
#include "pi_ros2_interface/pi-protocol.h"
#include "pi_ros2_interface/pi-messages.h"
}

// --- Your visualization yaw offset (kept as-is) ---
static const double kYaw = -M_PI/2.0;
static const Eigen::Quaterniond Q_YAW_OFFSET(
    Eigen::AngleAxisd(kYaw, Eigen::Vector3d::UnitZ()));

// --- Frame transforms (same as you used) ---
static const Eigen::Quaterniond NED_ENU_Q(0.0, std::sqrt(0.5), std::sqrt(0.5), 0.0);
static const Eigen::Quaterniond FRD_FLU_Q(0.0, 1.0, 0.0, 0.0);

class PiBridgeNode : public rclcpp::Node {
public:
  PiBridgeNode() : Node("pi_bridge_node")
  {
    declare_parameter<std::string>("serial_device", "/dev/ttyTHS1");
    declare_parameter<int>("baud", 921600);
    declare_parameter<double>("tx_rate_hz", 500.0);

    // independent publish rates
    declare_parameter<double>("rx_rate_hz", 1000.0);
    declare_parameter<double>("odom_rate_hz", 500.0);

    // independent publish rate for desired attitude
    declare_parameter<double>("att_sp_rate_hz", 100.0);

    serial_device_ = get_parameter("serial_device").as_string();
    int baud = get_parameter("baud").as_int();
    double tx_rate_hz = get_parameter("tx_rate_hz").as_double();
    double rx_rate_hz = get_parameter("rx_rate_hz").as_double();
    double odom_rate_hz = get_parameter("odom_rate_hz").as_double();
    double att_sp_rate_hz = get_parameter("att_sp_rate_hz").as_double();

    fd_ = open_serial(serial_device_.c_str(), baud);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to open serial: " + serial_device_);
    }

    states_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("states", 10);
    rc_pub_     = create_publisher<std_msgs::msg::Float32MultiArray>("rc_attitude", 10);
    odom_pub_   = create_publisher<nav_msgs::msg::Odometry>("odom_betaflight", 10);

    // publish desired attitude for RViz2
    att_sp_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("att_sp", 10);

    so3_command_sub_ = create_subscription<quadrotor_msgs::msg::SO3Command>(
      "eagle4/so3_cmd_2", 10,
      std::bind(&PiBridgeNode::so3_cmd_callback, this, std::placeholders::_1)
    );

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "eagle4/odom", 10,
      std::bind(&PiBridgeNode::odom_callback, this, std::placeholders::_1)
    );


    memset(&parser_, 0, sizeof(parser_));

    // default command (ROS frame)
    cmd_active_ = 0.0f;
    cmd_throttle_ = 0.001f;
    cmd_qw_=1.0f; cmd_qx_=0.0f; cmd_qy_=0.0f; cmd_qz_=0.0f;
    cmd_wx_ = 0.0f;
    cmd_wy_ = 0.0f;
    cmd_wz_ = 0.0f;

    // default last good state
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      have_valid_state_ = true;
      last_q_ros_ = Eigen::Quaterniond(1,0,0,0);
      last_w_ros_ = Eigen::Vector3d::Zero();
      last_state_stamp_ = this->now();
    }

    // default desired attitude storage
    {
      std::lock_guard<std::mutex> lk(sp_mtx_);
      have_valid_sp_ = false;
      last_q_sp_ros_ = Eigen::Quaterniond(1,0,0,0);
      last_sp_stamp_ = this->now();
    }

    // RX timer
    auto rx_period_us = (int64_t)(1e6 / std::max(1e-6, rx_rate_hz));
    rx_timer_ = create_wall_timer(
      std::chrono::microseconds(rx_period_us),
      std::bind(&PiBridgeNode::rx_spin, this)
    );

    // TX timer
    auto tx_period_us = (int64_t)(1e6 / std::max(1e-6, tx_rate_hz));
    tx_timer_ = create_wall_timer(
      std::chrono::microseconds(tx_period_us),
      std::bind(&PiBridgeNode::tx_spin, this)
    );

    // ODOM timer
    auto odom_period_us = (int64_t)(1e6 / std::max(1e-6, odom_rate_hz));
    odom_timer_ = create_wall_timer(
      std::chrono::microseconds(odom_period_us),
      std::bind(&PiBridgeNode::odom_spin, this)
    );

    // Desired attitude timer
    auto sp_period_us = (int64_t)(1e6 / std::max(1e-6, att_sp_rate_hz));
    att_sp_timer_ = create_wall_timer(
      std::chrono::microseconds(sp_period_us),
      std::bind(&PiBridgeNode::att_sp_spin, this)
    );

    RCLCPP_INFO(get_logger(),
      "PI bridge running. Serial=%s rx=%.1fHz tx=%.1fHz odom=%.1fHz att_sp=%.1fHz",
      serial_device_.c_str(), rx_rate_hz, tx_rate_hz, odom_rate_hz, att_sp_rate_hz);
  }

  ~PiBridgeNode() override {
    if (fd_ >= 0) close(fd_);
  }

private:
  static speed_t to_termios_baud(int baud) {
    switch (baud) {
      case 115200: return B115200;
      case 230400: return B230400;
      case 460800: return B460800;
      case 921600: return B921600;
      default:     return B921600;
    }
  }

  int open_serial(const char* dev, int baud) {
    int fd = ::open(dev, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
      RCLCPP_ERROR(get_logger(), "open(%s) failed: %s", dev, strerror(errno));
      return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
      RCLCPP_ERROR(get_logger(), "tcgetattr failed: %s", strerror(errno));
      close(fd);
      return -1;
    }

    cfmakeraw(&tty);
    cfsetispeed(&tty, to_termios_baud(baud));
    cfsetospeed(&tty, to_termios_baud(baud));

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
      RCLCPP_ERROR(get_logger(), "tcsetattr failed: %s", strerror(errno));
      close(fd);
      return -1;
    }
    return fd;
  }

  static void serial_writer(uint8_t byte) {
    if (instance_ && instance_->fd_ >= 0) {
      (void)::write(instance_->fd_, &byte, 1);
    }
  }

  void so3_cmd_callback(const quadrotor_msgs::msg::SO3Command::SharedPtr msg) {

    const Eigen::Vector3d fDes(msg->force.x, msg->force.y, msg->force.z);
    const Eigen::Quaterniond qDes(msg->orientation.w, msg->orientation.x,
                                msg->orientation.y, msg->orientation.z);

    // get corrected yaw from odom
    const tf2::Quaternion tfImuOdomYaw = getTfYaw(imuQ, odomQ);


    Eigen::Quaterniond qDesTransformed =
        Eigen::Quaterniond(tfImuOdomYaw.w(), tfImuOdomYaw.x(), tfImuOdomYaw.y(),
                         tfImuOdomYaw.z()) * qDes;

    // check psi for stability
    const Eigen::Matrix3d rDes(qDes);
    const Eigen::Matrix3d rCur(odomQ);
    const float psi =
      0.5f * (3.0f - (rDes(0, 0) * rCur(0, 0) + rDes(1, 0) * rCur(1, 0) +
                      rDes(2, 0) * rCur(2, 0) + rDes(0, 1) * rCur(0, 1) +
                      rDes(1, 1) * rCur(1, 1) + rDes(2, 1) * rCur(2, 1) +
                      rDes(0, 2) * rCur(0, 2) + rDes(1, 2) * rCur(1, 2) +
                      rDes(2, 2) * rCur(2, 2)));

    if (psi > 1.0f)
        RCLCPP_WARN(this->get_logger(),
                "Psi(%f) > 1.0, orientation error is too large!", psi);

    double throttle =
        fDes(0) * rCur(0, 2) + fDes(1) * rCur(1, 2) + fDes(2) * rCur(2, 2);

    cmd_throttle_ = throttle;

    cmd_qw_ = qDes.w();
    cmd_qx_ = qDes.x();
    cmd_qy_ = qDes.y();
    cmd_qz_ = qDes.z();

    // update desired angular velocity
    cmd_wx_ = msg->angular_velocity.x;
    cmd_wy_ = msg->angular_velocity.y;
    cmd_wz_ = msg->angular_velocity.z;
    

      return;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {

    odomQ.w() = msg->pose.pose.orientation.w;
    odomQ.x() = msg->pose.pose.orientation.x;
    odomQ.y() = msg->pose.pose.orientation.y;
    odomQ.z() = msg->pose.pose.orientation.z;
    return;
  }


tf2::Quaternion getTfYaw(Eigen::Quaterniond imu,
                                          Eigen::Quaterniond odom) {
  // convert to tf2::quaternion
  tf2::Quaternion imuTf = tf2::Quaternion(imu.x(), imu.y(), imu.z(), imu.w());
  tf2::Quaternion odomTf =
      tf2::Quaternion(odom.x(), odom.y(), odom.z(), odom.w());

  tf2::Matrix3x3(imuTf).getRPY(imuRoll, imuPitch, imuYaw);
  tf2::Matrix3x3(odomTf).getRPY(odomRoll, odomPitch, odomYaw);

  // create only yaw tf2::Quat
  tf2::Quaternion imuTfYaw;
  tf2::Quaternion odomTfYaw;
  imuTfYaw.setRPY(0.0, 0.0, imuYaw);
  odomTfYaw.setRPY(0.0, 0.0, odomYaw);
  const tf2::Quaternion yawCorrection = imuTfYaw * odomTfYaw.inverse();

  return yawCorrection;
}

  static bool quat_is_sane(const Eigen::Quaterniond& q_in) {
    const double w = q_in.w(), x = q_in.x(), y = q_in.y(), z = q_in.z();
    if (!std::isfinite(w) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    const double max_abs = 5.0;
    if (std::abs(w) > max_abs || std::abs(x) > max_abs || std::abs(y) > max_abs || std::abs(z) > max_abs) return false;
    const double n2 = w*w + x*x + y*y + z*z;
    if (n2 < 1e-6) return false;
    return true;
  }

  static bool vec_is_finite(const Eigen::Vector3d& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
  }

  void rx_spin() {
    uint8_t buf[256];
    ssize_t n = ::read(fd_, buf, sizeof(buf));
    if (n <= 0) return;

    for (ssize_t i = 0; i < n; ++i) {
      uint8_t id = piParse(&parser_, buf[i]);

      if (id == PI_MSG_EAGLE_STATES_ID) {
#if (PI_MODE & PI_RX) && (PI_MSG_EAGLE_STATES_MODE & PI_RX)
        if (piMsgEagleStatesRx) {
          // raw publish
          std_msgs::msg::Float32MultiArray out;
          out.data.resize(11);
          out.data[0]  = (float)piMsgEagleStatesRx->time_us;
          out.data[1]  = piMsgEagleStatesRx->ax;
          out.data[2]  = piMsgEagleStatesRx->ay;
          out.data[3]  = piMsgEagleStatesRx->az;
          out.data[4]  = piMsgEagleStatesRx->wx;
          out.data[5]  = piMsgEagleStatesRx->wy;
          out.data[6]  = piMsgEagleStatesRx->wz;
          out.data[7]  = piMsgEagleStatesRx->qw;
          out.data[8]  = piMsgEagleStatesRx->qx;
          out.data[9]  = piMsgEagleStatesRx->qy;
          out.data[10] = piMsgEagleStatesRx->qz;
          states_pub_->publish(out);

          Eigen::Quaterniond q_bf(piMsgEagleStatesRx->qw,
                                  piMsgEagleStatesRx->qx,
                                  piMsgEagleStatesRx->qy,
                                  piMsgEagleStatesRx->qz);

          Eigen::Vector3d w_bf(piMsgEagleStatesRx->wx,
                               piMsgEagleStatesRx->wy,
                               piMsgEagleStatesRx->wz);

          if (quat_is_sane(q_bf)) {
            q_bf.normalize();

            // Betaflight -> ROS (same as your previous working version)
            Eigen::Quaterniond q_ros = Q_YAW_OFFSET * (NED_ENU_Q * q_bf * FRD_FLU_Q);

            if (quat_is_sane(q_ros) && vec_is_finite(w_bf)) {
              q_ros.normalize();

              // Body rates FRD -> FLU only (same as before)
              Eigen::Vector3d w_ros = FRD_FLU_Q.toRotationMatrix() * w_bf;

              std::lock_guard<std::mutex> lk(state_mtx_);
              last_q_ros_ = q_ros;
              imuQ.w() = last_q_ros_.w();
              imuQ.x() = last_q_ros_.x();
              imuQ.y() = last_q_ros_.y();
              imuQ.z() = last_q_ros_.z();
              last_w_ros_ = w_ros;
              have_valid_state_ = true;
              last_state_stamp_ = this->now();
            }
          }
        }
#endif
      }

      if (id == PI_MSG_EAGLE_RC_ATTITUDE_ID) {
#if (PI_MODE & PI_RX) && (PI_MSG_EAGLE_RC_ATTITUDE_MODE & PI_RX)
        if (piMsgEagleRcAttitudeRx) {
          std_msgs::msg::Float32MultiArray out;
          out.data.resize(9);
          out.data[0] = (float)piMsgEagleRcAttitudeRx->time_us;
          out.data[1] = piMsgEagleRcAttitudeRx->throttle_d;
          out.data[2] = piMsgEagleRcAttitudeRx->qw_d;
          out.data[3] = piMsgEagleRcAttitudeRx->qx_d;
          out.data[4] = piMsgEagleRcAttitudeRx->qy_d;
          out.data[5] = piMsgEagleRcAttitudeRx->qz_d;
          out.data[6] = piMsgEagleRcAttitudeRx->wx_d;
          out.data[7] = piMsgEagleRcAttitudeRx->wy_d;
          out.data[8] = piMsgEagleRcAttitudeRx->wz_d;
          rc_pub_->publish(out);

          // store desired attitude (Betaflight -> ROS) for RViz2
          Eigen::Quaterniond q_sp_bf(piMsgEagleRcAttitudeRx->qw_d,
                                     piMsgEagleRcAttitudeRx->qx_d,
                                     piMsgEagleRcAttitudeRx->qy_d,
                                     piMsgEagleRcAttitudeRx->qz_d);

          if (quat_is_sane(q_sp_bf)) {
            q_sp_bf.normalize();
            Eigen::Quaterniond q_sp_ros = Q_YAW_OFFSET * (NED_ENU_Q * q_sp_bf * FRD_FLU_Q);
            if (quat_is_sane(q_sp_ros)) {
              q_sp_ros.normalize();
              std::lock_guard<std::mutex> lk(sp_mtx_);
              last_q_sp_ros_ = q_sp_ros;
              have_valid_sp_ = true;
              last_sp_stamp_ = this->now();
            }
          }
        }
#endif
      }
    }
  }

  void odom_spin() {
    Eigen::Quaterniond q;
    Eigen::Vector3d w;

    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      if (!have_valid_state_) return;
      q = last_q_ros_;
      w = last_w_ros_;
    }

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = this->now();
    odom.header.frame_id = "world";
    odom.child_frame_id  = "base_link";

    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();

    odom.twist.twist.angular.x = w.x();
    odom.twist.twist.angular.y = w.y();
    odom.twist.twist.angular.z = w.z();

    odom_pub_->publish(odom);
  }

  void att_sp_spin() {
    Eigen::Quaterniond qsp;
    {
      std::lock_guard<std::mutex> lk(sp_mtx_);
      if (!have_valid_sp_) return;
      qsp = last_q_sp_ros_;
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = this->now();
    pose.header.frame_id = "world";

    pose.pose.orientation.w = qsp.w();
    pose.pose.orientation.x = qsp.x();
    pose.pose.orientation.y = qsp.y();
    pose.pose.orientation.z = qsp.z();

    att_sp_pub_->publish(pose);
  }

  void tx_spin() {
    instance_ = this;

#if (PI_MODE & PI_TX) && (PI_MSG_EAGLE_OFFBOARD_ATTITUDE_MODE & PI_TX)
    piMsgEagleOffboardAttitudeTx.id  = PI_MSG_EAGLE_OFFBOARD_ATTITUDE_ID;
    piMsgEagleOffboardAttitudeTx.len = PI_MSG_EAGLE_OFFBOARD_ATTITUDE_PAYLOAD_LEN;

    piMsgEagleOffboardAttitudeTx.time_us = 0;
    piMsgEagleOffboardAttitudeTx.active_offboard = (uint8_t)(cmd_active_ > 0.5f);
    piMsgEagleOffboardAttitudeTx.throttle_d = cmd_throttle_;

    // ----- CRITICAL FIX: ROS -> Betaflight conversion before sending -----
    // cmd_q* is assumed ROS frame (ENU world + FLU body), because that is what you command from ROS.
    Eigen::Quaterniond q_cmd_ros(cmd_qw_, cmd_qx_, cmd_qy_, cmd_qz_);

    if (quat_is_sane(q_cmd_ros)) {
      q_cmd_ros.normalize();

      // Inverse of your RX mapping:
      // q_ros = Q_YAW_OFFSET * (NED_ENU_Q * q_bf * FRD_FLU_Q)
      // => q_bf = NED_ENU_Q^{-1} * Q_YAW_OFFSET^{-1} * q_ros * FRD_FLU_Q^{-1}
      Eigen::Quaterniond q_cmd_bf =
          NED_ENU_Q.inverse() *
          Q_YAW_OFFSET.inverse() *
          q_cmd_ros *
          FRD_FLU_Q.inverse();

      q_cmd_bf.normalize();

      Eigen::Vector3d w_cmd_ros(cmd_wx_,
                                cmd_wy_,
                                cmd_wz_);
      Eigen::Vector3d w_cmd_bf = w_cmd_ros * FRD_FLU_Q.toRotationMatrix().transpose();

      piMsgEagleOffboardAttitudeTx.qw_d = (float)q_cmd_bf.w();
      piMsgEagleOffboardAttitudeTx.qx_d = (float)q_cmd_bf.x();
      piMsgEagleOffboardAttitudeTx.qy_d = (float)q_cmd_bf.y();
      piMsgEagleOffboardAttitudeTx.qz_d = (float)q_cmd_bf.z();

      piMsgEagleOffboardAttitudeTx.wx_d = w_cmd_bf.x();
      piMsgEagleOffboardAttitudeTx.wy_d = w_cmd_bf.y();
      piMsgEagleOffboardAttitudeTx.wz_d = w_cmd_bf.z();
    } else {
      // safe fallback
      piMsgEagleOffboardAttitudeTx.qw_d = 1.0f;
      piMsgEagleOffboardAttitudeTx.qx_d = 0.0f;
      piMsgEagleOffboardAttitudeTx.qy_d = 0.0f;
      piMsgEagleOffboardAttitudeTx.qz_d = 0.0f;

      piMsgEagleOffboardAttitudeTx.wx_d = 1.0f;
      piMsgEagleOffboardAttitudeTx.wy_d = 0.0f;
      piMsgEagleOffboardAttitudeTx.wz_d = 0.0f;
    }

    piSendMsg((void*)&piMsgEagleOffboardAttitudeTx, serial_writer);
#endif
  }

private:
  std::string serial_device_;
  int fd_{-1};

  pi_parse_states_t parser_{};

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr states_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr rc_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr att_sp_pub_;

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr cmd_sub_;
  rclcpp::Subscription<quadrotor_msgs::msg::SO3Command>::SharedPtr so3_command_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr tx_timer_;
  rclcpp::TimerBase::SharedPtr odom_timer_;
  rclcpp::TimerBase::SharedPtr att_sp_timer_;

  // command (ROS frame)
  float cmd_active_{1.0f};
  float cmd_throttle_{1.0f};
  float cmd_qw_{1.0f}, cmd_qx_{0.0f}, cmd_qy_{0.0f}, cmd_qz_{0.0f};
  float cmd_wx_{0.0f}, cmd_wy_{0.0f}, cmd_wz_{0.0f};

  static PiBridgeNode* instance_;
    
  // Quternion odometry
  Eigen::Quaterniond odomQ, imuQ;
  double imuRoll, imuPitch, imuYaw;
  double odomRoll, odomPitch, odomYaw;

  // last valid state (ROS frame) for odom
  std::mutex state_mtx_;
  bool have_valid_state_{false};
  Eigen::Quaterniond last_q_ros_{1,0,0,0};
  Eigen::Vector3d last_w_ros_{0,0,0};
  rclcpp::Time last_state_stamp_{0, 0, RCL_ROS_TIME};

  // desired attitude storage (ROS frame) for RViz2
  std::mutex sp_mtx_;
  bool have_valid_sp_{false};
  Eigen::Quaterniond last_q_sp_ros_{1,0,0,0};
  rclcpp::Time last_sp_stamp_{0, 0, RCL_ROS_TIME};
};

PiBridgeNode* PiBridgeNode::instance_ = nullptr;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PiBridgeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
