#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/float32_multi_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <quadrotor_msgs/msg/beta_flight_states.hpp>
#include <quadrotor_msgs/msg/beta_flight_onboard_control.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <quadrotor_msgs/msg/so3_command.hpp>
#include <quadrotor_msgs/msg/trpy_command.hpp>

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
#include <algorithm>
#include <atomic>

extern "C" {
#include "pi_ros2_interface/pi-protocol.h"
#include "pi_ros2_interface/pi-messages.h"
}

// --- Visualization yaw offset (kept as-is) ---
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

    // Independent rates
    declare_parameter<double>("rx_rate_hz", 1000.0);
    declare_parameter<double>("tx_rate_hz", 500.0);
    declare_parameter<double>("beta_states_rate_hz", 500.0);
    declare_parameter<double>("onboard_control_rate_hz", 500.0);

    // false -> SO3 (same as previous "so3")
    // true  -> TRPY (same as previous "trpy")
    declare_parameter<bool>("control_type", false);

    serial_device_ = get_parameter("serial_device").as_string();
    int baud = get_parameter("baud").as_int();

    const double rx_rate_hz = get_parameter("rx_rate_hz").as_double();
    const double tx_rate_hz = get_parameter("tx_rate_hz").as_double();
    const double beta_states_rate_hz = get_parameter("beta_states_rate_hz").as_double();
    const double onboard_control_rate_hz = get_parameter("onboard_control_rate_hz").as_double();

    // Read control_type once at startup
    control_type_trpy_ = get_parameter("control_type").as_bool();

    // Set which mode we request from FC, based on your existing convention:
    // - "SO3" path was used for attitude
    // - "TRPY" path was used for acro
    if (!control_type_trpy_) { active_attitude_ = 1; active_acro_ = 0; }  // SO3
    else                     { active_attitude_ = 0; active_acro_ = 1; }  // TRPY

    // Open serial port
    fd_ = open_serial(serial_device_.c_str(), baud);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to open serial: " + serial_device_);
    }

    // Publishers: raw debug, filtered state, imu, onboard control (from FC)
    states_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("eagle4/states", 10);
    rc_pub_     = create_publisher<std_msgs::msg::Float32MultiArray>("eagle4/rc_attitude", 10);

    beta_pub_   = create_publisher<quadrotor_msgs::msg::BetaFlightStates>("eagle4/betaflight", 10);
    imu_pub_    = create_publisher<sensor_msgs::msg::Imu>("eagle4/imu", 10);

    onboard_control_pub_ =
      create_publisher<quadrotor_msgs::msg::BetaFlightOnboardControl>("eagle4/onboard", 10);

    // Subscribers:
    // We keep BOTH subscriptions, but each callback returns immediately unless it matches control_type.
    so3_command_sub_ = create_subscription<quadrotor_msgs::msg::SO3Command>(
      "eagle4/so3_cmd_2", 10,
      std::bind(&PiBridgeNode::so3_cmd_callback, this, std::placeholders::_1)
    );

    trpy_command_sub_ = create_subscription<quadrotor_msgs::msg::TRPYCommand>(
      "eagle4/trpy_cmd", 10,
      std::bind(&PiBridgeNode::trpy_cmd_callback, this, std::placeholders::_1)
    );

    // Used only for yaw alignment used in qDes transform
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "eagle4/odom", 10,
      std::bind(&PiBridgeNode::odom_callback, this, std::placeholders::_1)
    );

    memset(&parser_, 0, sizeof(parser_));

    // Defaults (ROS frame)
    set_safe_command_locked_unsafe(); // fills cmd_* (protected by cmd_mtx_)

    // last good state (ROS frame)
    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      have_valid_state_ = true;
      last_q_ros_ = Eigen::Quaterniond(1,0,0,0);
      last_w_ros_ = Eigen::Vector3d::Zero();
      last_w_dot_filtered_ros_ = Eigen::Vector3d::Zero();
      last_linear_acceleration_filtered_ros_ = Eigen::Vector3d::Zero();
      last_linear_acceleration_ros_ = Eigen::Vector3d::Zero();
      last_motor_ros_ = Eigen::Vector4d::Zero();
      last_state_stamp_ = this->now();
    }

    // desired setpoints decoded from FC onboard msg (ROS frame)
    {
      std::lock_guard<std::mutex> lk(sp_mtx_);
      have_valid_sp_ = false;
      last_q_sp_ros_ = Eigen::Quaterniond(1,0,0,0);
      last_w_sp_ros_ = Eigen::Vector3d::Zero();
      last_active_desired_thrust_ = 0.001;
      last_active_attitude_onboard_ = 0;
      last_active_acro_onboard_ = 0;
      last_active_offboard_ = 0;
      last_sp_stamp_ = this->now();
    }

    // RX timer
    rx_timer_ = create_wall_timer(
      std::chrono::microseconds((int64_t)(1e6 / std::max(1e-6, rx_rate_hz))),
      std::bind(&PiBridgeNode::rx_spin, this)
    );

    // TX timer
    tx_timer_ = create_wall_timer(
      std::chrono::microseconds((int64_t)(1e6 / std::max(1e-6, tx_rate_hz))),
      std::bind(&PiBridgeNode::tx_spin, this)
    );

    // Beta/IMU timer
    odom_timer_ = create_wall_timer(
      std::chrono::microseconds((int64_t)(1e6 / std::max(1e-6, beta_states_rate_hz))),
      std::bind(&PiBridgeNode::beta_spin, this)
    );

    // Onboard control publish timer
    onboard_control_timer_ = create_wall_timer(
      std::chrono::microseconds((int64_t)(1e6 / std::max(1e-6, onboard_control_rate_hz))),
      std::bind(&PiBridgeNode::onboard_control_spin, this)
    );

    RCLCPP_INFO(get_logger(),
      "PI bridge running. Serial=%s rx=%.1fHz tx=%.1fHz beta=%.1fHz onboard=%.1fHz control_type(trpy)=%s (send: att=%u acro=%u)",
      serial_device_.c_str(), rx_rate_hz, tx_rate_hz, beta_states_rate_hz, onboard_control_rate_hz,
      control_type_trpy_ ? "true" : "false",
      (unsigned)active_attitude_, (unsigned)active_acro_);
  }

  ~PiBridgeNode() override {
    shutdown_in_progress_.store(true);

    // Force safe command on shutdown
    {
      std::lock_guard<std::mutex> lk(cmd_mtx_);
      cmd_throttle_ = 0.001f;
      cmd_qw_ = 1.0f; cmd_qx_ = 0.0f; cmd_qy_ = 0.0f; cmd_qz_ = 0.0f;
      cmd_wx_ = 0.0f; cmd_wy_ = 0.0f; cmd_wz_ = 0.0f;
    }

    // Best-effort: send safety packet a few times before closing serial port
    for (int k = 0; k < 3; ++k) {
      tx_spin();
      ::usleep(2000); // 2ms
    }

    if (fd_ >= 0) close(fd_);
  }

private:
  // ---------------------------
  // Serial helpers
  // ---------------------------
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

  // ---------------------------
  // Frame / sanity helpers
  // ---------------------------
  tf2::Quaternion getTfYaw(const Eigen::Quaterniond& imu, const Eigen::Quaterniond& odom) {
    tf2::Quaternion imuTf(imu.x(), imu.y(), imu.z(), imu.w());
    tf2::Quaternion odomTf(odom.x(), odom.y(), odom.z(), odom.w());

    tf2::Matrix3x3(imuTf).getRPY(imuRoll_, imuPitch_, imuYaw_);
    tf2::Matrix3x3(odomTf).getRPY(odomRoll_, odomPitch_, odomYaw_);

    tf2::Quaternion imuTfYaw;
    tf2::Quaternion odomTfYaw;
    imuTfYaw.setRPY(0.0, 0.0, imuYaw_);
    odomTfYaw.setRPY(0.0, 0.0, odomYaw_);

    return imuTfYaw * odomTfYaw.inverse();
  }

  static bool quat_is_sane(const Eigen::Quaterniond& q_in) {
    const double w = q_in.w(), x = q_in.x(), y = q_in.y(), z = q_in.z();
    if (!std::isfinite(w) || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return false;
    const double max_abs = 5.0;
    if (std::abs(w) > max_abs || std::abs(x) > max_abs || std::abs(y) > max_abs || std::abs(z) > max_abs) return false;
    const double n2 = w*w + x*x + y*y + z*z;
    return (n2 >= 1e-6);
  }

  static bool vec_is_finite(const Eigen::Vector3d& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
  }

  static bool motors_is_finite(const Eigen::Vector4d& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z()) && std::isfinite(v.w());
  }

  // ---------------------------
  // Command callbacks (gated by control_type_trpy_)
  // ---------------------------
  void so3_cmd_callback(const quadrotor_msgs::msg::SO3Command::SharedPtr msg) {
    if (control_type_trpy_) return; // control_type=true -> TRPY only

    const Eigen::Vector3d fDes(msg->force.x, msg->force.y, msg->force.z);

    const Eigen::Quaterniond qDes(msg->orientation.w,
                                 msg->orientation.x,
                                 msg->orientation.y,
                                 msg->orientation.z);

    const Eigen::Vector3d wDes_ros(msg->angular_velocity.x,
                                  msg->angular_velocity.y,
                                  msg->angular_velocity.z);

    const tf2::Quaternion tfImuOdomYaw = getTfYaw(imuQ_, odomQ_);
    const Eigen::Quaterniond qYaw(tfImuOdomYaw.w(), tfImuOdomYaw.x(), tfImuOdomYaw.y(), tfImuOdomYaw.z());
    const Eigen::Quaterniond qDesTransformed = qYaw * qDes;

    const Eigen::Matrix3d rCur(odomQ_);
    const double throttle =
      fDes(0) * rCur(0, 2) + fDes(1) * rCur(1, 2) + fDes(2) * rCur(2, 2);

    std::lock_guard<std::mutex> lk(cmd_mtx_);
    cmd_throttle_ = (float)throttle;

    cmd_qw_ = (float)qDesTransformed.w();
    cmd_qx_ = (float)qDesTransformed.x();
    cmd_qy_ = (float)qDesTransformed.y();
    cmd_qz_ = (float)qDesTransformed.z();

    cmd_wx_ = (float)wDes_ros.x();
    cmd_wy_ = (float)wDes_ros.y();
    cmd_wz_ = (float)wDes_ros.z();
  }

  void trpy_cmd_callback(const quadrotor_msgs::msg::TRPYCommand::SharedPtr msg) {
    if (!control_type_trpy_) return; // control_type=false -> SO3 only

    const Eigen::Quaterniond qDes(msg->quaternion.w,
                                 msg->quaternion.x,
                                 msg->quaternion.y,
                                 msg->quaternion.z);

    const Eigen::Vector3d wDes_ros(msg->angular_velocity.x,
                                  msg->angular_velocity.y,
                                  msg->angular_velocity.z);

    const tf2::Quaternion tfImuOdomYaw = getTfYaw(imuQ_, odomQ_);
    const Eigen::Quaterniond qYaw(tfImuOdomYaw.w(), tfImuOdomYaw.x(), tfImuOdomYaw.y(), tfImuOdomYaw.z());
    const Eigen::Quaterniond qDesTransformed = qYaw * qDes;

    std::lock_guard<std::mutex> lk(cmd_mtx_);
    cmd_throttle_ = (float)msg->thrust;

    cmd_qw_ = (float)qDesTransformed.w();
    cmd_qx_ = (float)qDesTransformed.x();
    cmd_qy_ = (float)qDesTransformed.y();
    cmd_qz_ = (float)qDesTransformed.z();

    cmd_wx_ = (float)wDes_ros.x();
    cmd_wy_ = (float)wDes_ros.y();
    cmd_wz_ = (float)wDes_ros.z();
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    odomQ_.w() = msg->pose.pose.orientation.w;
    odomQ_.x() = msg->pose.pose.orientation.x;
    odomQ_.y() = msg->pose.pose.orientation.y;
    odomQ_.z() = msg->pose.pose.orientation.z;
  }

  // ---------------------------
  // RX: parse serial messages
  // ---------------------------
  void rx_spin() {
    uint8_t buf[256];
    const ssize_t n = ::read(fd_, buf, sizeof(buf));
    if (n <= 0) return;

    for (ssize_t i = 0; i < n; ++i) {
      const uint8_t id = piParse(&parser_, buf[i]);

      if (id == PI_MSG_EAGLE_STATES_ID) {
#if (PI_MODE & PI_RX) && (PI_MSG_EAGLE_STATES_MODE & PI_RX)
        if (piMsgEagleStatesRx) {
          std_msgs::msg::Float32MultiArray out;
          out.data.resize(21);
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

          out.data[11]  = piMsgEagleStatesRx->ax_filtered;
          out.data[12]  = piMsgEagleStatesRx->ay_filtered;
          out.data[13]  = piMsgEagleStatesRx->az_filtered;

          out.data[14]  = piMsgEagleStatesRx->wx_dot_filtered;
          out.data[15]  = piMsgEagleStatesRx->wy_dot_filtered;
          out.data[16]  = piMsgEagleStatesRx->wz_dot_filtered;

          out.data[17]  = piMsgEagleStatesRx->motor_0;
          out.data[18]  = piMsgEagleStatesRx->motor_1;
          out.data[19]  = piMsgEagleStatesRx->motor_2;
          out.data[20]  = piMsgEagleStatesRx->motor_3;

          states_pub_->publish(out);

          const Eigen::Quaterniond q_bf(piMsgEagleStatesRx->qw,
                                        piMsgEagleStatesRx->qx,
                                        piMsgEagleStatesRx->qy,
                                        piMsgEagleStatesRx->qz);

          const Eigen::Vector3d w_bf(piMsgEagleStatesRx->wx,
                                     piMsgEagleStatesRx->wy,
                                     piMsgEagleStatesRx->wz);

          const Eigen::Vector3d w_dot_filtered_bf(piMsgEagleStatesRx->wx_dot_filtered,
                                                  piMsgEagleStatesRx->wy_dot_filtered,
                                                  piMsgEagleStatesRx->wz_dot_filtered);

          const Eigen::Vector4d motors_velocity_raw(piMsgEagleStatesRx->motor_0,
                                                    piMsgEagleStatesRx->motor_1,
                                                    piMsgEagleStatesRx->motor_2,
                                                    piMsgEagleStatesRx->motor_3);

          const Eigen::Vector3d a_filtered_bf(piMsgEagleStatesRx->ax_filtered,
                                              piMsgEagleStatesRx->ay_filtered,
                                              piMsgEagleStatesRx->az_filtered);

          const Eigen::Vector3d a_bf(piMsgEagleStatesRx->ax,
                                     piMsgEagleStatesRx->ay,
                                     piMsgEagleStatesRx->az);

          if (!quat_is_sane(q_bf)) continue;

          Eigen::Quaterniond q_bf_n = q_bf;
          q_bf_n.normalize();

          Eigen::Quaterniond q_ros = Q_YAW_OFFSET * (NED_ENU_Q * q_bf_n * FRD_FLU_Q);
          if (!quat_is_sane(q_ros)) continue;

          if (!vec_is_finite(w_bf) ||
              !vec_is_finite(w_dot_filtered_bf) ||
              !vec_is_finite(a_filtered_bf) ||
              !vec_is_finite(a_bf) ||
              !motors_is_finite(motors_velocity_raw)) {
            continue;
          }

          q_ros.normalize();

          const Eigen::Matrix3d R_frd_flu = FRD_FLU_Q.toRotationMatrix();
          const Eigen::Vector3d w_ros               = R_frd_flu * w_bf;
          const Eigen::Vector3d w_dot_filtered_ros  = R_frd_flu * w_dot_filtered_bf;
          const Eigen::Vector3d a_filtered_ros      = R_frd_flu * a_filtered_bf;
          const Eigen::Vector3d a_ros               = R_frd_flu * a_bf;

          {
            std::lock_guard<std::mutex> lk(state_mtx_);
            last_q_ros_ = q_ros;
            imuQ_ = last_q_ros_;
            last_w_ros_ = w_ros;
            last_w_dot_filtered_ros_ = w_dot_filtered_ros;
            last_linear_acceleration_filtered_ros_ = a_filtered_ros;
            last_linear_acceleration_ros_ = a_ros;
            last_motor_ros_ = motors_velocity_raw;
            have_valid_state_ = true;
            last_state_stamp_ = this->now();
          }
        }
#endif
      }

      if (id == PI_MSG_EAGLE_ONBOARD_CONTROL_ID) {
#if (PI_MODE & PI_RX) && (PI_MSG_EAGLE_ONBOARD_CONTROL_MODE & PI_RX)
        if (piMsgEagleOnboardControlRx) {
          std_msgs::msg::Float32MultiArray out;
          out.data.resize(12);
          out.data[0] = (float)piMsgEagleOnboardControlRx->time_us;
          out.data[1] = piMsgEagleOnboardControlRx->throttle_d;

          out.data[2] = piMsgEagleOnboardControlRx->qw_d;
          out.data[3] = piMsgEagleOnboardControlRx->qx_d;
          out.data[4] = piMsgEagleOnboardControlRx->qy_d;
          out.data[5] = piMsgEagleOnboardControlRx->qz_d;

          out.data[6] = piMsgEagleOnboardControlRx->wx_d;
          out.data[7] = piMsgEagleOnboardControlRx->wy_d;
          out.data[8] = piMsgEagleOnboardControlRx->wz_d;

          out.data[9]  = piMsgEagleOnboardControlRx->active_offboard;
          out.data[10] = piMsgEagleOnboardControlRx->active_attitude;
          out.data[11] = piMsgEagleOnboardControlRx->active_acro;

          rc_pub_->publish(out);

          Eigen::Quaterniond q_sp_bf(piMsgEagleOnboardControlRx->qw_d,
                                     piMsgEagleOnboardControlRx->qx_d,
                                     piMsgEagleOnboardControlRx->qy_d,
                                     piMsgEagleOnboardControlRx->qz_d);

          Eigen::Vector3d w_sp_bf((double)piMsgEagleOnboardControlRx->wx_d,
                                  (double)piMsgEagleOnboardControlRx->wy_d,
                                  (double)piMsgEagleOnboardControlRx->wz_d);

          if (!quat_is_sane(q_sp_bf) || !vec_is_finite(w_sp_bf)) continue;

          q_sp_bf.normalize();
          Eigen::Quaterniond q_sp_ros = Q_YAW_OFFSET * (NED_ENU_Q * q_sp_bf * FRD_FLU_Q);
          if (!quat_is_sane(q_sp_ros)) continue;
          q_sp_ros.normalize();

          const Eigen::Vector3d w_sp_ros = FRD_FLU_Q.toRotationMatrix() * w_sp_bf;

          {
            std::lock_guard<std::mutex> lk(sp_mtx_);
            last_q_sp_ros_ = q_sp_ros;
            last_w_sp_ros_ = w_sp_ros;

            last_active_offboard_ = piMsgEagleOnboardControlRx->active_offboard;
            last_active_attitude_onboard_ = piMsgEagleOnboardControlRx->active_attitude;
            last_active_acro_onboard_ = piMsgEagleOnboardControlRx->active_acro;
            last_active_desired_thrust_ = piMsgEagleOnboardControlRx->throttle_d;

            have_valid_sp_ = true;
            last_sp_stamp_ = this->now();
          }
        }
#endif
      }
    }
  }

  // ---------------------------
  // Publish filtered state (betaflight msg + imu)
  // ---------------------------
  void beta_spin() {
    Eigen::Quaterniond q;
    Eigen::Vector3d w;
    Eigen::Vector3d a_filtered;
    Eigen::Vector3d a;
    Eigen::Vector3d w_dot_filtered;
    Eigen::Vector4d motors;

    {
      std::lock_guard<std::mutex> lk(state_mtx_);
      if (!have_valid_state_) return;
      q = last_q_ros_;
      w = last_w_ros_;
      a_filtered = last_linear_acceleration_filtered_ros_;
      a = last_linear_acceleration_ros_;
      w_dot_filtered = last_w_dot_filtered_ros_;
      motors = last_motor_ros_;
    }

    const auto stamp = this->now();

    quadrotor_msgs::msg::BetaFlightStates beta_msg;
    beta_msg.header.stamp = stamp;
    beta_msg.header.frame_id = "base_link";

    beta_msg.linear_acceleration.x = a.x();
    beta_msg.linear_acceleration.y = a.y();
    beta_msg.linear_acceleration.z = a.z();

    beta_msg.linear_acceleration_filtered.x = a_filtered.x();
    beta_msg.linear_acceleration_filtered.y = a_filtered.y();
    beta_msg.linear_acceleration_filtered.z = a_filtered.z();

    beta_msg.angular_velocity.x = w.x();
    beta_msg.angular_velocity.y = w.y();
    beta_msg.angular_velocity.z = w.z();

    beta_msg.quaternion.w = q.w();
    beta_msg.quaternion.x = q.x();
    beta_msg.quaternion.y = q.y();
    beta_msg.quaternion.z = q.z();

    beta_msg.angular_acceleration_filtered.x = w_dot_filtered.x();
    beta_msg.angular_acceleration_filtered.y = w_dot_filtered.y();
    beta_msg.angular_acceleration_filtered.z = w_dot_filtered.z();

    beta_msg.motor[0] = motors.x();
    beta_msg.motor[1] = motors.y();
    beta_msg.motor[2] = motors.z();
    beta_msg.motor[3] = motors.w();

    beta_pub_->publish(beta_msg);

    sensor_msgs::msg::Imu imu_msg;
    imu_msg.header.stamp = stamp;
    imu_msg.header.frame_id = "base_link";

    imu_msg.orientation.w = q.w();
    imu_msg.orientation.x = q.x();
    imu_msg.orientation.y = q.y();
    imu_msg.orientation.z = q.z();

    imu_msg.angular_velocity.x = w.x();
    imu_msg.angular_velocity.y = w.y();
    imu_msg.angular_velocity.z = w.z();

    imu_msg.linear_acceleration.x = a_filtered.x();
    imu_msg.linear_acceleration.y = a_filtered.y();
    imu_msg.linear_acceleration.z = a_filtered.z();

    imu_pub_->publish(imu_msg);
  }

  // ---------------------------
  // Publish onboard control (from FC) and throttled logs
  // ---------------------------
  void onboard_control_spin() {
    Eigen::Quaterniond qsp;
    Eigen::Vector3d wsp;
    uint8_t offboard_flag;
    uint8_t attitude_flag;
    uint8_t acro_flag;
    double desired_thrust;

    {
      std::lock_guard<std::mutex> lk(sp_mtx_);
      if (!have_valid_sp_) return;
      qsp = last_q_sp_ros_;
      wsp = last_w_sp_ros_;
      offboard_flag = last_active_offboard_;
      attitude_flag = last_active_attitude_onboard_;
      acro_flag = last_active_acro_onboard_;
      desired_thrust = last_active_desired_thrust_;
    }

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Onboard flags: offboard=%u attitude=%u acro=%u",
      (unsigned)offboard_flag, (unsigned)attitude_flag, (unsigned)acro_flag
    );

    quadrotor_msgs::msg::BetaFlightOnboardControl onboard_msg;
    const auto stamp = this->now();
    onboard_msg.header.stamp = stamp;
    onboard_msg.header.frame_id = "base_link";

    onboard_msg.angular_velocity_desired.x = wsp.x();
    onboard_msg.angular_velocity_desired.y = wsp.y();
    onboard_msg.angular_velocity_desired.z = wsp.z();

    onboard_msg.quaternion_desired.w = qsp.w();
    onboard_msg.quaternion_desired.x = qsp.x();
    onboard_msg.quaternion_desired.y = qsp.y();
    onboard_msg.quaternion_desired.z = qsp.z();

    onboard_msg.active_offboard = offboard_flag;
    onboard_msg.active_attitude = attitude_flag;
    onboard_msg.active_acro = acro_flag;
    onboard_msg.thrust_desired = desired_thrust;

    onboard_control_pub_->publish(onboard_msg);
  }

  // ---------------------------
  // TX: only send commands when FC is in offboard, and mode matches.
  // ---------------------------
  void tx_spin() {
    instance_ = this;

#if (PI_MODE & PI_TX) && (PI_MSG_EAGLE_OFFBOARD_MODE & PI_TX)
    uint8_t offboard_flag = 1;
    {
      std::lock_guard<std::mutex> lk(sp_mtx_);
      offboard_flag = last_active_offboard_;
    }

    const bool in_offboard   = (offboard_flag != 0);

    if (!in_offboard) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "FC not in offboard (offboard=%u). Not sending setpoints.",
        (unsigned)offboard_flag
      );
      return;
    }

    float cmd_throttle;
    float cmd_qw, cmd_qx, cmd_qy, cmd_qz;
    float cmd_wx, cmd_wy, cmd_wz;
    {
      std::lock_guard<std::mutex> lk(cmd_mtx_);
      cmd_throttle = cmd_throttle_;
      cmd_qw = cmd_qw_; cmd_qx = cmd_qx_; cmd_qy = cmd_qy_; cmd_qz = cmd_qz_;
      cmd_wx = cmd_wx_; cmd_wy = cmd_wy_; cmd_wz = cmd_wz_;
    }

    piMsgEagleOffboardTx.id  = PI_MSG_EAGLE_OFFBOARD_ID;
    piMsgEagleOffboardTx.len = PI_MSG_EAGLE_OFFBOARD_PAYLOAD_LEN;

    piMsgEagleOffboardTx.time_us = 0;
    piMsgEagleOffboardTx.throttle_d = cmd_throttle;
    piMsgEagleOffboardTx.active_attitude = active_attitude_;
    piMsgEagleOffboardTx.active_acro     = active_acro_;

    Eigen::Quaterniond q_cmd_ros(cmd_qw, cmd_qx, cmd_qy, cmd_qz);

    if (quat_is_sane(q_cmd_ros)) {
      q_cmd_ros.normalize();

      Eigen::Quaterniond q_cmd_bf =
          NED_ENU_Q.inverse() *
          Q_YAW_OFFSET.inverse() *
          q_cmd_ros *
          FRD_FLU_Q.inverse();
      q_cmd_bf.normalize();

      piMsgEagleOffboardTx.qw_d = (float)q_cmd_bf.w();
      piMsgEagleOffboardTx.qx_d = (float)q_cmd_bf.x();
      piMsgEagleOffboardTx.qy_d = (float)q_cmd_bf.y();
      piMsgEagleOffboardTx.qz_d = (float)q_cmd_bf.z();

      const Eigen::Vector3d w_cmd_ros((double)cmd_wx, (double)cmd_wy, (double)cmd_wz);
      const Eigen::Matrix3d R_frd_flu = FRD_FLU_Q.toRotationMatrix();      // FRD -> FLU
      const Eigen::Vector3d w_cmd_bf  = R_frd_flu.transpose() * w_cmd_ros; // FLU -> FRD

      piMsgEagleOffboardTx.wx_d = (float)w_cmd_bf.x();
      piMsgEagleOffboardTx.wy_d = (float)w_cmd_bf.y();
      piMsgEagleOffboardTx.wz_d = (float)w_cmd_bf.z();
    } else {
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Invalid command quaternion. Sending safe setpoints."
      );

      piMsgEagleOffboardTx.throttle_d = 0.001f;
      piMsgEagleOffboardTx.qw_d = 1.0f;
      piMsgEagleOffboardTx.qx_d = 0.0f;
      piMsgEagleOffboardTx.qy_d = 0.0f;
      piMsgEagleOffboardTx.qz_d = 0.0f;

      piMsgEagleOffboardTx.wx_d = 0.0f;
      piMsgEagleOffboardTx.wy_d = 0.0f;
      piMsgEagleOffboardTx.wz_d = 0.0f;
    }

    piSendMsg((void*)&piMsgEagleOffboardTx, serial_writer);
#endif
  }

  // ---------------------------
  // Safety helper (constructor use)
  // ---------------------------
  void set_safe_command_locked_unsafe() {
    std::lock_guard<std::mutex> lk(cmd_mtx_);
    cmd_throttle_ = 0.001f;
    cmd_qw_ = 1.0f; cmd_qx_ = 0.0f; cmd_qy_ = 0.0f; cmd_qz_ = 0.0f;
    cmd_wx_ = 0.0f; cmd_wy_ = 0.0f; cmd_wz_ = 0.0f;
  }

private:
  std::string serial_device_;
  int fd_{-1};
  pi_parse_states_t parser_{};

  // Publishers
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr states_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr rc_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<quadrotor_msgs::msg::BetaFlightStates>::SharedPtr beta_pub_;
  rclcpp::Publisher<quadrotor_msgs::msg::BetaFlightOnboardControl>::SharedPtr onboard_control_pub_;

  // Subscribers
  rclcpp::Subscription<quadrotor_msgs::msg::SO3Command>::SharedPtr so3_command_sub_;
  rclcpp::Subscription<quadrotor_msgs::msg::TRPYCommand>::SharedPtr trpy_command_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // Timers
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr tx_timer_;
  rclcpp::TimerBase::SharedPtr odom_timer_;
  rclcpp::TimerBase::SharedPtr onboard_control_timer_;

  // NEW: control type (fixed at startup)
  bool control_type_trpy_{false};

  // Command to FC (ROS frame): protected by cmd_mtx_
  std::mutex cmd_mtx_;
  float cmd_throttle_{0.001f};
  float cmd_qw_{1.0f}, cmd_qx_{0.0f}, cmd_qy_{0.0f}, cmd_qz_{0.0f};
  float cmd_wx_{0.0f}, cmd_wy_{0.0f}, cmd_wz_{0.0f};

  // Instance pointer for serial_writer
  static PiBridgeNode* instance_;

  // Quaternion odometry / yaw alignment
  Eigen::Quaterniond odomQ_{1,0,0,0};
  Eigen::Quaterniond imuQ_{1,0,0,0};
  double imuRoll_{0.0}, imuPitch_{0.0}, imuYaw_{0.0};
  double odomRoll_{0.0}, odomPitch_{0.0}, odomYaw_{0.0};

  // Last valid state (ROS frame)
  std::mutex state_mtx_;
  bool have_valid_state_{false};
  Eigen::Quaterniond last_q_ros_{1,0,0,0};
  Eigen::Vector3d last_w_ros_{0,0,0};
  Eigen::Vector3d last_w_dot_filtered_ros_{0,0,0};
  Eigen::Vector3d last_linear_acceleration_filtered_ros_{0,0,0};
  Eigen::Vector3d last_linear_acceleration_ros_{0,0,0};
  Eigen::Vector4d last_motor_ros_{0,0,0,0};
  rclcpp::Time last_state_stamp_{0, 0, RCL_ROS_TIME};

  // Desired setpoints decoded from FC onboard control message (ROS frame)
  std::mutex sp_mtx_;
  bool have_valid_sp_{false};
  Eigen::Quaterniond last_q_sp_ros_{1,0,0,0};
  Eigen::Vector3d last_w_sp_ros_{0,0,0};
  double last_active_desired_thrust_{0.001};
  uint8_t last_active_attitude_onboard_{0};
  uint8_t last_active_acro_onboard_{0};
  uint8_t last_active_offboard_{0};
  rclcpp::Time last_sp_stamp_{0, 0, RCL_ROS_TIME};

  // Flags we send to FC (requested command mode)
  uint8_t active_attitude_{0};
  uint8_t active_acro_{0};

  std::atomic<bool> shutdown_in_progress_{false};
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
