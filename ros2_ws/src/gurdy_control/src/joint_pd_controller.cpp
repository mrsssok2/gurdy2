// JointPDController: the SAME joint PD law that MuJoCo applies, as a ros2_control controller.
//
//   tau_i = clamp( kp * (q_ref_i - q_i) - kd * qdot_i , -effort_limit, +effort_limit )
//
// Claims the `effort` command interface and `position` + `velocity` state interfaces of each
// joint. References arrive on ~/commands (std_msgs/Float64MultiArray, one value per joint in the
// `joints` parameter order). On activation the reference is the current position, so the robot
// holds still until the first command. Gains come from robot.yaml (the launch file generates the
// parameter file), never typed here.
#include <algorithm>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace gurdy_control
{
using CallbackReturn = controller_interface::CallbackReturn;
using controller_interface::interface_configuration_type;
using controller_interface::InterfaceConfiguration;

class JointPDController : public controller_interface::ControllerInterface
{
public:
  CallbackReturn on_init() override
  {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<double>("kp", 0.0);
    auto_declare<double>("kd", 0.0);
    auto_declare<double>("effort_limit", 0.0);
    return CallbackReturn::SUCCESS;
  }

  InterfaceConfiguration command_interface_configuration() const override
  {
    InterfaceConfiguration cfg{interface_configuration_type::INDIVIDUAL, {}};
    for (const auto & j : joints_) cfg.names.push_back(j + "/effort");
    return cfg;
  }

  InterfaceConfiguration state_interface_configuration() const override
  {
    InterfaceConfiguration cfg{interface_configuration_type::INDIVIDUAL, {}};
    for (const auto & j : joints_) {
      cfg.names.push_back(j + "/position");
      cfg.names.push_back(j + "/velocity");
    }
    return cfg;
  }

  CallbackReturn on_configure(const rclcpp_lifecycle::State &) override
  {
    auto node = get_node();
    joints_ = node->get_parameter("joints").as_string_array();
    kp_ = node->get_parameter("kp").as_double();
    kd_ = node->get_parameter("kd").as_double();
    effort_limit_ = node->get_parameter("effort_limit").as_double();
    if (joints_.empty() || kp_ <= 0.0 || kd_ < 0.0 || effort_limit_ <= 0.0) {
      RCLCPP_ERROR(node->get_logger(), "need joints, kp > 0, kd >= 0, effort_limit > 0");
      return CallbackReturn::ERROR;
    }
    const size_t n = joints_.size();
    q_ref_.assign(n, 0.0);
    pending_.assign(n, 0.0);
    has_pending_ = false;
    sub_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/commands", rclcpp::SystemDefaultsQoS(),
      [this, n](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() != n) {
          RCLCPP_WARN_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), 2000,
            "command has %zu values, expected %zu - ignored", msg->data.size(), n);
          return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = msg->data;
        has_pending_ = true;
      });
    RCLCPP_INFO(
      node->get_logger(), "configured %zu joints, kp=%.3f kd=%.3f effort_limit=%.3f", n, kp_, kd_,
      effort_limit_);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State &) override
  {
    // Map interfaces by name (do not rely on the order the controller manager hands them over).
    const size_t n = joints_.size();
    pos_idx_.assign(n, SIZE_MAX);
    vel_idx_.assign(n, SIZE_MAX);
    cmd_idx_.assign(n, SIZE_MAX);
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < state_interfaces_.size(); ++k) {
        const auto & si = state_interfaces_[k];
        if (si.get_prefix_name() != joints_[i]) continue;
        if (si.get_interface_name() == "position") pos_idx_[i] = k;
        if (si.get_interface_name() == "velocity") vel_idx_[i] = k;
      }
      for (size_t k = 0; k < command_interfaces_.size(); ++k) {
        if (command_interfaces_[k].get_prefix_name() == joints_[i]) cmd_idx_[i] = k;
      }
      if (pos_idx_[i] == SIZE_MAX || vel_idx_[i] == SIZE_MAX || cmd_idx_[i] == SIZE_MAX) {
        RCLCPP_ERROR(get_node()->get_logger(), "interfaces missing for joint %s", joints_[i].c_str());
        return CallbackReturn::ERROR;
      }
      q_ref_[i] = state_interfaces_[pos_idx_[i]].get_optional().value_or(0.0);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    has_pending_ = false;
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override
  {
    for (auto & c : command_interfaces_) (void)c.set_value(0.0);
    return CallbackReturn::SUCCESS;
  }

  controller_interface::return_type update(const rclcpp::Time &, const rclcpp::Duration &) override
  {
    {  // never block the real-time loop: take a new reference only if the lock is free
      std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
      if (lock.owns_lock() && has_pending_) {
        q_ref_ = pending_;
        has_pending_ = false;
      }
    }
    for (size_t i = 0; i < joints_.size(); ++i) {
      const auto q = state_interfaces_[pos_idx_[i]].get_optional();
      const auto qd = state_interfaces_[vel_idx_[i]].get_optional();
      if (!q || !qd) continue;  // keep previous command if state read failed
      double tau = kp_ * (q_ref_[i] - *q) - kd_ * (*qd);
      tau = std::clamp(tau, -effort_limit_, effort_limit_);
      (void)command_interfaces_[cmd_idx_[i]].set_value(tau);
    }
    return controller_interface::return_type::OK;
  }

private:
  std::vector<std::string> joints_;
  double kp_{0.0}, kd_{0.0}, effort_limit_{0.0};
  std::vector<double> q_ref_, pending_;
  bool has_pending_{false};
  std::mutex mutex_;
  std::vector<size_t> pos_idx_, vel_idx_, cmd_idx_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
};

}  // namespace gurdy_control

PLUGINLIB_EXPORT_CLASS(gurdy_control::JointPDController, controller_interface::ControllerInterface)
