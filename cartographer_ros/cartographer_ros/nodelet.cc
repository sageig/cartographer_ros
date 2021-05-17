/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "Eigen/Core"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "cartographer/common/configuration_file_resolver.h"
#include "cartographer/common/fixed_ratio_sampler.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/port.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/map_builder.h"
#include "cartographer/mapping/map_builder_interface.h"
#include "cartographer/mapping/pose_extrapolator.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/metrics/register.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"
#include "cartographer_ros/map_builder_bridge.h"
#include "cartographer_ros/metrics/family_factory.h"
#include "cartographer_ros/msg_conversion.h"
#include "cartographer_ros/node_constants.h"
#include "cartographer_ros/node_options.h"
#include "cartographer_ros/sensor_bridge.h"
#include "cartographer_ros/tf_bridge.h"
#include "cartographer_ros/time_conversion.h"
#include "cartographer_ros/trajectory_options.h"
#include "cartographer_ros_msgs/FinishTrajectory.h"
#include "cartographer_ros_msgs/GetTrajectoryStates.h"
#include "cartographer_ros_msgs/ReadMetrics.h"
#include "cartographer_ros_msgs/StartTrajectory.h"
#include "cartographer_ros_msgs/StatusCode.h"
#include "cartographer_ros_msgs/StatusResponse.h"
#include "cartographer_ros_msgs/SubmapEntry.h"
#include "cartographer_ros_msgs/SubmapList.h"
#include "cartographer_ros_msgs/SubmapQuery.h"
#include "cartographer_ros_msgs/WriteState.h"
#include "geometry_msgs/PoseStamped.h"
#include "glog/logging.h"
#include "nav_msgs/Odometry.h"
#include "nodelet/nodelet.h"
#include "ros/ros.h"
#include "ros/serialization.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/MultiEchoLaserScan.h"
#include "sensor_msgs/NavSatFix.h"
#include "sensor_msgs/PointCloud2.h"
#include <swri_roscpp/parameters.h>
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/MarkerArray.h"


namespace cartographer_ros {

namespace carto = ::cartographer;

using carto::transform::Rigid3d;
using TrajectoryState =
    ::cartographer::mapping::PoseGraphInterface::TrajectoryState;

namespace {

std::string TrajectoryStateToString(const TrajectoryState trajectory_state) {
  switch (trajectory_state) {
    case TrajectoryState::ACTIVE:
      return "ACTIVE";
    case TrajectoryState::FINISHED:
      return "FINISHED";
    case TrajectoryState::FROZEN:
      return "FROZEN";
    case TrajectoryState::DELETED:
      return "DELETED";
  }
  return "";
}

}  // namespace

class CartographerNodelet : public nodelet::Nodelet
{
public:
  CartographerNodelet()
  {
    google::InitGoogleLogging("CartographerNodelet");
  }

  ~CartographerNodelet()
  { 
    FinishAllTrajectories();
    RunFinalOptimization();

    if (!save_state_filename_.empty()) {
      SerializeState(save_state_filename_,
                     true /* include_unfinished_submaps */);
    }
  }

  virtual void onInit()
  {
    node_handle_ = getNodeHandle();
    init_timer_ = node_handle_.createTimer(ros::Duration(1.0), &CartographerNodelet::Initialize, this, true);
  }

  void Initialize(const ros::TimerEvent& unused)
  {
    ros::NodeHandle pnh = getPrivateNodeHandle();
    std::string config_dir, config_basename, load_state_filename;
    bool collect_metrics, load_frozen_state, start_trajectory_with_default_topics;
    swri::param(pnh, "collect_metrics",collect_metrics, false);
    swri::param(pnh, "configuration_directory",config_dir, "");
    swri::param(pnh, "configuration_basename",config_basename, "");
    swri::param(pnh, "load_state_filename",load_state_filename, "");
    swri::param(pnh, "load_frozen_state",load_frozen_state, true);
    swri::param(pnh, "start_trajectory_with_default_topics",start_trajectory_with_default_topics, true);
    swri::param(pnh, "save_state_filename",save_state_filename_, "");

    if (config_dir.empty()) {
      ROS_ERROR("configuration_directory is missing");
    }
    if (config_basename.empty()) {
      ROS_ERROR("configuration_basename is missing");
    }

    if (collect_metrics) {
      metrics_registry_ = absl::make_unique<metrics::FamilyFactory>();
      carto::metrics::RegisterAllMetrics(metrics_registry_.get());
    }
    
    constexpr double kTfBufferCacheTimeInSeconds = 10.;
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(::ros::Duration(kTfBufferCacheTimeInSeconds));
    tf_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    TrajectoryOptions trajectory_options;
    std::tie(node_options_, trajectory_options) = LoadOptions(config_dir, config_basename);

    map_builder_ = cartographer::mapping::CreateMapBuilder(node_options_.map_builder_options);
    map_builder_bridge_ = std::make_shared<MapBuilderBridge>(node_options_, std::move(map_builder_), tf_buffer_.get());

    submap_list_publisher_ =
        node_handle_.advertise<::cartographer_ros_msgs::SubmapList>(
            kSubmapListTopic, kLatestOnlyPublisherQueueSize);
    trajectory_node_list_publisher_ =
        node_handle_.advertise<::visualization_msgs::MarkerArray>(
            kTrajectoryNodeListTopic, kLatestOnlyPublisherQueueSize);
    landmark_poses_list_publisher_ =
        node_handle_.advertise<::visualization_msgs::MarkerArray>(
            kLandmarkPosesListTopic, kLatestOnlyPublisherQueueSize);
    constraint_list_publisher_ =
        node_handle_.advertise<::visualization_msgs::MarkerArray>(
            kConstraintListTopic, kLatestOnlyPublisherQueueSize);
    if (node_options_.publish_tracked_pose) {
      tracked_pose_publisher_ =
          node_handle_.advertise<::geometry_msgs::PoseStamped>(
              kTrackedPoseTopic, kLatestOnlyPublisherQueueSize);
    }
    service_servers_.push_back(node_handle_.advertiseService(
        kSubmapCloudQueryServiceName, &CartographerNodelet::HandleSubmapCloudQuery, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kSubmapQueryServiceName, &CartographerNodelet::HandleSubmapQuery, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kTrajectoryQueryServiceName, &CartographerNodelet::HandleTrajectoryQuery, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kStartTrajectoryServiceName, &CartographerNodelet::HandleStartTrajectory, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kFinishTrajectoryServiceName, &CartographerNodelet::HandleFinishTrajectory, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kWriteStateServiceName, &CartographerNodelet::HandleWriteState, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kGetTrajectoryStatesServiceName, &CartographerNodelet::HandleGetTrajectoryStates, this));
    service_servers_.push_back(node_handle_.advertiseService(
        kReadMetricsServiceName, &CartographerNodelet::HandleReadMetrics, this));

    scan_matched_point_cloud_publisher_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>(
            kScanMatchedPointCloudTopic, kLatestOnlyPublisherQueueSize);

    wall_timers_.push_back(node_handle_.createWallTimer(
        ::ros::WallDuration(node_options_.submap_publish_period_sec),
        &CartographerNodelet::PublishSubmapList, this));
    if (node_options_.pose_publish_period_sec > 0) {
      publish_local_trajectory_data_timer_ = node_handle_.createTimer(
          ::ros::Duration(node_options_.pose_publish_period_sec),
          &CartographerNodelet::PublishLocalTrajectoryData, this);
    }
    wall_timers_.push_back(node_handle_.createWallTimer(
        ::ros::WallDuration(node_options_.trajectory_publish_period_sec),
        &CartographerNodelet::PublishTrajectoryNodeList, this));
    wall_timers_.push_back(node_handle_.createWallTimer(
        ::ros::WallDuration(node_options_.trajectory_publish_period_sec),
        &CartographerNodelet::PublishLandmarkPosesList, this));
    wall_timers_.push_back(node_handle_.createWallTimer(
        ::ros::WallDuration(kConstraintPublishPeriodSec),
        &CartographerNodelet::PublishConstraintList, this));

    if (!load_state_filename.empty()) {
      LoadState(load_state_filename, load_frozen_state);
    }

    if (start_trajectory_with_default_topics) {
      ROS_INFO("start_trajectory_with_default_topics");
      StartTrajectoryWithDefaultTopics(trajectory_options);
    }
  }

  // Subscribes to the 'topic' for 'trajectory_id' using the 'node_handle' and
  // calls 'handler' on the 'node' to handle messages. Returns the subscriber.
  template <typename MessageType>
  ::ros::Subscriber SubscribeWithHandler(
      void (CartographerNodelet::*handler)(int, const std::string&,
                            const typename MessageType::ConstPtr&),
      const int trajectory_id, const std::string& topic,
      ::ros::NodeHandle* const node_handle) {
    return node_handle->subscribe<MessageType>(
        topic, kInfiniteSubscriberQueueSize,
        boost::function<void(const typename MessageType::ConstPtr&)>(
            [this, handler, trajectory_id,
             topic](const typename MessageType::ConstPtr& msg) {
              (this->*handler)(trajectory_id, topic, msg);
            }));
  }

  ::ros::NodeHandle* node_handle() { return &node_handle_; }

  bool HandleSubmapCloudQuery(
      ::cartographer_ros_msgs::SubmapCloudQuery::Request& request,
      ::cartographer_ros_msgs::SubmapCloudQuery::Response& response) {
    absl::MutexLock lock(&mutex_);
    map_builder_bridge_->HandleSubmapCloudQuery(request, response);
    return true;
  }

  bool HandleSubmapQuery(
      ::cartographer_ros_msgs::SubmapQuery::Request& request,
      ::cartographer_ros_msgs::SubmapQuery::Response& response) {
    absl::MutexLock lock(&mutex_);
    map_builder_bridge_->HandleSubmapQuery(request, response);
    return true;
  }

  bool HandleTrajectoryQuery(
      ::cartographer_ros_msgs::TrajectoryQuery::Request& request,
      ::cartographer_ros_msgs::TrajectoryQuery::Response& response) {
    absl::MutexLock lock(&mutex_);
    response.status = TrajectoryStateToStatus(
        request.trajectory_id,
        {TrajectoryState::ACTIVE, TrajectoryState::FINISHED,
         TrajectoryState::FROZEN} /* valid states */);
    if (response.status.code != cartographer_ros_msgs::StatusCode::OK) {
      LOG(ERROR) << "Can't query trajectory from pose graph: "
                 << response.status.message;
      return true;
    }
    map_builder_bridge_->HandleTrajectoryQuery(request, response);
    return true;
  }

  void PublishSubmapList(const ::ros::WallTimerEvent& unused_timer_event) {
    absl::MutexLock lock(&mutex_);
    submap_list_publisher_.publish(map_builder_bridge_->GetSubmapList());
  }

  void AddExtrapolator(const int trajectory_id,
                             const TrajectoryOptions& options) {
    constexpr double kExtrapolationEstimationTimeSec = 0.001;  // 1 ms
    CHECK(extrapolators_.count(trajectory_id) == 0);
    const double gravity_time_constant =
        node_options_.map_builder_options.use_trajectory_builder_3d()
            ? options.trajectory_builder_options.trajectory_builder_3d_options()
                  .imu_gravity_time_constant()
            : options.trajectory_builder_options.trajectory_builder_2d_options()
                  .imu_gravity_time_constant();
    extrapolators_.emplace(
        std::piecewise_construct, std::forward_as_tuple(trajectory_id),
        std::forward_as_tuple(
            ::cartographer::common::FromSeconds(kExtrapolationEstimationTimeSec),
            gravity_time_constant));
  }

  void AddSensorSamplers(const int trajectory_id,
                               const TrajectoryOptions& options) {
    CHECK(sensor_samplers_.count(trajectory_id) == 0);
    sensor_samplers_.emplace(
        std::piecewise_construct, std::forward_as_tuple(trajectory_id),
        std::forward_as_tuple(
            options.rangefinder_sampling_ratio, options.odometry_sampling_ratio,
            options.fixed_frame_pose_sampling_ratio, options.imu_sampling_ratio,
            options.landmarks_sampling_ratio));
  }

  void PublishLocalTrajectoryData(const ::ros::TimerEvent& timer_event) {
    absl::MutexLock lock(&mutex_);
    for (const auto& entry : map_builder_bridge_->GetLocalTrajectoryData()) {
      const auto& trajectory_data = entry.second;

      auto& extrapolator = extrapolators_.at(entry.first);
      // We only publish a point cloud if it has changed. It is not needed at high
      // frequency, and republishing it would be computationally wasteful.
      if (trajectory_data.local_slam_data->time !=
          extrapolator.GetLastPoseTime()) {
        if (scan_matched_point_cloud_publisher_.getNumSubscribers() > 0) {
          // TODO(gaschler): Consider using other message without time
          // information.
          carto::sensor::TimedPointCloud point_cloud;
          point_cloud.reserve(trajectory_data.local_slam_data->range_data_in_local
                                  .returns.size());
          for (const cartographer::sensor::RangefinderPoint point :
               trajectory_data.local_slam_data->range_data_in_local.returns) {
            point_cloud.push_back(cartographer::sensor::ToTimedRangefinderPoint(
                point, 0.f /* time */));
          }
          scan_matched_point_cloud_publisher_.publish(ToPointCloud2Message(
              carto::common::ToUniversal(trajectory_data.local_slam_data->time),
              node_options_.map_frame,
              carto::sensor::TransformTimedPointCloud(
                  point_cloud, trajectory_data.local_to_map.cast<float>())));
        }
        extrapolator.AddPose(trajectory_data.local_slam_data->time,
                             trajectory_data.local_slam_data->local_pose);
      }

      geometry_msgs::TransformStamped stamped_transform;
      // If we do not publish a new point cloud, we still allow time of the
      // published poses to advance. If we already know a newer pose, we use its
      // time instead. Since tf knows how to interpolate, providing newer
      // information is better.
      const ::cartographer::common::Time now = std::max(
          FromRos(ros::Time::now()), extrapolator.GetLastExtrapolatedTime());
      stamped_transform.header.stamp =
          node_options_.use_pose_extrapolator
              ? ToRos(now)
              : ToRos(trajectory_data.local_slam_data->time);

      // Suppress publishing if we already published a transform at this time.
      // Due to 2020-07 changes to geometry2, tf buffer will issue warnings for
      // repeated transforms with the same timestamp.
      if (last_published_tf_stamps_.count(entry.first) &&
          last_published_tf_stamps_[entry.first] == stamped_transform.header.stamp)
        continue;
      last_published_tf_stamps_[entry.first] = stamped_transform.header.stamp;

      const Rigid3d tracking_to_local_3d =
          node_options_.use_pose_extrapolator
              ? extrapolator.ExtrapolatePose(now)
              : trajectory_data.local_slam_data->local_pose;
      const Rigid3d tracking_to_local = [&] {
        if (trajectory_data.trajectory_options.publish_frame_projected_to_2d) {
          return carto::transform::Embed3D(
              carto::transform::Project2D(tracking_to_local_3d));
        }
        return tracking_to_local_3d;
      }();

      const Rigid3d tracking_to_map =
          trajectory_data.local_to_map * tracking_to_local;

      if (trajectory_data.published_to_tracking != nullptr) {
        if (node_options_.publish_to_tf) {
          if (trajectory_data.trajectory_options.provide_odom_frame) {
            std::vector<geometry_msgs::TransformStamped> stamped_transforms;

            stamped_transform.header.frame_id = node_options_.map_frame;
            stamped_transform.child_frame_id =
                trajectory_data.trajectory_options.odom_frame;
            stamped_transform.transform =
                ToGeometryMsgTransform(trajectory_data.local_to_map);
            stamped_transforms.push_back(stamped_transform);

            stamped_transform.header.frame_id =
                trajectory_data.trajectory_options.odom_frame;
            stamped_transform.child_frame_id =
                trajectory_data.trajectory_options.published_frame;
            stamped_transform.transform = ToGeometryMsgTransform(
                tracking_to_local * (*trajectory_data.published_to_tracking));
            stamped_transforms.push_back(stamped_transform);

            tf_broadcaster_.sendTransform(stamped_transforms);
          } else {
            stamped_transform.header.frame_id = node_options_.map_frame;
            stamped_transform.child_frame_id =
                trajectory_data.trajectory_options.published_frame;
            stamped_transform.transform = ToGeometryMsgTransform(
                tracking_to_map * (*trajectory_data.published_to_tracking));
            tf_broadcaster_.sendTransform(stamped_transform);
          }
        }
        if (node_options_.publish_tracked_pose) {
          ::geometry_msgs::PoseStamped pose_msg;
          pose_msg.header.frame_id = node_options_.map_frame;
          pose_msg.header.stamp = stamped_transform.header.stamp;
          pose_msg.pose = ToGeometryMsgPose(tracking_to_map);
          tracked_pose_publisher_.publish(pose_msg);
        }
      }
    }
  }

  void PublishTrajectoryNodeList(
      const ::ros::WallTimerEvent& unused_timer_event) {
    if (trajectory_node_list_publisher_.getNumSubscribers() > 0) {
      absl::MutexLock lock(&mutex_);
      trajectory_node_list_publisher_.publish(
          map_builder_bridge_->GetTrajectoryNodeList());
    }
  }

  void PublishLandmarkPosesList(
      const ::ros::WallTimerEvent& unused_timer_event) {
    if (landmark_poses_list_publisher_.getNumSubscribers() > 0) {
      absl::MutexLock lock(&mutex_);
      landmark_poses_list_publisher_.publish(
          map_builder_bridge_->GetLandmarkPosesList());
    }
  }

  void PublishConstraintList(
      const ::ros::WallTimerEvent& unused_timer_event) {
    if (constraint_list_publisher_.getNumSubscribers() > 0) {
      absl::MutexLock lock(&mutex_);
      constraint_list_publisher_.publish(map_builder_bridge_->GetConstraintList());
    }
  }

  std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
  ComputeExpectedSensorIds(const TrajectoryOptions& options) const {
    using SensorId = cartographer::mapping::TrajectoryBuilderInterface::SensorId;
    using SensorType = SensorId::SensorType;
    std::set<SensorId> expected_topics;
    // Subscribe to all laser scan, multi echo laser scan, and point cloud topics.
    for (const std::string& topic :
         ComputeRepeatedTopicNames(kLaserScanTopic, options.num_laser_scans)) {
      expected_topics.insert(SensorId{SensorType::RANGE, topic});
    }
    for (const std::string& topic : ComputeRepeatedTopicNames(
             kMultiEchoLaserScanTopic, options.num_multi_echo_laser_scans)) {
      expected_topics.insert(SensorId{SensorType::RANGE, topic});
    }
    for (const std::string& topic :
         ComputeRepeatedTopicNames(kPointCloud2Topic, options.num_point_clouds)) {
      expected_topics.insert(SensorId{SensorType::RANGE, topic});
    }
    // For 2D SLAM, subscribe to the IMU if we expect it. For 3D SLAM, the IMU is
    // required.
    if (node_options_.map_builder_options.use_trajectory_builder_3d() ||
        (node_options_.map_builder_options.use_trajectory_builder_2d() &&
         options.trajectory_builder_options.trajectory_builder_2d_options()
             .use_imu_data())) {
      expected_topics.insert(SensorId{SensorType::IMU, kImuTopic});
    }
    // Odometry is optional.
    if (options.use_odometry) {
      expected_topics.insert(SensorId{SensorType::ODOMETRY, kOdometryTopic});
    }
    // NavSatFix is optional.
    if (options.use_nav_sat) {
      expected_topics.insert(
          SensorId{SensorType::FIXED_FRAME_POSE, kNavSatFixTopic});
    }
    // Landmark is optional.
    if (options.use_landmarks) {
      expected_topics.insert(SensorId{SensorType::LANDMARK, kLandmarkTopic});
    }
    return expected_topics;
  }

  int AddTrajectory(const TrajectoryOptions& options) {
    ROS_INFO("AddTrajectory");
    const std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>
        expected_sensor_ids = ComputeExpectedSensorIds(options);
    const int trajectory_id =
        map_builder_bridge_->AddTrajectory(expected_sensor_ids, options);
    AddExtrapolator(trajectory_id, options);
    AddSensorSamplers(trajectory_id, options);
    LaunchSubscribers(options, trajectory_id);
    wall_timers_.push_back(node_handle_.createWallTimer(
        ::ros::WallDuration(kTopicMismatchCheckDelaySec),
        &CartographerNodelet::MaybeWarnAboutTopicMismatch, this, /*oneshot=*/true));
    for (const auto& sensor_id : expected_sensor_ids) {
      subscribed_topics_.insert(sensor_id.id);
    }
    return trajectory_id;
  }

  void LaunchSubscribers(const TrajectoryOptions& options,
                               const int trajectory_id) {
    for (const std::string& topic :
         ComputeRepeatedTopicNames(kLaserScanTopic, options.num_laser_scans)) {
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<sensor_msgs::LaserScan>(
               &CartographerNodelet::HandleLaserScanMessage, trajectory_id, topic, &node_handle_),
           topic});
    }
    for (const std::string& topic : ComputeRepeatedTopicNames(
             kMultiEchoLaserScanTopic, options.num_multi_echo_laser_scans)) {
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<sensor_msgs::MultiEchoLaserScan>(
               &CartographerNodelet::HandleMultiEchoLaserScanMessage, trajectory_id, topic,
               &node_handle_),
           topic});
    }
    for (const std::string& topic :
         ComputeRepeatedTopicNames(kPointCloud2Topic, options.num_point_clouds)) {
      ROS_INFO("PointCloud2 topic: %s",topic.c_str());
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<sensor_msgs::PointCloud2>(
               &CartographerNodelet::HandlePointCloud2Message, trajectory_id, topic,
               &node_handle_),
           topic});
    }

    // For 2D SLAM, subscribe to the IMU if we expect it. For 3D SLAM, the IMU is
    // required.
    if (node_options_.map_builder_options.use_trajectory_builder_3d() ||
        (node_options_.map_builder_options.use_trajectory_builder_2d() &&
         options.trajectory_builder_options.trajectory_builder_2d_options()
             .use_imu_data())) {
      ROS_INFO("IMU topic: %s",kImuTopic);
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<sensor_msgs::Imu>(&CartographerNodelet::HandleImuMessage,
                                                  trajectory_id, kImuTopic,
                                                  &node_handle_),
           kImuTopic});
    }

    if (options.use_odometry) {
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<nav_msgs::Odometry>(&CartographerNodelet::HandleOdometryMessage,
                                                    trajectory_id, kOdometryTopic,
                                                    &node_handle_),
           kOdometryTopic});
    }
    if (options.use_nav_sat) {
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<sensor_msgs::NavSatFix>(
               &CartographerNodelet::HandleNavSatFixMessage, trajectory_id, kNavSatFixTopic,
               &node_handle_),
           kNavSatFixTopic});
    }
    if (options.use_landmarks) {
      subscribers_[trajectory_id].push_back(
          {CartographerNodelet::SubscribeWithHandler<cartographer_ros_msgs::LandmarkList>(
               &CartographerNodelet::HandleLandmarkMessage, trajectory_id, kLandmarkTopic,
               &node_handle_),
           kLandmarkTopic});
    }
  }

  bool ValidateTrajectoryOptions(const TrajectoryOptions& options) {
    ROS_INFO("ValidateTrajectoryOptions");
    if (node_options_.map_builder_options.use_trajectory_builder_2d()) {
      return options.trajectory_builder_options
          .has_trajectory_builder_2d_options();
    }
    if (node_options_.map_builder_options.use_trajectory_builder_3d()) {
      ROS_INFO("Use 3d trajectory");
      return options.trajectory_builder_options
          .has_trajectory_builder_3d_options();
    }
    ROS_INFO("return false");
    return false;
  }

  bool ValidateTopicNames(const TrajectoryOptions& options) {
    for (const auto& sensor_id : ComputeExpectedSensorIds(options)) {
      const std::string& topic = sensor_id.id;
      if (subscribed_topics_.count(topic) > 0) {
        LOG(ERROR) << "Topic name [" << topic << "] is already used.";
        return false;
      }
    }
    return true;
  }

  cartographer_ros_msgs::StatusResponse TrajectoryStateToStatus(
      const int trajectory_id, const std::set<TrajectoryState>& valid_states) {
    const auto trajectory_states = map_builder_bridge_->GetTrajectoryStates();
    cartographer_ros_msgs::StatusResponse status_response;

    const auto it = trajectory_states.find(trajectory_id);
    if (it == trajectory_states.end()) {
      status_response.message =
          absl::StrCat("Trajectory ", trajectory_id, " doesn't exist.");
      status_response.code = cartographer_ros_msgs::StatusCode::NOT_FOUND;
      return status_response;
    }

    status_response.message =
        absl::StrCat("Trajectory ", trajectory_id, " is in '",
                     TrajectoryStateToString(it->second), "' state.");
    status_response.code =
        valid_states.count(it->second)
            ? cartographer_ros_msgs::StatusCode::OK
            : cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
    return status_response;
  }

  cartographer_ros_msgs::StatusResponse FinishTrajectoryUnderLock(
      const int trajectory_id) {
    cartographer_ros_msgs::StatusResponse status_response;
    if (trajectories_scheduled_for_finish_.count(trajectory_id)) {
      status_response.message = absl::StrCat("Trajectory ", trajectory_id,
                                             " already pending to finish.");
      status_response.code = cartographer_ros_msgs::StatusCode::OK;
      LOG(INFO) << status_response.message;
      return status_response;
    }

    // First, check if we can actually finish the trajectory.
    status_response = TrajectoryStateToStatus(
        trajectory_id, {TrajectoryState::ACTIVE} /* valid states */);
    if (status_response.code != cartographer_ros_msgs::StatusCode::OK) {
      LOG(ERROR) << "Can't finish trajectory: " << status_response.message;
      return status_response;
    }

    // Shutdown the subscribers of this trajectory.
    // A valid case with no subscribers is e.g. if we just visualize states.
    if (subscribers_.count(trajectory_id)) {
      for (auto& entry : subscribers_[trajectory_id]) {
        entry.subscriber.shutdown();
        subscribed_topics_.erase(entry.topic);
        LOG(INFO) << "Shutdown the subscriber of [" << entry.topic << "]";
      }
      CHECK_EQ(subscribers_.erase(trajectory_id), 1);
    }
    map_builder_bridge_->FinishTrajectory(trajectory_id);
    trajectories_scheduled_for_finish_.emplace(trajectory_id);
    status_response.message =
        absl::StrCat("Finished trajectory ", trajectory_id, ".");
    status_response.code = cartographer_ros_msgs::StatusCode::OK;
    return status_response;
  }

  bool HandleStartTrajectory(
      ::cartographer_ros_msgs::StartTrajectory::Request& request,
      ::cartographer_ros_msgs::StartTrajectory::Response& response) {
    TrajectoryOptions trajectory_options;
    std::tie(std::ignore, trajectory_options) = LoadOptions(
        request.configuration_directory, request.configuration_basename);

    if (request.use_initial_pose) {
      const auto pose = ToRigid3d(request.initial_pose);
      if (!pose.IsValid()) {
        response.status.message =
            "Invalid pose argument. Orientation quaternion must be normalized.";
        LOG(ERROR) << response.status.message;
        response.status.code =
            cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
        return true;
      }

      // Check if the requested trajectory for the relative initial pose exists.
      response.status = TrajectoryStateToStatus(
          request.relative_to_trajectory_id,
          {TrajectoryState::ACTIVE, TrajectoryState::FROZEN,
           TrajectoryState::FINISHED} /* valid states */);
      if (response.status.code != cartographer_ros_msgs::StatusCode::OK) {
        LOG(ERROR) << "Can't start a trajectory with initial pose: "
                   << response.status.message;
        return true;
      }

      ::cartographer::mapping::proto::InitialTrajectoryPose
          initial_trajectory_pose;
      initial_trajectory_pose.set_to_trajectory_id(
          request.relative_to_trajectory_id);
      *initial_trajectory_pose.mutable_relative_pose() =
          cartographer::transform::ToProto(pose);
      initial_trajectory_pose.set_timestamp(cartographer::common::ToUniversal(
          ::cartographer_ros::FromRos(ros::Time(0))));
      *trajectory_options.trajectory_builder_options
           .mutable_initial_trajectory_pose() = initial_trajectory_pose;
    }

    if (!ValidateTrajectoryOptions(trajectory_options)) {
      response.status.message = "Invalid trajectory options.";
      LOG(ERROR) << response.status.message;
      response.status.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
    } else if (!ValidateTopicNames(trajectory_options)) {
      response.status.message = "Topics are already used by another trajectory.";
      LOG(ERROR) << response.status.message;
      response.status.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
    } else {
      response.status.message = "Success.";
      response.trajectory_id = AddTrajectory(trajectory_options);
      response.status.code = cartographer_ros_msgs::StatusCode::OK;
    }
    return true;
  }

  void StartTrajectoryWithDefaultTopics(const TrajectoryOptions& options) {
    ROS_INFO("StartTrajectoryWithDefaultTopics");
    absl::MutexLock lock(&mutex_);
    ROS_INFO("locked");
    CHECK(ValidateTrajectoryOptions(options));
    ROS_INFO("Checked");
    AddTrajectory(options);
  }

  std::vector<
      std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>>
  ComputeDefaultSensorIdsForMultipleBags(
      const std::vector<TrajectoryOptions>& bags_options) const {
    using SensorId = cartographer::mapping::TrajectoryBuilderInterface::SensorId;
    std::vector<std::set<SensorId>> bags_sensor_ids;
    for (size_t i = 0; i < bags_options.size(); ++i) {
      std::string prefix;
      if (bags_options.size() > 1) {
        prefix = "bag_" + std::to_string(i + 1) + "_";
      }
      std::set<SensorId> unique_sensor_ids;
      for (const auto& sensor_id : ComputeExpectedSensorIds(bags_options.at(i))) {
        unique_sensor_ids.insert(SensorId{sensor_id.type, prefix + sensor_id.id});
      }
      bags_sensor_ids.push_back(unique_sensor_ids);
    }
    return bags_sensor_ids;
  }

  int AddOfflineTrajectory(
      const std::set<cartographer::mapping::TrajectoryBuilderInterface::SensorId>&
          expected_sensor_ids,
      const TrajectoryOptions& options) {
    absl::MutexLock lock(&mutex_);
    const int trajectory_id =
        map_builder_bridge_->AddTrajectory(expected_sensor_ids, options);
    AddExtrapolator(trajectory_id, options);
    AddSensorSamplers(trajectory_id, options);
    return trajectory_id;
  }

  bool HandleGetTrajectoryStates(
      ::cartographer_ros_msgs::GetTrajectoryStates::Request& request,
      ::cartographer_ros_msgs::GetTrajectoryStates::Response& response) {
    using TrajectoryState =
        ::cartographer::mapping::PoseGraphInterface::TrajectoryState;
    absl::MutexLock lock(&mutex_);
    response.status.code = ::cartographer_ros_msgs::StatusCode::OK;
    response.trajectory_states.header.stamp = ros::Time::now();
    for (const auto& entry : map_builder_bridge_->GetTrajectoryStates()) {
      response.trajectory_states.trajectory_id.push_back(entry.first);
      switch (entry.second) {
        case TrajectoryState::ACTIVE:
          response.trajectory_states.trajectory_state.push_back(
              ::cartographer_ros_msgs::TrajectoryStates::ACTIVE);
          break;
        case TrajectoryState::FINISHED:
          response.trajectory_states.trajectory_state.push_back(
              ::cartographer_ros_msgs::TrajectoryStates::FINISHED);
          break;
        case TrajectoryState::FROZEN:
          response.trajectory_states.trajectory_state.push_back(
              ::cartographer_ros_msgs::TrajectoryStates::FROZEN);
          break;
        case TrajectoryState::DELETED:
          response.trajectory_states.trajectory_state.push_back(
              ::cartographer_ros_msgs::TrajectoryStates::DELETED);
          break;
      }
    }
    return true;
  }

  bool HandleFinishTrajectory(
      ::cartographer_ros_msgs::FinishTrajectory::Request& request,
      ::cartographer_ros_msgs::FinishTrajectory::Response& response) {
    absl::MutexLock lock(&mutex_);
    response.status = FinishTrajectoryUnderLock(request.trajectory_id);
    return true;
  }

  bool HandleWriteState(
      ::cartographer_ros_msgs::WriteState::Request& request,
      ::cartographer_ros_msgs::WriteState::Response& response) {
    absl::MutexLock lock(&mutex_);
    if (map_builder_bridge_->SerializeState(request.filename,
                                           request.include_unfinished_submaps)) {
      response.status.code = cartographer_ros_msgs::StatusCode::OK;
      response.status.message =
          absl::StrCat("State written to '", request.filename, "'.");
    } else {
      response.status.code = cartographer_ros_msgs::StatusCode::INVALID_ARGUMENT;
      response.status.message =
          absl::StrCat("Failed to write '", request.filename, "'.");
    }
    return true;
  }

  bool HandleReadMetrics(
      ::cartographer_ros_msgs::ReadMetrics::Request& request,
      ::cartographer_ros_msgs::ReadMetrics::Response& response) {
    absl::MutexLock lock(&mutex_);
    response.timestamp = ros::Time::now();
    if (!metrics_registry_) {
      response.status.code = cartographer_ros_msgs::StatusCode::UNAVAILABLE;
      response.status.message = "Collection of runtime metrics is not activated.";
      return true;
    }
    metrics_registry_->ReadMetrics(&response);
    response.status.code = cartographer_ros_msgs::StatusCode::OK;
    response.status.message = "Successfully read metrics.";
    return true;
  }

  void FinishAllTrajectories() {
    absl::MutexLock lock(&mutex_);
    for (const auto& entry : map_builder_bridge_->GetTrajectoryStates()) {
      if (entry.second == TrajectoryState::ACTIVE) {
        const int trajectory_id = entry.first;
        CHECK_EQ(FinishTrajectoryUnderLock(trajectory_id).code,
                 cartographer_ros_msgs::StatusCode::OK);
      }
    }
  }

  bool FinishTrajectory(const int trajectory_id) {
    absl::MutexLock lock(&mutex_);
    return FinishTrajectoryUnderLock(trajectory_id).code ==
           cartographer_ros_msgs::StatusCode::OK;
  }

  void RunFinalOptimization() {
    {
      for (const auto& entry : map_builder_bridge_->GetTrajectoryStates()) {
        const int trajectory_id = entry.first;
        if (entry.second == TrajectoryState::ACTIVE) {
          LOG(WARNING)
              << "Can't run final optimization if there are one or more active "
                 "trajectories. Trying to finish trajectory with ID "
              << std::to_string(trajectory_id) << " now.";
          CHECK(FinishTrajectory(trajectory_id))
              << "Failed to finish trajectory with ID "
              << std::to_string(trajectory_id) << ".";
        }
      }
    }
    // Assuming we are not adding new data anymore, the final optimization
    // can be performed without holding the mutex.
    map_builder_bridge_->RunFinalOptimization();
  }

  void HandleOdometryMessage(const int trajectory_id,
                                   const std::string& sensor_id,
                                   const nav_msgs::Odometry::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (!sensor_samplers_.at(trajectory_id).odometry_sampler.Pulse()) {
      return;
    }
    auto sensor_bridge_ptr = map_builder_bridge_->sensor_bridge(trajectory_id);
    auto odometry_data_ptr = sensor_bridge_ptr->ToOdometryData(msg);
    if (odometry_data_ptr != nullptr) {
      extrapolators_.at(trajectory_id).AddOdometryData(*odometry_data_ptr);
    }
    sensor_bridge_ptr->HandleOdometryMessage(sensor_id, msg);
  }

  void HandleNavSatFixMessage(const int trajectory_id,
                                    const std::string& sensor_id,
                                    const sensor_msgs::NavSatFix::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (!sensor_samplers_.at(trajectory_id).fixed_frame_pose_sampler.Pulse()) {
      return;
    }
    map_builder_bridge_->sensor_bridge(trajectory_id)
        ->HandleNavSatFixMessage(sensor_id, msg);
  }

  void HandleLandmarkMessage(
      const int trajectory_id, const std::string& sensor_id,
      const cartographer_ros_msgs::LandmarkList::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (!sensor_samplers_.at(trajectory_id).landmark_sampler.Pulse()) {
      return;
    }
    map_builder_bridge_->sensor_bridge(trajectory_id)
        ->HandleLandmarkMessage(sensor_id, msg);
  }

  void HandleImuMessage(const int trajectory_id,
                              const std::string& sensor_id,
                              const sensor_msgs::Imu::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (prev_imu_msg_ && (msg->header.stamp - prev_imu_msg_->header.stamp).toSec() < 0.0)
    {
      return;
    }
    else
    {
      prev_imu_msg_ = msg;
    }
    if (!sensor_samplers_.at(trajectory_id).imu_sampler.Pulse()) {
      return;
    }
    auto sensor_bridge_ptr = map_builder_bridge_->sensor_bridge(trajectory_id);
    auto imu_data_ptr = sensor_bridge_ptr->ToImuData(msg);
    if (imu_data_ptr != nullptr) {
      extrapolators_.at(trajectory_id).AddImuData(*imu_data_ptr);
    }
    sensor_bridge_ptr->HandleImuMessage(sensor_id, msg);
  }

  void HandleLaserScanMessage(const int trajectory_id,
                                    const std::string& sensor_id,
                                    const sensor_msgs::LaserScan::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (!sensor_samplers_.at(trajectory_id).rangefinder_sampler.Pulse()) {
      return;
    }
    map_builder_bridge_->sensor_bridge(trajectory_id)
        ->HandleLaserScanMessage(sensor_id, msg);
  }

  void HandleMultiEchoLaserScanMessage(
      const int trajectory_id, const std::string& sensor_id,
      const sensor_msgs::MultiEchoLaserScan::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (!sensor_samplers_.at(trajectory_id).rangefinder_sampler.Pulse()) {
      return;
    }
    map_builder_bridge_->sensor_bridge(trajectory_id)
        ->HandleMultiEchoLaserScanMessage(sensor_id, msg);
  }

  void HandlePointCloud2Message(
      const int trajectory_id, const std::string& sensor_id,
      const sensor_msgs::PointCloud2::ConstPtr& msg) {
    absl::MutexLock lock(&mutex_);
    if (!sensor_samplers_.at(trajectory_id).rangefinder_sampler.Pulse()) {
      return;
    }
    map_builder_bridge_->sensor_bridge(trajectory_id)
        ->HandlePointCloud2Message(sensor_id, msg);
  }

  void SerializeState(const std::string& filename,
                            const bool include_unfinished_submaps) {
    absl::MutexLock lock(&mutex_);
    CHECK(
        map_builder_bridge_->SerializeState(filename, include_unfinished_submaps))
        << "Could not write state.";
  }

  void LoadState(const std::string& state_filename,
                       const bool load_frozen_state) {
    absl::MutexLock lock(&mutex_);
    map_builder_bridge_->LoadState(state_filename, load_frozen_state);
  }

  void MaybeWarnAboutTopicMismatch(
      const ::ros::WallTimerEvent& unused_timer_event) {
    ::ros::master::V_TopicInfo ros_topics;
    ::ros::master::getTopics(ros_topics);
    std::set<std::string> published_topics;
    std::stringstream published_topics_string;
    for (const auto& it : ros_topics) {
      std::string resolved_topic = node_handle_.resolveName(it.name, false);
      published_topics.insert(resolved_topic);
      published_topics_string << resolved_topic << ",";
    }
    bool print_topics = false;
    for (const auto& entry : subscribers_) {
      int trajectory_id = entry.first;
      for (const auto& subscriber : entry.second) {
        std::string resolved_topic = node_handle_.resolveName(subscriber.topic);
        if (published_topics.count(resolved_topic) == 0) {
          LOG(WARNING) << "Expected topic \"" << subscriber.topic
                       << "\" (trajectory " << trajectory_id << ")"
                       << " (resolved topic \"" << resolved_topic << "\")"
                       << " but no publisher is currently active.";
          print_topics = true;
        }
      }
    }
    if (print_topics) {
      LOG(WARNING) << "Currently available topics are: "
                   << published_topics_string.str();
    }
  }

private:
  struct Subscriber {
    ::ros::Subscriber subscriber;

    // ::ros::Subscriber::getTopic() does not necessarily return the same
    // std::string
    // it was given in its constructor. Since we rely on the topic name as the
    // unique identifier of a subscriber, we remember it ourselves.
    std::string topic;
  };
  NodeOptions node_options_;

  tf2_ros::TransformBroadcaster tf_broadcaster_;

  absl::Mutex mutex_;
  std::unique_ptr<cartographer_ros::metrics::FamilyFactory> metrics_registry_;
  std::shared_ptr<MapBuilderBridge> map_builder_bridge_ GUARDED_BY(mutex_);

  ::ros::NodeHandle node_handle_;
  ::ros::Publisher submap_list_publisher_;
  ::ros::Publisher trajectory_node_list_publisher_;
  ::ros::Publisher landmark_poses_list_publisher_;
  ::ros::Publisher constraint_list_publisher_;
  ::ros::Publisher tracked_pose_publisher_;
  // These ros::ServiceServers need to live for the lifetime of the node.
  std::vector<::ros::ServiceServer> service_servers_;
  ::ros::Publisher scan_matched_point_cloud_publisher_;

  struct TrajectorySensorSamplers {
    TrajectorySensorSamplers(const double rangefinder_sampling_ratio,
                             const double odometry_sampling_ratio,
                             const double fixed_frame_pose_sampling_ratio,
                             const double imu_sampling_ratio,
                             const double landmark_sampling_ratio)
        : rangefinder_sampler(rangefinder_sampling_ratio),
          odometry_sampler(odometry_sampling_ratio),
          fixed_frame_pose_sampler(fixed_frame_pose_sampling_ratio),
          imu_sampler(imu_sampling_ratio),
          landmark_sampler(landmark_sampling_ratio) {}

    ::cartographer::common::FixedRatioSampler rangefinder_sampler;
    ::cartographer::common::FixedRatioSampler odometry_sampler;
    ::cartographer::common::FixedRatioSampler fixed_frame_pose_sampler;
    ::cartographer::common::FixedRatioSampler imu_sampler;
    ::cartographer::common::FixedRatioSampler landmark_sampler;
  };

  // These are keyed with 'trajectory_id'.
  std::map<int, ::cartographer::mapping::PoseExtrapolator> extrapolators_;
  std::map<int, ::ros::Time> last_published_tf_stamps_;
  std::unordered_map<int, TrajectorySensorSamplers> sensor_samplers_;
  std::unordered_map<int, std::vector<Subscriber>> subscribers_;
  std::unordered_set<std::string> subscribed_topics_;
  std::unordered_set<int> trajectories_scheduled_for_finish_;

  // We have to keep the timer handles of ::ros::WallTimers around, otherwise
  // they do not fire.
  std::vector<::ros::WallTimer> wall_timers_;

  // The timer for publishing local trajectory data (i.e. pose transforms and
  // range data point clouds) is a regular timer which is not triggered when
  // simulation time is standing still. This prevents overflowing the transform
  // listener buffer by publishing the same transforms over and over again.
  ::ros::Timer publish_local_trajectory_data_timer_;

  ::ros::Timer init_timer_;
  std::string save_state_filename_;
  std::shared_ptr<tf2_ros::TransformListener> tf_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<cartographer::mapping::MapBuilderInterface> map_builder_;

  sensor_msgs::Imu::ConstPtr prev_imu_msg_;

}; // class CartographerNodelet
}  // namespace cartographer_ros

// Register nodelet plugin
#include <swri_nodelet/class_list_macros.h>
SWRI_NODELET_EXPORT_CLASS(cartographer_ros, CartographerNodelet)
