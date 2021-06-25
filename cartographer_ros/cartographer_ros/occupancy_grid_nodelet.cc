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

#include <cmath>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/synchronization/mutex.h"
#include "cairo/cairo.h"
#include "cartographer/common/port.h"
#include "cartographer/io/image.h"
#include "cartographer/io/submap_painter.h"
#include "cartographer/mapping/id.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer_ros/msg_conversion.h"
#include "cartographer_ros/node_constants.h"
#include "cartographer_ros/ros_log_sink.h"
#include "cartographer_ros/submap.h"
#include "cartographer_ros_msgs/SubmapList.h"
#include "cartographer_ros_msgs/SubmapQuery.h"
#include "cartographer_ros_msgs/SubmapCloudQuery.h"
#include "gflags/gflags.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Odometry.h"
#include "nodelet/nodelet.h"
#include "ros/ros.h"
#include <swri_roscpp/parameters.h>

namespace cartographer_ros {

using ::cartographer::io::PaintSubmapSlicesResult;
using ::cartographer::io::SubmapSlice;
using ::cartographer::mapping::SubmapId;

class OccupancyGridNodelet : public nodelet::Nodelet
{
public:
  OccupancyGridNodelet()
  {
  }
  ~OccupancyGridNodelet() {}

  virtual void onInit()
  {
    node_handle_ = getNodeHandle();
    init_timer_ = node_handle_.createTimer(ros::Duration(1.0), &OccupancyGridNodelet::Initialize, this, true);
  }

  void Initialize(const ros::TimerEvent& unused)
  {
    ros::NodeHandle pnh = getPrivateNodeHandle();
    double publish_period_sec;
    swri::param(pnh, "altitude_threshold",altitude_threshold_, 4.0);
    swri::param(pnh, "minimum_probability",minimum_probability_, 0.2);
    swri::param(pnh, "resolution",resolution_, 0.05);
    swri::param(pnh, "publish_period_sec",publish_period_sec, 1.0);
    swri::param(pnh, "include_frozen_submaps",include_frozen_submaps_, true);
    swri::param(pnh, "include_unfrozen_submaps",include_unfrozen_submaps_, true);

    // CHECK(FLAGS_include_frozen_submaps || FLAGS_include_unfrozen_submaps)
    //     << "Ignoring both frozen and unfrozen submaps makes no sense.";
    // if ()

    client_ = node_handle_.serviceClient<::cartographer_ros_msgs::SubmapQuery>(kSubmapQueryServiceName);
    cloud_client_ = node_handle_.serviceClient<::cartographer_ros_msgs::SubmapCloudQuery>(kSubmapCloudQueryServiceName);

    odometry_subscriber_ = node_handle_.subscribe("odom", kLatestOnlyPublisherQueueSize,
      &OccupancyGridNodelet::HandleOdometry, this);

    submap_list_subscriber_ = node_handle_.subscribe(
      kSubmapListTopic, kLatestOnlyPublisherQueueSize,
      boost::function<void(const cartographer_ros_msgs::SubmapList::ConstPtr&)>(
        [this](const cartographer_ros_msgs::SubmapList::ConstPtr& msg) {
          HandleSubmapList(msg);
        }));

    occupancy_grid_publisher_ = node_handle_.advertise<::nav_msgs::OccupancyGrid>(
      "occupancy_grid", kLatestOnlyPublisherQueueSize, true /* latched */);

    voxel_cloud_publisher_ = node_handle_.advertise<::sensor_msgs::PointCloud2>(
      "voxel_cloud", kLatestOnlyPublisherQueueSize, true);
        
    occupancy_grid_publisher_timer_ = node_handle_.createWallTimer(
      ::ros::WallDuration(publish_period_sec), &OccupancyGridNodelet::DrawAndPublish, this);
  }

  // Node(const Node&) = delete;
  // Node& operator=(const Node&) = delete;

  void HandleOdometry(const nav_msgs::OdometryConstPtr& msg)
  {
    odom_ = msg;
  }

  void HandleSubmapList(
      const cartographer_ros_msgs::SubmapList::ConstPtr& msg) {
    absl::MutexLock locker(&mutex_);

    // We do not do any work if nobody listens.
    if (occupancy_grid_publisher_.getNumSubscribers() == 0 &&
        voxel_cloud_publisher_.getNumSubscribers() == 0) {
      return;
    }

    // Keep track of submap IDs that don't appear in the message anymore.
    std::set<SubmapId> submap_ids_to_delete;
    for (const auto& pair : submap_slices_) {
      submap_ids_to_delete.insert(pair.first);
    }

    for (const auto& submap_msg : msg->submap) {
      const SubmapId id{submap_msg.trajectory_id, submap_msg.submap_index};
      if (odom_ && std::fabs(odom_->pose.pose.position.z - submap_msg.pose.position.z) > altitude_threshold_ &&
        submap_msg != msg->submap.back()) {
        continue;
      }
      submap_ids_to_delete.erase(id);
      if ((submap_msg.is_frozen && !include_frozen_submaps_) ||
          (!submap_msg.is_frozen && !include_unfrozen_submaps_)) {
        continue;
      }
      SubmapSlice& submap_slice = submap_slices_[id];
      submap_slice.pose = ToRigid3d(submap_msg.pose);
      submap_slice.metadata_version = submap_msg.submap_version;
      if (submap_slice.surface != nullptr &&
          submap_slice.version == submap_msg.submap_version) {
        continue;
      }

      auto fetched_textures =
          ::cartographer_ros::FetchSubmapTextures(id, &client_);
      if (fetched_textures == nullptr) {
        continue;
      } else if (fetched_textures->textures.empty()) {
        continue;
      }
      submap_slice.version = fetched_textures->version;

      // We use the first texture only. By convention this is the highest
      // resolution texture and that is the one we want to use to construct the
      // map for ROS.
      const auto fetched_texture = fetched_textures->textures.begin();
      submap_slice.width = fetched_texture->width;
      submap_slice.height = fetched_texture->height;
      submap_slice.slice_pose = fetched_texture->slice_pose;
      submap_slice.resolution = fetched_texture->resolution;
      submap_slice.cairo_data.clear();
      submap_slice.surface = ::cartographer::io::DrawTexture(
          fetched_texture->pixels.intensity, fetched_texture->pixels.alpha,
          fetched_texture->width, fetched_texture->height,
          &submap_slice.cairo_data);

      auto fetched_cloud =
        ::cartographer_ros::FetchSubmapCloud(id,minimum_probability_,false,&cloud_client_);
      if (fetched_cloud != nullptr && voxel_cloud_.data.empty()) {
        voxel_cloud_ = *fetched_cloud;
      } else if (fetched_cloud != nullptr) {
        voxel_cloud_.width += fetched_cloud->width;
        voxel_cloud_.data.insert(voxel_cloud_.data.end(),fetched_cloud->data.begin(),fetched_cloud->data.end());
      }
    }

    // Delete all submaps that didn't appear in the message.
    for (const auto& id : submap_ids_to_delete) {
      submap_slices_.erase(id);
    }

    last_timestamp_ = msg->header.stamp;
    last_frame_id_ = msg->header.frame_id;
  }

  void DrawAndPublish(const ::ros::WallTimerEvent& unused_timer_event) {
    absl::MutexLock locker(&mutex_);
    if (submap_slices_.empty() || last_frame_id_.empty()) {
      return;
    }
    auto painted_slices = PaintSubmapSlices(submap_slices_, resolution_);
    std::unique_ptr<nav_msgs::OccupancyGrid> msg_ptr = CreateOccupancyGridMsg(
        painted_slices, resolution_, last_frame_id_, last_timestamp_);
    occupancy_grid_publisher_.publish(*msg_ptr);
    if (voxel_cloud_.data.empty()) {
      return;
    }
    voxel_cloud_.header.stamp = ros::Time::now();
    voxel_cloud_.header.frame_id = last_frame_id_;
    voxel_cloud_publisher_.publish(voxel_cloud_);
    voxel_cloud_.data.clear();
  }

 private:
  ::ros::NodeHandle node_handle_;
  double resolution_;

  absl::Mutex mutex_;
  ::ros::ServiceClient client_ GUARDED_BY(mutex_);
  ::ros::ServiceClient cloud_client_ GUARDED_BY(mutex_);
  ::ros::Subscriber odometry_subscriber_ GUARDED_BY(mutex_);
  ::ros::Subscriber submap_list_subscriber_ GUARDED_BY(mutex_);
  ::ros::Publisher occupancy_grid_publisher_ GUARDED_BY(mutex_);
  ::ros::Publisher voxel_cloud_publisher_ GUARDED_BY(mutex_);
  std::map<SubmapId, SubmapSlice> submap_slices_ GUARDED_BY(mutex_);
  ::ros::WallTimer occupancy_grid_publisher_timer_;
  ::ros::Timer init_timer_;
  std::string last_frame_id_;
  ros::Time last_timestamp_;
  bool include_frozen_submaps_;
  bool include_unfrozen_submaps_;
  sensor_msgs::PointCloud2 voxel_cloud_ GUARDED_BY(mutex_);
  nav_msgs::Odometry::ConstPtr odom_;
  double altitude_threshold_;
  float minimum_probability_;
}; // class OccupancyGridNodelet
}  // namespace cartographer_ros

// Register nodelet plugin
#include <swri_nodelet/class_list_macros.h>
SWRI_NODELET_EXPORT_CLASS(cartographer_ros, OccupancyGridNodelet)
