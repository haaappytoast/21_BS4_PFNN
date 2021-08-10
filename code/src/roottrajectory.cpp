#include "aPFNN/roottrajectory.h"
#include <iostream>

namespace a::pfnn
{

    
    void get_root_info(std::vector<Mat4> &root_world_trf,
                       const agl::spModel model,
                       const agl::Motion motion,
                       const agl::spJoint leftShoulder,
                       const agl::spJoint rightShoulder,
                       const agl::spJoint hips)
    {
        int pose_num = motion.poses.size();
        root_world_trf.resize(pose_num);

        Vec3 leftShoulder_worldPos;
        Vec3 rightShoulder_worldPos;
        Vec3 shoulder_dir;
        Vec3 hips_dir;

        for (int i = 0; i < pose_num; ++i)
        {
            // set model
            model->set_pose(motion.poses.at(i));
            model->root()->update_world_trf_children(); // update global trfs

            //** get facing direction of root
            leftShoulder_worldPos = leftShoulder->world_pos();
            rightShoulder_worldPos = rightShoulder->world_pos();

            Vec3 shoulder_dir = (rightShoulder_worldPos - leftShoulder_worldPos).normalized();
            Vec3 shoulder_facing_dir = (Vec3::UnitY().cross(shoulder_dir)).normalized();

            Vec3 hips_facing_dir = hips->world_trf().col(2).head<3>();

            // make sure that every facing direction has no y-component
            shoulder_dir(1, 0) = 0.0f;
            hips_facing_dir(1, 0) = 0.0f;

            Vec3 avg_facing_dir = (shoulder_facing_dir + hips_facing_dir).normalized();

            // make sure that angle between y-axis and avg_facing_dir is 90 degree
            assert(("angle between y-axis and avg_facing_dir has to be 90 degree", avg_facing_dir.dot(Vec3::UnitY()) < 2e-4));

            //** get world_trf of root
            Mat4 root_world_trf_ = Mat4::Identity();

            // y-direction of root
            Vec3 y_axis = Vec3::UnitY();

            root_world_trf_.block<3, 1>(0, 1) = y_axis;

            // z-direction of root
            root_world_trf_.block<3, 1>(0, 2) = avg_facing_dir;

            // x-direction of root
            Vec3 x_dir = (y_axis.cross(avg_facing_dir)).normalized();
            root_world_trf_.block<3, 1>(0, 0) = x_dir;

            assert(("rotation matrix of root has to be Unitary", root_world_trf_.block<3, 3>(0, 0).isUnitary() == true));

            // translation of root
            Vec3 translation = hips->world_pos();
            translation(1, 0) = 0.0f; // make y-direction of translation to 0 because it has no height
            root_world_trf_.block<3, 1>(0, 3) = translation;
       
            root_world_trf.at(i) = root_world_trf_;
        }

    }
}