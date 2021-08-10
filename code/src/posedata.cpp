#include "aPFNN/posedata.h"

namespace a::pfnn
{
    static void get_jnts_local_info(std::vector<Vec3> &jnts_local_positions,    // size: noj     //output
                             std::vector<Vec3> &jnts_local_aaxis,               // size: noj     //output
                             std::vector<Vec3> &jnts_local_velocity,            // size: noj     //output
                             agl::spModel model,
                             const Mat4& rootTrf,
                             const agl::Pose &previousPose,
                             const agl::Pose &currentPose,
                             const std::vector<std::string> &joint_names)
    {
        //set variables
        const int jnt_num = joint_names.size();

        jnts_local_positions.resize(jnt_num);
        jnts_local_aaxis.resize(jnt_num);
        jnts_local_velocity.resize(jnt_num);

        std::vector<Vec3> previous_lPos;
        previous_lPos.resize(jnt_num);

        //** Previous Frame
        {
            model->set_pose(previousPose);
            model->root()->update_world_trf_children(); // update global trfs

            for (int j = 0; j < jnt_num; j++)
            {
                // get name of each joint
                std::string jnt = joint_names.at(j);

                // get jnt_local_pos
                Vec4 jnt_world_pos_homo = Vec4::Ones(4);
                jnt_world_pos_homo.block<3, 1>(0, 0) = model->joint(jnt)->world_pos();
                Vec3 jnt_local_pos = (rootTrf.inverse() * jnt_world_pos_homo).block<3, 1>(0, 0);
                previous_lPos.at(j) = jnt_local_pos;
            }                     
        }

        //** get joints local pos and local AAxis
        {
            //** current Frame
            model->set_pose(currentPose);
            model->root()->update_world_trf_children(); // update global trfs

            for (int j = 0; j < jnt_num; j++)
            {
                // get name of each joint
                std::string jnt = joint_names.at(j);

                // get jnt_local_pos
                Vec4 jnt_world_pos_homo = Vec4::Ones();
                jnt_world_pos_homo.block<3, 1>(0, 0) = model->joint(jnt)->world_pos();

                Vec3 jnt_local_pos = (rootTrf.inverse() * jnt_world_pos_homo).block<3, 1>(0, 0);
                
                // get local_orientation of jnt
                Mat4 world_rot = model->joint(jnt)->world_trf();

                Mat3 jnt_local_rot = (rootTrf.inverse() * world_rot).block<3, 3>(0, 0);

                AAxis jnt_local_aaxis_(jnt_local_rot);
                float angle = jnt_local_aaxis_.angle(); // scalar
                Vec3 axis = jnt_local_aaxis_.axis();    // unit vector
                Vec3 jnt_local_aaxis_3d = angle * axis;

                jnts_local_positions.at(j) = jnt_local_pos;
                jnts_local_aaxis.at(j) = jnt_local_aaxis_3d;

                //** reconstruction
                // float langle = jnts_local_aaxis.at(j).norm();
                // Vec3 laxis = jnts_local_aaxis.at(j) / langle;
                // AAxis local_aaxis = AAxis(langle, laxis);
                // Quat local_quat(local_aaxis);
            }
        }

        {
            //** get velocity info
            for (int i = 0; i < jnt_num; ++i)
            {
                Vec3 jnt_local_vel = (jnts_local_positions.at(i)- previous_lPos.at(i)) / (1.0f / 60.0f);
                jnts_local_velocity.at(i) = jnt_local_vel;
            }
        }
        
        // //** 다시 원래 pose로 돌아가기!!
        // model->set_pose(currentPose);
        // model->root()->update_world_trf_children(); // update global trfs
    }

    static void get_local_trj_info(std::vector<Vec3> &local_trj_positions,      // size: sample_timings.size()     //output
                            std::vector<Vec3> &local_trj_directions,            // size: sample_timings.size()     //output
                            const int pidx,                                     // current frame index
                            const std::vector<Mat4> &root_trjs,
                            const std::vector<int> &sample_timings)
    {
        int windowSize = sample_timings.size();

        local_trj_positions.resize(windowSize);
        local_trj_directions.resize(windowSize);

        // rootTrf
        Mat4 rootTrf = root_trjs.at(pidx);

        for (int i = 0; i < windowSize; ++i)
        {
            int frame_idx = pidx + sample_timings.at(i);
            Mat4 frame_trf = root_trjs.at(frame_idx);
            Vec3 frame_pos = frame_trf.block<3, 1>(0, 3);

            //** get root_local_trj_position
            Vec4 root_world_pos_homo = Vec4::Ones(4);
            root_world_pos_homo.block<3, 1>(0, 0) = frame_pos;
            Vec3 trj_local_pos = (rootTrf.inverse() * root_world_pos_homo).block<3, 1>(0, 0);

            //** get root_local_trj_direction
            Vec4 z_dir = Vec4::Zero();
            z_dir.head<3>() = frame_trf.block<3, 1>(0, 2);
            Vec3 local_z_dir = (rootTrf.inverse() * z_dir).block<3, 1>(0, 0);

            local_trj_positions.at(i) = trj_local_pos;
            local_trj_directions.at(i) = local_z_dir;
        }
    }

    //! needed to be changed when the terrain height is different to 0
    static void get_local_trj_heights(std::vector<Vec3> &local_trj_heights,
                                      const std::vector<int> &sample_timings)
    {
        int windowSize = sample_timings.size();
        local_trj_heights.resize(windowSize);
        for (int i = 0; i < windowSize; ++i)
        {
            local_trj_heights.at(i) = Vec3::Zero();
        }
    }

    PFNN_XY_FrameData get_frame_data(
        int pidx,                               // 50 <= frame < nof - 60
        agl::spModel model,
        const std::vector<agl::Pose> &poses,    // nof
        const std::vector<Mat4> &root_trjs,     // nof
        const std::vector<float> &phase,
        const std::vector<std::string> &joint_names,
        const std::vector<int> &sample_timings,
        const bool left_contact,
        const bool right_contact)
    {
        int window = sample_timings.size();

        //** pidx에 대한 constraint assertion
        assert(("out of range! sampled frame is less than 0", (pidx + sample_timings.at(0)) >= 0));
        assert(("out of range! sampled frame is larger than size of poses", 
                (pidx + sample_timings.at(window - 1) < poses.size())));
        assert(("out of range! it has no previous frame", (pidx - 1) >= 0));

        //** setting variables
        const int jnt_num = joint_names.size();
        const float timeStep = 1.0f / 60.0f;

        //** data that will be returned
        PFNN_XY_FrameData result;
        
        //** Tensor of each information
        std::vector<Vec3> position_info;
        std::vector<Vec3> orientation_info;
        std::vector<Vec3> velocity_info;

        std::vector<Vec3> local_trj_pos_; // size: window size
        std::vector<Vec3> local_trj_dir_; // size: window size
        std::vector<Vec3> local_trj_h_; // size: window size

        //** get rootTrf
        Mat4 rootTrf = root_trjs.at(pidx);
        Mat4 previous_rootTrf = root_trjs.at(pidx - 1);
        result.world_root_trf = rootTrf;

        //** get joint local information
        get_jnts_local_info(position_info, orientation_info, velocity_info,
                            model, rootTrf, poses.at(pidx - 1), poses.at(pidx), joint_names);

        result.jnts_local_positions = position_info;
        result.jnts_local_aaxis = orientation_info;
        result.jnts_local_velocity = velocity_info;

        //** get root_local_velocity
        Vec3 currentPos = rootTrf.col(3).head(3);
        Vec3 previousPos = previous_rootTrf.col(3).head(3);

        Vec3 root_global_vel = (currentPos - previousPos) / timeStep;
        Vec4 root_global_vel_homo = Vec4::Zero();
        root_global_vel_homo.block<3, 1>(0, 0) = root_global_vel;

        //! be careful that the root_local_vel is relative to previous_rootTrf
        Vec4 root_local_vel_homo = (previous_rootTrf.inverse() * root_global_vel_homo);
        // Vec4 root_local_vel_homo = (rootTrf.inverse() * root_global_vel_homo);
        Vec3 root_local_vel = root_local_vel_homo.head<3>();

        result.root_local_velocity = root_local_vel;

        //** get root_angular_velocity
        Vec3 z_current = rootTrf.col(2).head(3);
        Vec3 z_previous = previous_rootTrf.col(2).head(3);
        float dTheta = agl::get_sangle(z_previous, z_current, Vec3::UnitY());
        float root_angular_vel = dTheta / timeStep;

        result.root_angular_velocity = root_angular_vel;

        //*get Phase & dPhase
        float phase_change = phase.at(pidx) - phase.at(pidx - 1);
        if (phase_change < 0)
        {
            phase_change += 2 * M_PI;
            phase_change = fmod(phase_change, 2 * M_PI);
        }

        result.dPhase = phase_change;

        //** get local_trj_info
        get_local_trj_info(local_trj_pos_, local_trj_dir_, pidx, root_trjs, sample_timings);

        result.local_trj_positions = local_trj_pos_;
        result.local_trj_directions = local_trj_dir_;

        //! needed to be changed when the terrain height is different to 0
        //** get local trajectory heights of the three left/right/center sample points
        get_local_trj_heights(local_trj_h_, sample_timings);

        //** get contact label
        result.l_contact = left_contact;
        result.r_contact = right_contact;

        result.local_trj_heights = local_trj_h_;
        return result;
    }
}