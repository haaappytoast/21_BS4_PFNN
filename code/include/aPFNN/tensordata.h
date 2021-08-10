#pragma once
#include <aOpenGL.h>
#include <aLibTorch.h>
#include "config.h"

namespace a::pfnn
{
    //** Data setting when Training
    /**
     * @brief with PFNN_XY_FrameData, make it into X_tensor input data
     * 
     * @param prev_data              PFNN_XY_FrameData of previous frame
     * @param curr_data             PFNN_XY_FrameData of current frame
     * @return Tensor 
     **/
    Tensor to_X_tensor(const PFNN_XY_FrameData &prev_data, const PFNN_XY_FrameData &curr_data);

    /**
     * @brief with PFNN_XY_FrameData, make it into Y_tensor output data
     * 
     * @param curr_data              PFNN_XY_FrameData of current frame
     * @param latt_data              PFNN_XY_FrameData of latter frame
     * @return Tensor 
     **/
    Tensor to_Y_tensor(const PFNN_XY_FrameData &curr_data, const PFNN_XY_FrameData &latt_data);



    //** Data Setting when Testing
    /**
     * @brief with Y tensor, parse it into y_data
     * 
     * @param model              
     * @param prev_phase       phase of previous frame to calculate phase of current frame
     * @param prev_rootTrf     root transformation of previous frame to calculate root transform of current frame 
     * @param Y_tensor         output Tensor of PFNN
     * @param windowSize       window size of sampling times (typically 12)
     * @param joints_names     names of joints used in PFNN
     * @return PFNN_Y_Data 
     **/
    PFNN_Y_Data parse_Y_tensor_to_y_data(agl::spModel model,
                                      const float prev_phase,
                                      const Mat4 prev_rootTrf,
                                      const Tensor &Y_tensor,
                                      const int windowSize,
                                      const std::vector<std::string> joints_names);

    //** Data Setting when Testing

    /**
     * @brief from y_data, make it into X_tensor
     * 
     * @param y_data              data gotten from [parse_Y_tensor_to_y_data()] function
     * 
     * @return Tensor 
     **/
    Tensor to_X_tensor_from_Y(const PFNN_Y_Data& y_data);


    /**
     * @brief from y_data, make it into X_tensor
     * 
     * @param y_tensor              output of network
     * @param windowSize            window size (in PFNN network: 12)
     * @param jnt_size              size of joints used for input
     * 
     * @return XTensor 
     **/
    Tensor to_X_tensor_from_YTensor(const Tensor& y_tensor, int windowSize, int jnt_size);


    /**
     * @brief reconstruct pose from output y_data
     * @param model                     
     * @param y_data              data gotten from [parse_Y_tensor_to_y_data()] function
     * 
     * @return agl::Pose  
     **/
    agl::Pose reconstruct_pose(agl::spModel model,
                               const PFNN_Y_Data& y_data,
                               const std::vector<std::string> joints_names);

    // /**
    //  * @brief blend trajectory direction with predicted direction
    //  * 
    //  * @param rootTrf          rootTrf to change target_dir(global) to local
    //  * @param orig_trj_dir     original trajectory direction which is gotten from output of network
    //  * @param target_dir       target_direction gotten from game-pad control stick
    //  * @param scale_dir        scale used in interpolation

    //  * @return Vec3 (interpolated direction) 
    //  **/
    // Vec3 blend_traj_direction(const Mat4 rootTrf,
    //                           Vec3 orig_trj_dir,
    //                           Vec3 target_dir,
    //                           const float dir_scale);

    // /**
    //  * @brief blend trajectory positions with predicted velocity
    //  * 
    //  * @param prev_orig_trj_pos              
    //  * @param curr_orig_trj_pos       
    //  * @param prev_blended_trj_pos       previous position of blended trajectory
    //  * @param target_vel                target_velocity gotten from game-pad control stick
    //  * @param pos_scale                 scale used in interpolation
    //  * @return (interpolated position) 
    //  **/
    // Vec3 blend_traj_position(const Mat4& rootTrf,
    //                          const Vec3& prev_orig_trj_pos,
    //                          const Vec3& curr_orig_trj_pos,
    //                          const Vec3& prev_blended_trj_pos,
    //                          const Vec3& target_vel,
    //                          const float pos_scale);


    /**
     * @brief blend trajectory positions with predicted velocity
     * 
     * @param rootTrf          rootTrf to change target_dir(global) to local
     * @param trj_orig_pos              original trajectory position which is gotten from output of network
     * @param pos_bias                  additional bias that controls the responsiveness of the charac
     * @param target_vel                target_velocity gotten from game-pad control stick
     * @param windowSize                
     * @return (interpolated position) 
     **/
    vVec3 blend_traj_poss(const Mat4& rootTrf,
                          const vVec3& trj_orig_pos,
                          const float pos_bias,
                          const Vec3& target_vel,
                          const int windowSize);

    vVec3 blend_traj_dirs(const Mat4 rootTrf,
                          const vVec3 trj_orig_dir,
                          const float dir_bias,
                          const Vec3 target_dir,
                          const int windowSize);

    Tensor jntInfo_fromRecon(agl::spModel model,
                                const agl::Pose& pose,
                                const Mat4& rootTrf,
                                const std::vector<std::string>& joint_names,
                                const std::vector<Vec3>& prev_jnt_glob_pos);

    void prev_trajInfo_fromRecon(Tensor& trj_prev_lposT, 
                            Tensor& trj_prev_ldirT,
                            vVec3& trj_prev_lposV, 
                            vVec3& trj_prev_ldirV, 
                            const vVec3& trj_glb_prev_pos, 
                            const vVec3& trj_glb_prev_dir,
                            const float windowSize,
                            const Mat4& rt,
                            const std::vector<int> sample_timing);
    
    void latter_trajInfo_fromRecon(Tensor &trj_lat_lposT,
                                 Tensor &trj_lat_ldirT,
                                 const vVec3 &trj_lpos,
                                 const vVec3 &trj_ldir,
                                 const float windowSize);
}