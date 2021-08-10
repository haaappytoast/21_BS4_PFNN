#pragma once
#include <aOpenGL.h>
#include <aLibTorch.h>
#include "aPFNN/config.h"


namespace a::pfnn {


    /**
     * @brief Get the frame data object
     * @param model
     * @param poses                 
     * @param root_trjs            root trajectory info at each frame       //size: poses.size()
     * @param phase                phase info(range from 0 to 2 * M_PI)     //size: poses.size()
     * @param joint_names          
     * @param sample_timings       { -60 -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50}
     *                             t = 12 sampled surrounding frames 
     *                             covering 1 second of motion in the past 
     *                             and 0.9 seconds of motion in the future
     * @param left_contact
     * @param right_contact
     * 
     * @return PFNN_XY_FrameData 
     **/
    PFNN_XY_FrameData get_frame_data(
        int pidx, // 60 <= frame < nof - 50
        agl::spModel model,
        const std::vector<agl::Pose> &poses, // nof
        const std::vector<Mat4> &root_trjs,  // nof
        const std::vector<float> &phase,
        const std::vector<std::string> &joint_names,
        const std::vector<int> &sample_timings,
        const bool left_contact,
        const bool right_contact);
};