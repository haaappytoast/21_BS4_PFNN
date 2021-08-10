#pragma once
#include <aOpenGL.h>

namespace a::pfnn {
    
    /**
     * @brief approximate_standard_foot_height 
     * 
     * @param left_foot_contact_label   label information of whether left foot is contacting on the ground at current frame
     * @param right_foot_contact_label  label information of whether right foot is contacting on the ground at current frame
     * @param phase                     range from 0 to 2 * M_PI
     * @param min_frame_of_contact      minimum number of serial frames which will be labeled as standing motion (two feet are both contacting with ground)
     **/
    void label_phase_info(const std::vector<bool> &left_foot_contact_label,
                          const std::vector<bool> &right_foot_contact_label,
                          std::vector<float> &phase,
                          const int min_frame_of_contact);
}