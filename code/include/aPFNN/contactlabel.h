#pragma once
#include <aOpenGL.h>

namespace a::pfnn {

/**
 * @brief approximate_standard_foot_height 
 * 
 * @param FootToeBase joint of footToeBase
 * @param FootToeEnd joint of footToeEnd
 * @param Foot joint of foot
 **/
float approximate_standard_foot_height(const agl::spJoint FootToeBase,
                                       const agl::spJoint FootToeEnd,
                                       const agl::spJoint Foot);

/**
 * @brief label_contact_info
 * 
 * @param left_foot_contact_label       label whether left foot is contacting on the ground at current frame 
 * @param right_foot_contact_label      label whether right foot is contacting on the ground at current frame                  
 * @param model                         model
 * @param leftFoot                      leftFoot of model
 * @param rightFoot                     rightFoot of model
 * @param foot_height_standard          initial foot height of model when the foot is contacting on the ground 
 * @param height_epsilon                some threshold of foot height
 * @param speed_epsilon                 some threshold of speed of foot
 * @param timeStep                      timeStep of each clip
 **/
void label_contact_info(std::vector<bool>& left_foot_contact_label, 
                        std::vector<bool>& right_foot_contact_label,
                        agl::spModel model, 
                        const agl::Motion& motion,
                        agl::spJoint leftFoot,
                        agl::spJoint rightFoot,
                        const float foot_height_standard,
                        const float height_epsilon,
                        const float speed_epsilon,
                        const float timeStep);

/**
 * @brief filter_contact_info
 * 
 * @param left_foot_contact_label   label whether left foot is contacting on the ground at current frame 
 * @param right_foot_contact_label  label whether right foot is contacting on the ground at current frame                   
 * @param former_window             window size of former frames - check whether the foot was contacting or not    
 * @param latter_window             window size of latter frames - check whether the foot was contacting or not         
 * @param maximum_missing           filtering하고 싶은 foot contact 정보 - foot이 ground와 contacting하지 않는 연속된 frame의 maximum 개수
 **/
void filter_contact_info(std::vector<bool>& left_foot_contact_label, 
                         std::vector<bool>& right_foot_contact_label, 
                         const int former_window, 
                         const int latter_window,
                         const int maximum_missing);
}