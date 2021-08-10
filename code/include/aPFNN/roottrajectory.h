#pragma once
#include <aOpenGL.h>
#include <aLibTorch.h>

namespace a::pfnn
{
   /**
    * @brief get_root_info
    * 
    * @param root_world_trf    world_transformation matrix of root for each frame of motion
    * @param model             model
    * @param motion            motion data
    * @param leftShoulder      joint of leftShoulder
    * @param rightShoulder     joint of rightShoulder
    * @param hips              joint of hips
    **/
   void get_root_info(std::vector<Mat4> &root_world_trf,
                      const agl::spModel model,
                      const agl::Motion motion,
                      const agl::spJoint leftShoulder,
                      const agl::spJoint rightShoulder,
                      const agl::spJoint hips);
}