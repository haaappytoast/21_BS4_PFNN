#pragma once
#include <aLibTorch.h>

namespace a::pfnn
{
    /**
     * @brief normalize tensor (dimension of input has to be 1)
     * @param features              input tensor of which to be normalized (dim has to be 1)
     * 
     * @return std::vector<Tensor> containing [normalized_input, mean, stdev] 
     **/
    std::vector<Tensor> normalize_tensor(const Tensor &feature);
    
    /**
     * @brief normalize tensors (dimension of input has to be 2)
     * @param features              input tensor of which to be normalized for each row or column (dim has to be 2)
     * @param dimen                   dimension of which to be normalized (if 0; each row is normalized, for 1; each column is normalized)
     * 
     * @return Tensor containing [normalized_inputs] 
     **/
    Tensor normalize_tensors(const Tensor &features, const int dimen);
    

    /**
     * @brief normalize_tensor with given mean and stdev (dimension of input has to be 1)
     * @param feature             
     * @param mean_stdev        Tensor = {mean, stdev}
     * @param range             range of mean_stdev to find out which are gonna be used
     * 
     * @return Tensor containing [normalized_inputs] 
     **/
    Tensor normalize_tensor_with_stdev(const Tensor &feature, const Tensor &mean_stdev, const Tensor &range);    

    /**
     * @brief normalize tensor (dimension of input has to be 2)
     * @param features              input tensor of which to be normalized for each row (dim has to be 2)
     * @param dimen                 dimension of which to be normalized (if 0; each row is normalized, for 1; each column is normalized)
     * 
     * @return Tensor containing vector of [normalized_input, mean, stdev] for each row/column
     **/
    std::vector<std::vector<Tensor>> normalize_tensors_seperate(const Tensor &features, const int dimen);

    /**
     * @brief scale tensors
     * @param features              input tensor of which to be normalized for each row (dim has to be 2)
     * @param range_of_idx          index array of which to be scaled
     * @param scale                 
     * @return Tensor containing vector of [normalized_input, mean, stdev] for each row 
     **/
    Tensor scale_tensor(const Tensor &features, const Tensor &range_of_idx, const float scale);


    /**
     * @brief denormalize tensors with given stdard deviation and mean value
     * @param features              input tensor of which to be normalized for each row (dim has to be 2)
     * @param std_mean              Tensor of stdandard deviation and mean value (size: no of feature x 2)                 
     * @return Tensors denormalized 
     **/    
    Tensor denormalize_tensor(const Tensor &features, Tensor& std_mean);

    Tensor to_int_type_tensor(const Tensor &feature, const Tensor&range_of_idx);
    
    }