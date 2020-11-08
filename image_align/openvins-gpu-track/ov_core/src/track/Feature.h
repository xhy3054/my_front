#ifndef OV_CORE_FEATURE_H
#define OV_CORE_FEATURE_H


#include <map>
#include <vector>
#include <Eigen/Eigen>


/**
 * @brief Sparse feature class used to collect measurements
 *
 * This feature class allows for holding of all tracking information for a given feature.
 * Each feature has a unique ID assigned to it, and should have a set of feature tracks alongside it.
 * See the FeatureDatabase class for details on how we load information into this, and how we delete features.
 */
class Feature
{

public:

    /// Unique ID of this feature
    size_t featid;

    /// If this feature should be deleted
    bool to_delete;

    /// UV coordinates that this feature has been seen from (mapped by camera ID)
    std::map<size_t,std::vector<Eigen::Vector2f>> uvs;

    /// UV normalized coordinates that this feature has been seen from (mapped by camera ID)
    std::map<size_t,std::vector<Eigen::Vector2f>> uvs_norm;

    /// Timestamps of each UV measurement (mapped by camera ID)
    std::map<size_t,std::vector<double>> timestamps;

    /// Triangulated inverse position of this feature, in the anchor frame
    Eigen::Vector3d p_invFinA;

    /// Triangulated position of this feature, in the anchor frame
    Eigen::Vector3d p_FinA;

    /// What camera ID our pose is anchored in!! By default the first measurement is the anchor.
    size_t anchor_cam_id;

    /// Triangulated position of this feature, in the global frame
    Eigen::Vector3d p_FinG;

};




#endif /* OV_CORE_FEATURE_H */