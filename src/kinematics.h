#ifndef KINEMATICS_H
#define KINEMATICS_H

#include "Eigen/Dense"
#include "link.h"

Eigen::VectorXd getQ0(const Eigen::VectorXd& q_init, const Link& links);

#endif
