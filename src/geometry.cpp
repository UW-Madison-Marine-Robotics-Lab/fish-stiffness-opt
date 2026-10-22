#include "geometry.h"

Geometry::Geometry(double L)
    : L(L), R1 (0.08 * L), R2 (2 * M_PI / (1.42 * L)),
    R3 (0.0019 * L), R4 (2 * M_PI / (1.44 * L)),
    r1 (0.2969), r2 (-0.1260), r3 (-0.3516), r4 (0.2843), r5 (-0.1015) {}

Eigen::VectorXd Geometry::calc_R(const Eigen::VectorXd& ls) const {
    // Eigen::VectorXd R_values = R1 * (ls.array() * R2).sin() + R3 * ((ls.array() * R4).exp() - 1);
    // return R_values;
    const double l0 = 0.8 * L;
    const double R0 = R1 * std::sin(R2 * l0) + R3 * (std::exp(R4 * l0) - 1);

    Eigen::ArrayXd l = ls.array();

    Eigen::ArrayXd R_original =
        R1 * (l * R2).sin()
        + R3 * ((l * R4).exp() - 1);

    Eigen::ArrayXd R_linear =
        R0 + 0.3 * (l - l0);   // slope = 0.3

    Eigen::ArrayXd R_values =
        (l <= l0).select(R_original, R_linear);

    return R_values.matrix();
}

Eigen::VectorXd Geometry::calc_r(const Eigen::VectorXd& ls) const {
    Eigen::ArrayXd x = ls.array() / L;
    Eigen::ArrayXd r_values = L * 0.6 * (
        r1 * x.sqrt()
        + r2 * x 
        + r3 * x.square()
        + r4 * x.cube()
        + r5 * x.pow(4)
    );
    return r_values.matrix();
}

double Geometry::calc_R(double l) const
{
    // return R1 * std::sin(R2 * l) + R3 * (std::exp(R4 * l) - 1);
    const double l0 = 0.8 * L;
    const double R0 = R1 * std::sin(R2 * l0) + R3 * (std::exp(R4 * l0) - 1);

    if (l <= l0) {
        return R1 * std::sin(R2 * l) + R3 * (std::exp(R4 * l) - 1);
    } else {
        return R0 + 0.3 * (l - l0);   // slope = 0.3
    }
}

double Geometry::calc_r(double l) const
{
    double x = l / L;
    return L * 0.6 * (
        r1 * std::sqrt(x)
        + r2 * x 
        + r3 * std::pow(x, 2)
        + r4 * std::pow(x, 3)
        + r5 * std::pow(x, 4)
    );
}
