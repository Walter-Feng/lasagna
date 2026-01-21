// slots of atm
#define CHARGE_OF 0
#define PTR_COORD 1
#define NUC_MOD_OF 2
#define PTR_ZETA 3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS 6

// slots of bas
#define ATOM_OF 0
#define ANG_OF 1
#define NPRIM_OF 2
#define NCTR_OF 3
#define KAPPA_OF 4
#define PTR_EXP 5
#define PTR_COEFF 6
#define PTR_BAS_COORD 7
#define BAS_SLOTS 8

#define atm(SLOT, I) atm[ATM_SLOTS * (I) + (SLOT)]
#define bas(SLOT, I) bas[BAS_SLOTS * (I) + (SLOT)]

#include "overlap.cuh"
#include <math.h>
#include <stdio.h>

namespace ovlp {
template <typename T, int i_angular, int j_angular>
__global__ void
ovlp_kernel(T *ovlp, const int *pair_indices, const int n_primitives,
            const int n_pairs, const int *primitive_to_function,
            const int n_functions, const int *atm, const int atm_stride,
            const int *bas, const int bas_stride, const T *env,
            const int env_stride) {
  atm += blockIdx.y * atm_stride;
  bas += blockIdx.y * bas_stride;
  env += blockIdx.y * env_stride;

  int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair_idx >= n_pairs)
    return;

  const int primitive_pair = pair_indices[pair_idx];
  const int i_primitive = primitive_pair / n_primitives;
  const int j_primitive = primitive_pair % n_primitives;

  const T alpha = env[bas(PTR_EXP, i_primitive)];
  const T beta = env[bas(PTR_EXP, j_primitive)];

  const T c1 = env[bas(PTR_COEFF, i_primitive)] * common_fac_sp<T, i_angular>();
  const T c2 = env[bas(PTR_COEFF, j_primitive)] * common_fac_sp<T, j_angular>();

  const int i_atom = bas(ATOM_OF, i_primitive);
  const int j_atom = bas(ATOM_OF, j_primitive);

  const int i_coord_offset = atm(PTR_COORD, i_atom);
  const int j_coord_offset = atm(PTR_COORD, j_atom);

  const T i_x = env[i_coord_offset + 0];
  const T i_y = env[i_coord_offset + 1];
  const T i_z = env[i_coord_offset + 2];

  const T j_x = env[j_coord_offset + 0];
  const T j_y = env[j_coord_offset + 1];
  const T j_z = env[j_coord_offset + 2];

  const T ix_to_jx = j_x - i_x;
  const T iy_to_jy = j_y - i_y;
  const T iz_to_jz = j_z - i_z;
  const T pair_distance_squared =
      ix_to_jx * ix_to_jx + iy_to_jy * iy_to_jy + iz_to_jz * iz_to_jz;

  const T pair_exponent = alpha + beta;
  T prefactor = sqrt(M_PI / pair_exponent);
  prefactor *= prefactor * prefactor * c1 * c2;
  prefactor *= exp(-alpha * beta / pair_exponent * pair_distance_squared);

  if (i_primitive == j_primitive) {
    prefactor *= 0.5;
  }

  const int i_function_index = primitive_to_function[i_primitive];
  const int j_function_index = primitive_to_function[j_primitive];

  ovlp += blockIdx.y * n_functions * n_functions +
          i_function_index * n_functions + j_function_index;

  if constexpr (i_angular == 0 && j_angular == 0) {
    atomicAdd(ovlp, prefactor);
  } else {
    const T factor_a = -alpha / pair_exponent;
    const T factor_b = 0.5 / pair_exponent;

    T x_pairs[(i_angular + 1) * (j_angular + 1)];
    vertical_recursion<T, i_angular + j_angular>(x_pairs, prefactor,
                                                 factor_a * ix_to_jx, factor_b);
    horizontal_recursion<T, i_angular, j_angular>(x_pairs, ix_to_jx);

    T y_pairs[(i_angular + 1) * (j_angular + 1)];
    vertical_recursion<T, i_angular + j_angular>(y_pairs, 1,
                                                 factor_a * iy_to_jy, factor_b);
    horizontal_recursion<T, i_angular, j_angular>(y_pairs, iy_to_jy);

    T z_pairs[(i_angular + 1) * (j_angular + 1)];
    vertical_recursion<T, i_angular + j_angular>(z_pairs, 1,
                                                 factor_a * iz_to_jz, factor_b);
    horizontal_recursion<T, i_angular, j_angular>(z_pairs, iz_to_jz);

    write_spherical_function_pairs<T, i_angular, j_angular>(
        ovlp, x_pairs, y_pairs, z_pairs, n_functions);
  }
}

} // namespace ovlp
extern "C" {

#define ovlp_kernel_macro(T, i, j)                                             \
  case i * 10 + j:                                                             \
    ovlp::ovlp_kernel<T, i, j><<<block_grid, block_size>>>(                    \
        ovlp, pair_indices, n_primitives, n_pairs, primitive_to_function,      \
        n_functions, atm, atm_stride, bas, bas_stride, env, env_stride);       \
    break;

void overlap_double(double *ovlp, const int *pair_indices, const int n_pairs,
                    const int n_primitives, const int *primitive_to_function,
                    const int n_functions, const int *atm, const int atm_stride,
                    const int *bas, const int bas_stride, const double *env,
                    const int env_stride, const int n_configurations,
                    const int i_angular, const int j_angular) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) {
    ovlp_kernel_macro(double, 0, 0);
    ovlp_kernel_macro(double, 0, 1);
    ovlp_kernel_macro(double, 0, 2);
    ovlp_kernel_macro(double, 0, 3);
    ovlp_kernel_macro(double, 0, 4);

    ovlp_kernel_macro(double, 1, 0);
    ovlp_kernel_macro(double, 1, 1);
    ovlp_kernel_macro(double, 1, 2);
    ovlp_kernel_macro(double, 1, 3);
    ovlp_kernel_macro(double, 1, 4);

    ovlp_kernel_macro(double, 2, 0);
    ovlp_kernel_macro(double, 2, 1);
    ovlp_kernel_macro(double, 2, 2);
    ovlp_kernel_macro(double, 2, 3);
    ovlp_kernel_macro(double, 2, 4);

    ovlp_kernel_macro(double, 3, 0);
    ovlp_kernel_macro(double, 3, 1);
    ovlp_kernel_macro(double, 3, 2);
    ovlp_kernel_macro(double, 3, 3);
    ovlp_kernel_macro(double, 3, 4);

    ovlp_kernel_macro(double, 4, 0);
    ovlp_kernel_macro(double, 4, 1);
    ovlp_kernel_macro(double, 4, 2);
    ovlp_kernel_macro(double, 4, 3);
    ovlp_kernel_macro(double, 4, 4);
  }
}

void overlap_float(float *ovlp, const int *pair_indices, const int n_pairs,
                   const int n_primitives, const int *primitive_to_function,
                   const int n_functions, const int *atm, const int atm_stride,
                   const int *bas, const int bas_stride, const float *env,
                   const int env_stride, const int n_configurations,
                   const int i_angular, const int j_angular) {
  const dim3 block_size{256, 1, 1};
  const dim3 block_grid{(uint)((n_pairs + 255) / 256), (uint)n_configurations,
                        1};

  switch (i_angular * 10 + j_angular) {
    ovlp_kernel_macro(float, 0, 0);
    ovlp_kernel_macro(float, 0, 1);
    ovlp_kernel_macro(float, 0, 2);
    ovlp_kernel_macro(float, 0, 3);
    ovlp_kernel_macro(float, 0, 4);

    ovlp_kernel_macro(float, 1, 0);
    ovlp_kernel_macro(float, 1, 1);
    ovlp_kernel_macro(float, 1, 2);
    ovlp_kernel_macro(float, 1, 3);
    ovlp_kernel_macro(float, 1, 4);

    ovlp_kernel_macro(float, 2, 0);
    ovlp_kernel_macro(float, 2, 1);
    ovlp_kernel_macro(float, 2, 2);
    ovlp_kernel_macro(float, 2, 3);
    ovlp_kernel_macro(float, 2, 4);

    ovlp_kernel_macro(float, 3, 0);
    ovlp_kernel_macro(float, 3, 1);
    ovlp_kernel_macro(float, 3, 2);
    ovlp_kernel_macro(float, 3, 3);
    ovlp_kernel_macro(float, 3, 4);

    ovlp_kernel_macro(float, 4, 0);
    ovlp_kernel_macro(float, 4, 1);
    ovlp_kernel_macro(float, 4, 2);
    ovlp_kernel_macro(float, 4, 3);
    ovlp_kernel_macro(float, 4, 4);
  }
}
}
