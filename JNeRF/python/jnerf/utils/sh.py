import jittor as jt

class HardCodedSH:
    """
    Code adapted from TensoIR sh.py
    """
    def __init__(self, degree=4):
        self.degree = degree
        if degree > 4 or degree < 0:
            raise ValueError("Currently, 0-4 SH degree are supported")
        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761,
        ]

    def eval_sh(self, sh_coeff, rays_d):
        """
        Evaluate spherical harmonics at unit directions
        using hardcoded SH polynomials.
        ... Can be 0 or more batch dimensions.
        :param sh_coeff: torch.Tensor SH coeffs (..., C, )
        :param rays_d: torch.Tensor unit directions (..., 3)
        :return: (..., C)
        """
        deg = self.degree
        if (deg + 1) ** 2 != sh_coeff.shape[-1]:
            raise ValueError("SH coefficient must equal to (degree + 1) ** 2, "
                             "current degress={}, sh_coeff={}".format(deg, sh_coeff.shape[-1]))
        result = self.C0 * sh_coeff[..., 0]
        if deg > 0:
            x, y, z = rays_d[..., 0:1], rays_d[..., 1:2], rays_d[..., 2:3]
            result = (result -
                      self.C1 * y * sh_coeff[..., 1] +
                      self.C1 * z * sh_coeff[..., 2] -
                      self.C1 * x * sh_coeff[..., 3])
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result = (result +
                          self.C2[0] * xy * sh_coeff[..., 4] +
                          self.C2[1] * yz * sh_coeff[..., 5] +
                          self.C2[2] * (2.0 * zz - xx - yy) * sh_coeff[..., 6] +
                          self.C2[3] * xz * sh_coeff[..., 7] +
                          self.C2[4] * (xx - yy) * sh_coeff[..., 8])

                if deg > 2:
                    result = (result +
                              self.C3[0] * y * (3 * xx - yy) * sh_coeff[..., 9] +
                              self.C3[1] * xy * z * sh_coeff[..., 10] +
                              self.C3[2] * y * (4 * zz - xx - yy) * sh_coeff[..., 11] +
                              self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeff[..., 12] +
                              self.C3[4] * x * (4 * zz - xx - yy) * sh_coeff[..., 13] +
                              self.C3[5] * z * (xx - yy) * sh_coeff[..., 14] +
                              self.C3[6] * x * (xx - 3 * yy) * sh_coeff[..., 15])
                    if deg > 3:
                        result = (result + self.C4[0] * xy * (xx - yy) * sh_coeff[..., 16] +
                                  self.C4[1] * yz * (3 * xx - yy) * sh_coeff[..., 17] +
                                  self.C4[2] * xy * (7 * zz - 1) * sh_coeff[..., 18] +
                                  self.C4[3] * yz * (7 * zz - 3) * sh_coeff[..., 19] +
                                  self.C4[4] * (zz * (35 * zz - 30) + 3) * sh_coeff[..., 20] +
                                  self.C4[5] * xz * (7 * zz - 3) * sh_coeff[..., 21] +
                                  self.C4[6] * (xx - yy) * (7 * zz - 1) * sh_coeff[..., 22] +
                                  self.C4[7] * xz * (xx - 3 * yy) * sh_coeff[..., 23] +
                                  self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh_coeff[..., 24])
        return result

    def eval_sh_bases(self, rays_d):
        """
        Evaluate spherical harmonics bases at unit directions,
        without taking linear combination.
        At each point, the final result may be obtained through simple multiplication.
        """
        deg = self.degree
        result = jt.empty((*rays_d.shape[:-1], (deg + 1) ** 2), dtype=rays_d.dtype)
        result[..., 0] = self.C0
        if deg > 0:
            x, y, z = rays_d.unbind(-1)
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if deg > 1:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)

                if deg > 2:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)

                    if deg > 3:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result
