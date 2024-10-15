import jittor as jt
import jittor.nn as nn

import numpy as np
# from jnerf.models.position_encoders.neus_encoder.embedder import get_embedder
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        self.cfg = get_cfg()
        d_in = self.cfg.encoder.sdf_encoder.input_dims

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if self.cfg.encoder.sdf_encoder.multires > 0:
            embed_fn = build_from_cfg(self.cfg.encoder.sdf_encoder, ENCODERS)
            input_ch = embed_fn.out_dim
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        multires = self.cfg.encoder.sdf_encoder.multires
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            # strange init 
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        lin.weight = jt.nn.init.gauss_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        lin.bias = jt.nn.init.constant_(lin.bias, -bias)
                    else:
                        lin.weight = jt.nn.init.gauss_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        lin.bias = jt.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    lin.bias = jt.nn.init.constant_(lin.bias, 0.0)
                    lin.weight[:, 3:] = jt.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    lin.weight[:, :3] = jt.nn.init.gauss_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    lin.bias = jt.nn.init.constant_(lin.bias, 0.0)
                    lin.weight = jt.nn.init.gauss_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    lin.weight[:, -(dims[0] - 3):] = jt.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    lin.bias = jt.nn.init.constant_(lin.bias, 0.0)
                    lin.weight = jt.nn.init.gauss_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def execute(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = jt.concat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return jt.concat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.execute(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.execute(x)

    def gradient(self, x):

        y = self.sdf(x)
        gradients = jt.grad(
            y,
            x,
            retain_graph=True)

        return gradients

    def get_all(self, x, is_training=True):
        with jt.enable_grad():
            x.start_grad()
            tmp = self.execute(x)
            y, feature = tmp[:, :1], tmp[:, 1:]
            gradients = jt.grad(
                y,
                x,
                retain_graph=True)
            if not is_training:
                return y.detach(), feature.detach(), gradients.detach()
            else:
                return y, feature, gradients


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 squeeze_out=True,
                 network_type=None,
                 output_bias=0.0,
                 output_scale=1.0):
        super().__init__()

        self.cfg = get_cfg()

        # nothing to do with pos encoder in original JNeuS
        # but, our method has a positional encoder for xyz
        # TODO: add position encoder
        d_in = d_in

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        assert network_type is not None

        if self.cfg.encoder[network_type].multires > 0:
            embedview_fn = build_from_cfg(self.cfg.encoder[network_type], ENCODERS)
            input_ch = embedview_fn.out_dim
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.output_bias = output_bias
        self.output_scale = output_scale

    def execute(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = jt.concat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = jt.concat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = jt.concat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.output_scale * (x + self.output_bias)
        if self.squeeze_out:
            x = jt.sigmoid(x)
        return x


# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()

        self.cfg = get_cfg()

        self.D = D
        self.W = W
        self.d_in = self.cfg.encoder.nerf_pos_encoder.input_dims
        self.d_in_view = self.cfg.encoder.nerf_dir_encoder.input_dims
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if self.cfg.encoder.nerf_pos_encoder.multires > 0:
            self.embed_fn = build_from_cfg(self.cfg.encoder.nerf_pos_encoder, ENCODERS)
            self.input_ch = self.embed_fn.out_dim

        if self.cfg.encoder.nerf_dir_encoder.multires > 0:
            self.embed_fn_view = build_from_cfg(self.cfg.encoder.nerf_dir_encoder, ENCODERS)
            self.input_ch_view = self.embed_fn_view.out_dim

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def execute(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.concat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = nn.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.variance = jt.Var(init_val)

    def execute(self, x):
        return jt.ones([len(x), 1]) * jt.exp(self.variance * 10.0)


@NETWORKS.register_module()
class NeuralTORecon(nn.Module):
    def __init__(self, nerf_network, sdf_network, variance_network, rendering_network, rendering_network_inner, beta,
                 alpha_factor, *args, **kw):
        super().__init__(*args, **kw)
        self.nerf_outside = NeRF(**nerf_network)
        self.sdf_network = SDFNetwork(**sdf_network)
        self.deviation_network = SingleVarianceNetwork(**variance_network)
        self.color_network = RenderingNetwork(**rendering_network)
        self.beta_network = nn.Parameter(jt.Var(beta))
        self.factor = nn.Parameter(jt.Var(alpha_factor))  # define the learnable extinction coefficient
        self.color_network_inner = RenderingNetwork(**rendering_network_inner)


@NETWORKS.register_module()
class NeuralTOMaterial(nn.Module):

    def __init__(self, diffuse_albedo_network, trans_albedo_network, roughness_network, *args, **kw):
        super().__init__(*args, **kw)
        self.diffuse_albedo_network = RenderingNetwork(**diffuse_albedo_network)
        self.trans_albedo_network = RenderingNetwork(**trans_albedo_network)
        self.roughness_network = RenderingNetwork(**roughness_network)

    def execute(self, points, normals, feature_vectors, output_gradient=False):
        diffuse_albedo = self.diffuse_albedo_network(points, normals, -normals, feature_vectors).abs()[..., [2, 1, 0]]
        trans_albedo = self.trans_albedo_network(points, normals, None, feature_vectors).abs().clamp(0, 1)
        specular_roughness = self.roughness_network(points, normals, None, feature_vectors).abs() + 0.01
        result = {}
        # for i in ["roughness_grad"]:
        #     if output_gradient:
        #         r_output = jt.ones_like(specular_roughness, requires_grad=False)
        #         r_grad = jt.autograd.grad(
        #             outputs=specular_roughness,
        #             inputs=points,
        #             grad_outputs=r_output,
        #             create_graph=True,
        #             retain_graph=True,
        #             only_inputs=True,
        #             allow_unused=True
        #         )[0]
        #     else:
        #         r_grad = jt.zeros_like(specular_roughness)

        #     result[i] = r_grad
        result["diffuse_albedo"] = diffuse_albedo
        result["trans_albedo"] = trans_albedo
        result["specular_roughness"] = specular_roughness
        return result