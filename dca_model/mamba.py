import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from .attention import CBAM_Atten

class GumbelTopKSorter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, rank_score, tau=0.5, hard=True):
        B, C, H, W = features.shape
        L = H * W
        _, indices = torch.topk(rank_score, k=L, dim=-1, sorted=True)
        # back_indices = torch.argsort(indices, dim=-1)  # [B, C, L]
        _, back_indices = torch.topk(indices, k=rank_score.shape[-1], dim=-1, largest=False)

        indices = indices.expand(-1, C, -1)  # [B, C, L]
        back_indices = back_indices.expand(-1, C, -1)  # [B, C, L]

        flat_features = features.view(B, C, L)
        sorted_features = torch.gather(flat_features, dim=-1, index=indices)  # 收集排序后特征

        ctx.save_for_backward(indices, back_indices, flat_features)
        ctx.tau = tau
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W
        return sorted_features, back_indices

    @staticmethod
    def backward(ctx, grad_output, _): # [B, C, L]
        indices, back_indices, features = ctx.saved_tensors
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W
        L = H * W

        grad_mean = grad_output.mean(dim=1, keepdim=True)
        back_indices_mean = back_indices[:, 0, :][:, None, :] # [B, 1, L]

        grad_features = torch.gather(grad_output, dim=-1, index=back_indices).view(B, C, H, W)
        grad_mean = torch.gather(grad_mean, dim=-1, index=back_indices_mean).view(B, 1, H*W)


        return grad_features, grad_mean

class SS2D(nn.Module):
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


class MHSS2D_CBAM(SS2D):
    def __init__(self, d_model, d_state=16, dropout=0., CBAM_ker_size=7):
        super().__init__()
        # 固定为原实验的四头 MHSS2DCasCBAM3，保留原参数名称和计算顺序。
        bias = False
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = 2 * d_model
        self.K = 4
        self.inner_dim = self.d_inner // self.K
        self.scan_dim = self.inner_dim * self.K
        self.dt_rank = math.ceil(d_model / 16)
        self.mix_proj = nn.Linear(self.inner_dim * self.K, self.d_inner, bias=bias)
        self.in_proj = nn.Linear(d_model, self.scan_dim + self.d_inner, bias=bias)
        self.rank_conv_list = nn.ModuleList([
            CBAM_Atten(self.inner_dim, kernel_size=CBAM_ker_size) for _ in range(self.K)
        ])
        self.conv2d = nn.Conv2d(self.scan_dim, self.scan_dim, groups=self.scan_dim,
                                bias=True, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        x_proj = [nn.Linear(self.inner_dim, self.dt_rank + self.d_state * 2, bias=False)
                  for _ in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0)) # (K=4, N, inner)
        dt_projs = [self.dt_init(self.dt_rank, self.inner_dim) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K=4, inner)
        self.A_logs = self.A_log_init(self.d_state, self.inner_dim, copies=self.K, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.inner_dim, copies=self.K, merge=True) # (K=4, D, N)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.forward_core = self.forward_core_cas3

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        meta = kwargs.get('meta', None)


        xz = self.in_proj(x)
        x, z = torch.split(xz, [self.scan_dim, self.d_inner], dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.act(self.conv2d(x)) # (b, d, h, w)


        y = self.forward_core(x) # (b, c, l)

        # assert y == torch.float32
        # print(f'y: {y.shape}')
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)

        # spilt into 4 branches
        # y_spilt = torch.split(y, self.inner_dim, dim=-1)  # list of (B, H, W, inner_dim)
        # pca = PCA(n_components=3, whiten=True)

        # for i, phase in enumerate(y_spilt):
        #     for j, _feature in enumerate(phase):
        #         feature = _feature.squeeze().detach().cpu().view(self.inner_dim, -1).permute(1, 0)

        #         # pca.fit(feature)
        #         # projected_image = torch.from_numpy(pca.transform(feature.numpy())).view(H, W, 3)
        #         km = KMeans(n_clusters=3,
        #                     init='k-means++',
        #                     n_init='auto',
        #                     max_iter=300,
        #                     random_state=42,
        #                     )
        #         projected_image = km.fit_predict(feature.numpy())[..., None]

        #         projected_image = torch.from_numpy(projected_image).view(H, W, -1)
        #         projected_image = projected_image.permute(2, 0, 1)
        #         print(np.unique(projected_image))
        #         # 一共分了五类，给每一类一个颜色可视化出来
        #         for i in range(5):
        #             projected_image[0][projected_image[0]==i] = i * 51

        #         # projected_image = torch.nn.functional.sigmoid(projected_image).permute(2, 0, 1)
        #         projected_image = F.interpolate(projected_image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)

        #         input_img = self._get_input_img(meta)
        #         os.makedirs('viz', exist_ok=True)
        #         projected_image = projected_image.permute(1,2,0)
        #         print(f'projected_image: {projected_image.shape}')
        #         print(np.unique(projected_image))

        #         cv2.imwrite(f'viz/{self.name}_dmamba_casX_feature_{_feature.shape[0]}_{i}_{j}.png',
        #                     projected_image.numpy())


        # 统计总图片数量
        # all_features = [(i, j, f) for i, phase in enumerate(y_spilt) for j, f in enumerate(phase)]
        # N = len(all_features) + 1

        # fig, axes = plt.subplots(1, N, figsize=(4*N, 4))
        # if N == 1:
        #     axes = [axes]
        # axes[0].imshow(input_img)

        # for idx, (i, j, _feature) in enumerate(all_features):

        #     feature = _feature.squeeze().detach().cpu().view(self.inner_dim, -1).permute(1, 0)
        #     feature_np = feature.numpy()  # [HW, C]

        #     pca = PCA(n_components=3)
        #     rgb = pca.fit_transform(feature_np)  # [HW, 3]

        #     # ---- 归一化到 0~1 范围 ----
        #     rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

        #     # ---- reshape 成 H×W×3 ----
        #     rgb = torch.tensor(rgb).view(H, W, 3).permute(2, 0, 1).unsqueeze(0)

        #     # ---- resize to 256 ----
        #     rgb = F.interpolate(rgb, size=(256, 256), mode='bilinear', align_corners=False)
        #     rgb = rgb[0].permute(1, 2, 0).numpy()

        #     # ---- 子图绘制 ----
        #     ax = axes[idx+1]
        #     ax.imshow(rgb)
        #     ax.set_title(f"P{i}-F{j}")
        #     ax.axis("off")

        # plt.tight_layout()
        # plt.savefig(f'viz/{self.name}_all_features_row.png', dpi=300)
        # plt.show()




        y = self.out_norm(self.mix_proj(y))


        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    def forward_core_cas3(self, x: torch.Tensor, meta=None):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = self.K


        x_split = torch.split(x, C//K, dim=1)
        cas_v = torch.zeros(B, C//K, H, W, device=x.device, dtype=x.dtype)

        cat_out_y = []

        Ds = self.Ds.float().view(-1)
        Ds = Ds.view(K, -1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        As = As.view(K, -1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        dt_projs_bias = dt_projs_bias.view(K, -1)


        # (b, c, l) for each

        for i in range(K):

            if i != 0:
                input_x = x_split[i] + cas_v
            else:
                input_x = x_split[i]

            rank_logic = self.rank_conv_list[i](input_x) # (b, 1, h, w)

            # input_x = rank_logic.expand(-1, self.inner_dim, -1, -1) * inptu_x
            rank_score = rank_logic.view(B, -1, L)
            xs, _ = GumbelTopKSorter.apply(input_x, rank_score)

            # rank_indexes = torch.argsort(rank_score, dim=-1, descending=True)
            _, rank_indexes = torch.topk(rank_score, k=rank_score.shape[-1], dim=-1)
            # # # rank_indexes =  torchsort.soft_rank(rank_score.squeeze(), regularization_strength=1.0)

            # back_indexes = torch.argsort(rank_indexes, dim=-1)
            _, back_indexes = torch.topk(rank_indexes, k=rank_score.shape[-1], dim=-1, largest=False)
            # # # back_indexes = torchsort.soft_rank(rank_indexes, regularization_strength=1.0)

            # back_indexes = torchsort.soft_rank(rank_indexes, dim=-1)
            rank_indexes = rank_indexes.expand(-1, self.inner_dim, -1) # (b, c, l)
            back_indexes = back_indexes.expand(-1, self.inner_dim, -1)

            xs = xs[:,None,...]

            # if self.cas_XS_type == 1:
            #     xs = torch.gather(input_x.view(B, -1, L), dim=-1, index=rank_indexes)[:,None,...]
            # else:
            #     xs = torch.gather(x_split[i].view(B, -1, L), dim=-1, index=rank_indexes)[:,None,...]

            # save_path = f'rank_score/{self.name}'
            # os.makedirs(save_path, exist_ok=True)
            # heat_map = rank_score[0].cpu().numpy()
            # indexs = rank_indexes[0][0][None,...].cpu().numpy()
            # # heat_map = np.array([i for i in range(0,H*W)])[::-1].reshape(H,W)
            # # heat_map = (heat_map - heat_map.min()) / (heat_map.max() - heat_map.min())
            # heat_map_rank = np.zeros_like(heat_map)
            # heat_map_rank[:,indexs] = np.arange(H*W)
            # heat_map_rank = (heat_map_rank - heat_map_rank.min()) / (heat_map_rank.max() - heat_map_rank.min())
            # heat_map = (heat_map_rank * 255).astype('uint8').reshape((H,W))

            # # heat_map = cv2.equalizeHist(heat_map)
            # heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
            # heat_map_high = cv2.resize(heat_map, (256, 256))
            # cv2.imwrite(f'{save_path}/{i}.png', heat_map)
            # cv2.imwrite(f'{save_path}/{i}_high.png', heat_map_high)

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight[i][None, ...])
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, 1, -1, L), self.dt_projs_weight[i][None, ...])

            xs = xs.float().view(B, -1, L)  # (b, k * d, l)

            dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
            Bs = Bs.float().view(B, 1, -1, L)
            Cs = Cs.float().view(B, 1, -1, L)

                # xs = input_x.view(B, -1, L)[:,None,...]

            out_y = self.selective_scan(
                xs, dts, As[i],
                Bs, Cs, Ds[i], z=None,
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
                return_last_state=False)


            out_y = out_y.view(B, -1, L) # b c l

            out_y = torch.gather(out_y, dim=-1, index=back_indexes)
            cat_out_y.append(out_y)
            cas_v = rearrange(out_y,  'b c (h w)-> b c h w', h=H, w=W)

        out = torch.cat(cat_out_y, dim=1)



        return out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim, name=None, CBAM_ker_size=7):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.name = name
        self.self_attention = MHSS2D_CBAM(hidden_dim, CBAM_ker_size=CBAM_ker_size)
        self.drop_path = DropPath(0.)

    def forward(self, input, meta=None):
        x = input + self.drop_path(self.self_attention(self.ln_1(input), meta=meta))
        return x
