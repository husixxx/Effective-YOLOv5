# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync
from utils.plots import feature_visualization
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.autoanchor import check_anchor_order
from models.experimental import *
from models.common import *
from models.pruned_common import *
import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)], indexing='ij')
        else:
            yv, xv = torch.meshgrid([torch.arange(ny, device=d), torch.arange(nx, device=d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None, mask_bn=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value

        self.mask_bn = mask_bn
        if self.mask_bn is not None:
            self.model, self.save, self.from_to_map = parse_pruned_model(
                self.mask_bn, deepcopy(self.yaml), ch=[ch])  # model, savelist
        else:
            self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist

        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _profile_one_layer(self, m, x, dt):
        c = isinstance(m, Detect)  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            LOGGER.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             LOGGER.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):
        """Fuse Conv2d() and BatchNorm2d() layers inplace."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn") and m.bn is not None:
                try:
                    # Add allow_shape_mismatch=True for pruned models
                    m.conv = fuse_conv_and_bn(m.conv, m.bn, allow_shape_mismatch=True)
                    delattr(m, "bn")
                    m.forward = m.forward_fuse  # update forward
                except Exception as e:
                    LOGGER.warning(f"Error fusing layers: {e}")
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

DetectionModel = Model

def parse_model(d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, ECALayer]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
            if m in {ECALayer}:
                args = [ch[f]]  # Ensure the correct number of input channels is passed
                n = 1  # Uistime sa, že sa neopakujeme
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in [BiFPN_Concat2, BiFPN_Concat3]:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def parse_pruned_model(mask_bn, d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    fromlayer = []  # last module bn layer name
    from_to_map = {}
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    
    # Print available keys in mask_bn for debugging
    LOGGER.info(f"Available BN layers in mask_bn: {list(mask_bn.keys())[:10]}...")
    
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = f"model.{i}"
        
        # Function to safely get BN layer name and check if it exists in mask_bn
        def get_bn_layer(base_name, possible_suffixes):
            for suffix in possible_suffixes:
                full_name = f"{base_name}{suffix}"
                if full_name in mask_bn:
                    return full_name
            # If no match found, print warning and return None
            LOGGER.warning(f"Could not find BN layer for {base_name} with any of {possible_suffixes}")
            LOGGER.info(f"Available keys containing {base_name}: {[k for k in mask_bn.keys() if base_name in k]}")
            return None
        
        if m in [Conv]:
            # Try different possible BN layer naming patterns
            named_m_bn = get_bn_layer(named_m_base, [".bn", ".conv.bn"])
            
            if not named_m_bn:
                LOGGER.warning(f"Skipping pruning for layer {named_m_base} (Conv), BN layer not found")
                # Handle the case when BN layer is not found - use the original width
                c1, c2 = ch[f], args[0] if args else ch[f]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
            else:
                # Normal processing when BN layer is found
                bnc = int(mask_bn[named_m_bn].sum())
                c1, c2 = ch[f], bnc
                args = [c1, c2, *args[1:]]
                
                if i > 0:
                    from_to_map[named_m_bn] = fromlayer[f]
                fromlayer.append(named_m_bn)

        elif m in [C3Pruned]:
            named_m_cv1_bn = get_bn_layer(named_m_base, [".cv1.bn"])
            named_m_cv2_bn = get_bn_layer(named_m_base, [".cv2.bn"])
            named_m_cv3_bn = get_bn_layer(named_m_base, [".cv3.bn"])
            
            # Special handling for problematic layers based on index
            is_problematic_layer = named_m_base in ["model.2", "model.17", "model.21", "model.24"]
            after_bifpn = False
            
            # Check if this C3 comes after a BiFPN layer (which causes dimension issues)
            if i > 0 and i < len(d['backbone'] + d['head']) and \
            (d['backbone'] + d['head'])[i-1][2] in ["BiFPN_Concat2", "BiFPN_Concat3"]:
                after_bifpn = True
                LOGGER.warning(f"C3Pruned after BiFPN at {named_m_base}: Special handling needed")
            
            if not named_m_cv1_bn or not named_m_cv2_bn or not named_m_cv3_bn:
                LOGGER.warning(f"Skipping pruning for layer {named_m_base} (C3Pruned), some BN layers not found")
                c2 = ch[f]
                fromlayer.append(fromlayer[-1] if fromlayer else "")
            elif after_bifpn or is_problematic_layer:
                # Special handling for layers after BiFPN or known problematic layers
                LOGGER.warning(f"Using special conservative pruning for {named_m_base}")
                
                # Get input channels from previous layer
                cv1in = ch[f]
                
                # More conservative channel reduction for these critical layers
                cv1out = max(8, (cv1in * 2 // 3 + 7) // 8 * 8)  # ~2/3 of input channels
                cv2out = cv1out  # Same dimension for cv2
                cv3out = cv1in   # Output matches input for stability
                
                # Set channel mappings
                from_to_map[named_m_cv1_bn] = fromlayer[f]
                from_to_map[named_m_cv2_bn] = fromlayer[f]  # Connect directly to input
                fromlayer.append(named_m_cv3_bn)
                
                # Create consistent bottleneck dimensions
                bottle_args = []
                chin = [cv1out]  # Input to first bottleneck
                
                c3fromlayer = [named_m_cv1_bn]
                
                # Create bottleneck with consistent dimensions
                for p in range(n):
                    bottle_cv1in = chin[-1]
                    bottle_cv1out = bottle_cv1in  # 1:1 ratio for stable bottleneck
                    bottle_cv2out = bottle_cv1in
                    
                    # Record actual BN layers if they exist (for weight transfer)
                    named_m_bottle_cv1_bn = get_bn_layer(f"{named_m_base}.m.{p}", [".cv1.bn"])
                    named_m_bottle_cv2_bn = get_bn_layer(f"{named_m_base}.m.{p}", [".cv2.bn"])
                    
                    if named_m_bottle_cv1_bn and named_m_bottle_cv2_bn:
                        from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[p]
                        from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                        
                    chin.append(bottle_cv2out)
                    bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
                    c3fromlayer.append(named_m_bottle_cv2_bn if named_m_bottle_cv2_bn else c3fromlayer[-1])
                    
                    LOGGER.info(f"Bottleneck {named_m_base}.m.{p}: in={bottle_cv1in}, mid={bottle_cv1out}, out={bottle_cv2out}")
                
                # Set final arguments for C3Pruned
                args = [cv1in, cv1out, cv2out, cv3out, n, args[-1]]
                args.insert(4, bottle_args)
                c2 = cv3out
                n = 1
                
            else:
                # Standard handling for normal C3Pruned layers
                from_to_map[named_m_cv1_bn] = fromlayer[f]
                from_to_map[named_m_cv2_bn] = fromlayer[f]  # Connect directly to input, not cv1
                fromlayer.append(named_m_cv3_bn)

                # Get channel counts from masks
                cv1in = ch[f]
                cv1out = int(mask_bn[named_m_cv1_bn].sum())
                cv2out = int(mask_bn[named_m_cv2_bn].sum())
                cv3out = int(mask_bn[named_m_cv3_bn].sum())
                
                # Round to multiples of 8 for better hardware efficiency
                cv1out = max(8, (cv1out + 7) // 8 * 8)
                cv2out = max(8, (cv2out + 7) // 8 * 8)
                cv3out = max(8, (cv3out + 7) // 8 * 8)
                
                # Debug output
                LOGGER.info(f"C3Pruned {named_m_base}: in={cv1in}, cv1={cv1out}, cv2={cv2out}, cv3={cv3out}")
                
                # Process bottleneck layers
                bottle_args = []
                chin = [cv1out]  # Input to bottleneck is output from cv1
                
                c3fromlayer = [named_m_cv1_bn]
                for p in range(n):
                    named_m_bottle_cv1_bn = get_bn_layer(f"{named_m_base}.m.{p}", [".cv1.bn"])
                    named_m_bottle_cv2_bn = get_bn_layer(f"{named_m_base}.m.{p}", [".cv2.bn"])
                    
                    bottle_cv1in = chin[-1]  # Must match previous output
                    
                    if not named_m_bottle_cv1_bn or not named_m_bottle_cv2_bn:
                        # Use same channel dimensions throughout bottleneck
                        bottle_cv1out = bottle_cv1in
                        bottle_cv2out = bottle_cv1in
                    else:
                        # Use pruned dimensions but ensure they're valid
                        bottle_cv1out = int(mask_bn[named_m_bottle_cv1_bn].sum())
                        bottle_cv2out = int(mask_bn[named_m_bottle_cv2_bn].sum())
                        
                        # For model.2 specifically, force matching dimensions
                        if named_m_base == "model.2":
                            bottle_cv1out = bottle_cv1in
                            bottle_cv2out = bottle_cv1in
                        
                        # Always ensure dimensions are multiples of 8
                        bottle_cv1out = max(8, (bottle_cv1out + 7) // 8 * 8)
                        bottle_cv2out = max(8, (bottle_cv2out + 7) // 8 * 8)
                        
                        from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[p]
                        from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                    
                    # Log dimensions
                    LOGGER.info(f"Bottleneck {named_m_base}.m.{p}: in={bottle_cv1in}, mid={bottle_cv1out}, out={bottle_cv2out}")
                    
                    # Add this layer's output to chin for next layer's input
                    chin.append(bottle_cv2out)
                    bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
                    c3fromlayer.append(named_m_bottle_cv2_bn if named_m_bottle_cv2_bn else c3fromlayer[-1])
                
                # Set C3Pruned arguments
                args = [cv1in, cv1out, cv2out, cv3out, n, args[-1]]
                args.insert(4, bottle_args)
                c2 = cv3out
                n = 1
                

        elif m in [SPPFPruned]:
            named_m_cv1_bn = get_bn_layer(named_m_base, [".cv1.bn"])
            named_m_cv2_bn = get_bn_layer(named_m_base, [".cv2.bn"])
            
            if not named_m_cv1_bn or not named_m_cv2_bn:
                LOGGER.warning(f"Skipping pruning for layer {named_m_base} (SPPFPruned), some BN layers not found")
                c2 = ch[f]
                fromlayer.append(fromlayer[-1] if fromlayer else "")
            else:
                cv1in = ch[f]
                from_to_map[named_m_cv1_bn] = fromlayer[f]
                from_to_map[named_m_cv2_bn] = [named_m_cv1_bn] * 4
                fromlayer.append(named_m_cv2_bn)
                cv1out = int(mask_bn[named_m_cv1_bn].sum())
                cv2out = int(mask_bn[named_m_cv2_bn].sum())
                args = [cv1in, cv1out, cv2out, *args[1:]]
                c2 = cv2out

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m in {ECALayer}:
            args = [ch[f]]  
            n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)
        elif m is Detect:
            # Safely map detect layer inputs
            if len(f) > 0 and f[0] < len(fromlayer):
                from_to_map[named_m_base + ".m.0"] = fromlayer[f[0]]
            if len(f) > 1 and f[1] < len(fromlayer):
                from_to_map[named_m_base + ".m.1"] = fromlayer[f[1]]
            if len(f) > 2 and f[2] < len(fromlayer):
                from_to_map[named_m_base + ".m.2"] = fromlayer[f[2]]
            
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m in [BiFPN_Concat2, BiFPN_Concat3]:
    # Create BiFPN module with dimension adaptation
            bifpn = BiFPN_Concat2() if m is BiFPN_Concat2 else BiFPN_Concat3()
            
            # Detailed source layer tracking
            input_layers = []
            expected_channels = []
            base_channels = []
            
            # Track all possible layer dimensions for detailed diagnostics
            LOGGER.info(f"\n===== {named_m_base} ({m.__name__}) SOURCE LAYER ANALYSIS =====")
            for idx, x in enumerate(f):
                if x < len(ch):
                    base_ch = ch[x]
                    expected_ch = base_ch
                    base_channels.append(base_ch)
                    expected_channels.append(expected_ch)
                    input_layers.append(x)
                    
                    # Get more context about the source layer
                    src_layer_type = d['backbone'][x][2] if x < len(d['backbone']) else d['head'][x-len(d['backbone'])][2] if x-len(d['backbone']) < len(d['head']) else "unknown"
                    
                    # Add detailed information about this input source
                    LOGGER.info(f"  Input {idx}: Layer {x} ({src_layer_type}) → Base channels: {base_ch}, Expected: {expected_ch}")
                    
                    # Check for potential issues with source layer
                    if x > i:
                        LOGGER.warning(f"  ⚠️ Input {idx}: References future layer {x} (current: {i})")
            
            # Log raw total vs calculated conservative estimate
            raw_total = sum(expected_channels)
            LOGGER.info(f"  Raw channel total: {raw_total}")
            
            # For robustness, calculate output channels conservatively
            # This ensures C3 layers that follow will have correct dimensions
            conservative_estimate = 0
            for x in f:
                if x < len(ch):
                    # Add a safety margin to prevent channel mismatches
                    safety_margin = 32 if ch[x] > 100 else 16
                    conservative_estimate += ch[x] + safety_margin
                    LOGGER.info(f"  Layer {x}: {ch[x]} channels + {safety_margin} safety margin")
            
            # Store the original estimate for reference
            original_estimate = sum(expected_channels)
            
            # Use the conservative estimate for downstream layers
            c2 = conservative_estimate
            
            # Log the discrepancy between original and conservative estimates
            percent_increase = ((conservative_estimate - original_estimate) / original_estimate) * 100 if original_estimate > 0 else 0
            LOGGER.info(f"  Conservative estimate: {conservative_estimate} channels (+{percent_increase:.1f}% buffer)")
            
            # Store the expected channel info on the module for runtime verification
            bifpn.expected_channels = c2
            bifpn.source_indices = f
            bifpn.input_channel_estimates = expected_channels
            bifpn.named_base = named_m_base
            
            # Add a hook for runtime channel verification
            def verify_channels_hook(module, inputs, output):
                input_shapes = [inp.shape for inp in inputs[0]]
                actual_channels = [shape[1] for shape in input_shapes]
                total_channels = sum(actual_channels)
                
                # Log detailed comparison
                LOGGER.info(f"\n===== RUNTIME CHANNEL VERIFICATION: {module.named_base} =====")
                LOGGER.info(f"  Expected inputs: {module.input_channel_estimates}")
                LOGGER.info(f"  Actual inputs: {actual_channels}")
                LOGGER.info(f"  Total expected: {sum(module.input_channel_estimates)}, Actual: {total_channels}")
                
                for idx, (expected, actual) in enumerate(zip(module.input_channel_estimates, actual_channels)):
                    if expected != actual:
                        src_idx = module.source_indices[idx]
                        LOGGER.warning(f"  ⚠️ Source {idx} (layer {src_idx}): Expected {expected}, Got {actual} channels")
                
                if module.expected_channels != total_channels:
                    LOGGER.warning(f"  ⚠️ Total channels mismatch: Expected {module.expected_channels}, Got {total_channels}")
                    if hasattr(output, 'shape'):
                        LOGGER.info(f"  Output shape: {output.shape}")
            
            # Register the hook
            bifpn.register_forward_hook(verify_channels_hook)
            
            # Set the module arguments
            args = [bifpn]
            
            # Create proper mapping for weight transfer
            input_bns = []
            for x in f:
                if x < len(fromlayer):
                    input_bns.append(fromlayer[x])
                    # Log what we're connecting
                    if isinstance(fromlayer[x], list):
                        LOGGER.info(f"  Connecting from layer {x}: Multiple connections {fromlayer[x]}")
                    else:
                        LOGGER.info(f"  Connecting from layer {x}: {fromlayer[x]}")
                else:
                    input_bns.append("")
                    LOGGER.warning(f"  Missing connection for source {x}, layer index out of range")
            
            fromlayer.append(input_bns)
            LOGGER.info("=" * 60)
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m in [Focus]:
            # Focus modules in YOLOv5s have batch norm inside a nested Conv module
            named_m_bn = get_bn_layer(named_m_base, [".conv.bn"])
            
            if not named_m_bn:
                LOGGER.warning(f"Skipping pruning for layer {named_m_base} (Focus), BN layer not found")
                c1, c2 = ch[f], args[0] if args else 64  # Default to original width
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)
            else:
                bnc = int(mask_bn[named_m_bn].sum())
                c1, c2 = ch[f], bnc
                args = [c1, c2, *args[1:]]
                fromlayer.append(named_m_bn)
        else:
            LOGGER.info(f"Unrecognized module type: {m}")
            c2 = ch[f]
            fromtmp = fromlayer[-1] if fromlayer else ""
            fromlayer.append(fromtmp)

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    
    LOGGER.info(f"Created from_to_map with {len(from_to_map)} connections")
    return nn.Sequential(*layers), sorted(save), from_to_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(FILE.stem, opt)
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    if opt.profile:
        img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    # Test all models
    if opt.test:
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # LOGGER.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
