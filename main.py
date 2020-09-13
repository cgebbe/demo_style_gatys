import torch
import torchvision
from PIL import Image
import collections
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def main():
    # load images as tensors
    tensors = {}
    device = torch.device('cuda')
    width, height = 512,512 # 320,320
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((width, height)),
        torchvision.transforms.ToTensor(),
    ])
    dirname = 'pair5'
    for filename in ['content', 'style']:
        im = Image.open('{}/{}.png'.format(dirname, filename))
        tensors[filename] = transforms(im).to(device).unsqueeze(0)
        plot(tensors[filename], filename, do_unnormalize=False)

    # init network with hooks
    # net = Net_mobilev2().to(device)
    net = Net_VGG().to(device)
    hook_per_name = net.apply_hook()
    net.eval()

    # get "target" feature values
    feature_outputs = dict()
    for key in tensors.keys():
        for name, hook in hook_per_name.items():
            hook.clear()
        out = net(tensors[key])
        feature_outputs[key] = {k: v.out.detach().clone() for k, v in hook_per_name.items()}

    # init random image to optimize
    x = torch.rand(1, 3, height, width).to(device)
    # x = tensors['content'].detach().clone()
    x.requires_grad = True
    optim = torch.optim.Adam([x], lr=0.05)

    # perform optimization
    nepochs = 300
    pbar = tqdm.trange(nepochs+1)
    for i in pbar:
        # plot results
        if i % (nepochs // 3) == 0 and True:
            fig, ax = plot(x, title="step={}".format(i), do_unnormalize=False)
            plt.show()
            plt.close(fig)

        # perform forward pass
        x.data.clamp_(0, 1)
        out = net(x)
        feature_outputs['optim'] = {k: v.out for k, v in hook_per_name.items()}
        loss = calc_loss(feature_outputs, x)

        # update pbar
        info_str = ""
        loss_tot = 0
        for k, v in loss.items():
            info_str += "{}={:.3f},".format(k, v)
            loss_tot += v
        pbar.set_description(info_str)

        # perform optimization
        optim.zero_grad()
        loss_tot.backward()
        optim.step()


def plot(x, title=None, do_unnormalize=False):
    """
    Plots an image tensor
    :param x: Normalized GPU tensor
    :return:
    """
    # convert normalized tensor to default numpy array
    x = x.detach().clone().cpu()[0, ...]
    x = x.permute(1, 2, 0).numpy()  # <C,H,W> to <H,W,C>
    if do_unnormalize:  # from [x0,x1] to [0,1]
        mean = np.asarray([0.485, 0.456, 0.406])  # <3>
        std = np.asarray([0.229, 0.224, 0.225])  # <3>
        for i in range(3):
            x[:, :, i] = (x[:, :, i] * std[i]) + mean[i]
    x *= 255.0
    x = np.clip(x, 0, 255)
    x = np.round(x).astype(np.uint8)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(x)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def calc_loss(fout, x):
    loss = collections.defaultdict(int)

    # regularization
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = x[:, :, :, 1:] - x[:, :, :, -1:]
    loss['regu'] = (dx ** 2).mean() + (dy ** 2).mean()
    loss['regu'] *= 1/0.3

    # content loss
    for lname in ['block4_conv2']:
        diff = fout['optim'][lname] - fout['content'][lname]
        loss['content'] += (diff ** 2).mean()
    loss['content'] *= 1/0.2

    # style loss via gram_matrix
    def calc_gram_matrix(x):
        """ Calculates the gram-matrix <C,C> for a tensor <N,C,H,W> """
        N, C, H, W = x.shape
        x = x.view(N, C, -1)  # <N,C,H*W>
        x_transpose = x.transpose(1, 2)  # <N,H*W,C>
        mat = torch.bmm(x, x_transpose)  # <N,C,C>
        mat /= (H * W)
        return mat

    for lname in ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]:
        m_optim = calc_gram_matrix(fout['optim'][lname])
        m_style = calc_gram_matrix(fout['style'][lname])
        diff = m_optim - m_style
        loss['style'] += (diff ** 2).mean()
    loss['style'] *= 1/2.5 * 10

    return loss


class SaveOutputHook():
    def __init__(self):
        self.out = None

    def __call__(self, module, inputs, outputs):
        self.out = outputs

    def clear(self):
        self.out = None


class Net_VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._net = torchvision.models.vgg19(pretrained=True).features
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

        # replace MaxPool layers with AveragePool Layers
        for name, module in self._net.named_children():
            if isinstance(module, torch.nn.MaxPool2d):
                self._net[int(name)] = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = [self.normalize(t) for t in x]
        x = torch.cat(x, dim=0)
        if x.ndim < 4:
            x = x.unsqueeze(0)
        x = self._net.forward(x)
        return x

    def apply_hook(self):
        hook_per_name = {}
        hookname_per_layer = {"22": "block4_conv2",  # for content
                              "1": "block1_conv1",  # this and sequent for style
                              "6": "block2_conv1",
                              "11": "block3_conv1",
                              "20": "block4_conv1",
                              "29": "block5_conv1",
                              }
        for i, (layer_name, layer) in enumerate(self._net.named_children()):
            if layer_name in hookname_per_layer:
                print("Hooking layer {}: {}={}".format(i, layer_name, layer))
                hookname = hookname_per_layer[layer_name]
                hook = SaveOutputHook()
                hook_per_name[hookname] = hook
                layer.register_forward_hook(hook)
        assert len(hook_per_name) == len(hookname_per_layer)
        return hook_per_name


if __name__ == '__main__':
    main()
