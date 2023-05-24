import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = (256, 256)


def read_image(image_path):
    pipeline = transforms.Compose(
        [transforms.Resize((img_size)),
         transforms.ToTensor()])

    img = Image.open(image_path).convert('RGB')
    img = pipeline(img).unsqueeze(0)
    return img.to(device, torch.float)


def save_image(tensor, image_path):
    toPIL = transforms.ToPILImage()
    img = tensor.detach().cpu().clone()
    img = img.squeeze(0)
    img = toPIL(img)
    img.save(image_path)


# Hyperparameters
style_img = read_image('dldemos/StyleTransfer/picasso.jpg')
content_img = read_image('dldemos/StyleTransfer/dancing.jpg')

default_content_layers = ['conv_4']
default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_weight = 1e4
content_weight = 1


class ContentLoss(torch.nn.Module):

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram(x: torch.Tensor):
    # x is a [n, c, h, w] array
    n, c, h, w = x.shape

    features = x.reshape(n * c, h * w)
    features = torch.mm(features, features.T) / n / c / h / w
    return features


class StyleLoss(torch.nn.Module):

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = gram(target.detach()).detach()

    def forward(self, input):
        G = gram(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(torch.nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).to(device).reshape(-1, 1, 1)
        self.std = torch.tensor(std).to(device).reshape(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_model_and_losses(content_img, style_img, content_layers, style_layers):
    num_loss = 0
    expected_num_loss = len(content_layers) + len(style_layers)
    content_losses = []
    style_losses = []

    model = torch.nn.Sequential(
        Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, torch.nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, torch.nn.ReLU):
            name = f'relu_{i}'
            layer = torch.nn.ReLU(inplace=False)
        elif isinstance(layer, torch.nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, torch.nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(
                f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img)
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)
            num_loss += 1

        if name in style_layers:
            target_feature = model(style_img)
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)
            num_loss += 1

        if num_loss >= expected_num_loss:
            break

    return model, content_losses, style_losses


input_img = torch.randn(1, 3, *img_size, device=device)
model, content_losses, style_losses = get_model_and_losses(
    content_img, style_img, default_content_layers, default_style_layers)

input_img.requires_grad_(True)
model.requires_grad_(False)

optimizer = optim.LBFGS([input_img])
steps = 0
prev_loss = 0
while steps <= 1000 and prev_loss < 100:

    def closure():
        with torch.no_grad():
            input_img.clamp_(0, 1)
        global steps
        global prev_loss
        optimizer.zero_grad()
        model(input_img)
        content_loss = 0
        style_loss = 0
        for ls in content_losses:
            content_loss += ls.loss
        for ls in style_losses:
            style_loss += ls.loss
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()
        steps += 1
        if steps % 50 == 0:
            print(f'Step {steps}:')
            print(f'Loss: {loss}')
            save_image(input_img, f'work_dirs/output_{steps}.jpg')
        prev_loss = loss
        return loss

    optimizer.step(closure)
with torch.no_grad():
    input_img.clamp_(0, 1)
save_image(input_img, 'work_dirs/output.jpg')
