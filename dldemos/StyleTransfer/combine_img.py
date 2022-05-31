import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_size = (256, 256)


def read_image(image_path):
    pipeline = transforms.Compose(
        [transforms.Resize((img_size)),
         transforms.ToTensor()])

    img = Image.open(image_path)
    img = pipeline(img).unsqueeze(0)
    return img.to(device, torch.float)


def save_image(tensor, image_path):
    toPIL = transforms.ToPILImage()
    img = tensor.detach().cpu().clone()
    img = img.squeeze(0)
    img = toPIL(img)
    img.save(image_path)


style_img = read_image('dldemos/StyleTransfer/picasso.jpg')
content_img = read_image('dldemos/StyleTransfer/dancing.jpg')

input_img = torch.randn(1, 3, *img_size, device=device)
input_img.requires_grad_(True)
optimizer = optim.LBFGS([input_img])
steps = 0
while steps <= 10:

    def closure():
        global steps
        optimizer.zero_grad()
        loss = F.mse_loss(input_img, style_img) + F.mse_loss(
            input_img, content_img)
        loss.backward()
        steps += 1
        if steps % 1 == 0:
            print(f'Step {steps}:')
            print(f'Loss: {loss}')

        return loss

    optimizer.step(closure)

save_image(input_img, 'work_dirs/output.jpg')
