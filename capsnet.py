from __future__ import print_function

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='PyTorch Dynamic Routing Between Capsules paper implementation')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 5)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--routing-iterations', type=int, default=3, metavar='N',
                    help='number of routing iterations in the routing algorithm')
parser.add_argument('--use-decoder', type=bool, default=True, metavar='N',
                    help='if true then decoder network will be appended to the digits layer')
parser.add_argument("--reconstruction-loss-weight", type=float, default=0.0005, metavar='N',
                    help='the weight for the reconstruction loss component of the total loss')
parser.add_argument("--generated-images", type=str, default=os.path.dirname(os.path.realpath(__file__)) + "/generated/",
                    help='the path where sample images from decoder network will be generated')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class RandomShift(object):
    def __call__(self, tensor):
        shift_dir = random.randint(0, 2)

        if shift_dir in [0,2]:
            shift_amount = random.randint(0, 2)
            if shift_amount > 0:
                if random.randint(0, 1) == 0:
                    tensor = torch.cat([tensor, torch.FloatTensor(1, shift_amount,28).zero_()], dim=1)[:,shift_amount:28+shift_amount,:]
                else:
                    tensor = torch.cat([torch.FloatTensor(1, shift_amount, 28).zero_(), tensor], dim=1)[:,0:28,:]

        if shift_dir in [1,2]:
            shift_amount = random.randint(0, 2)
            if shift_amount > 0:
                if random.randint(0, 1) == 0:
                    tensor = torch.cat([tensor, torch.FloatTensor(1, 28, shift_amount).zero_()], dim=2)[:,:,shift_amount:28+shift_amount]
                else:
                    tensor = torch.cat([torch.FloatTensor(1, 28, shift_amount).zero_(), tensor], dim=2)[:,:,0:28]

        return tensor


transform = transforms.Compose([transforms.ToTensor(), RandomShift()])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform),
                                           batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, transform=transform),
                                          batch_size=args.test_batch_size, shuffle=True, **kwargs)


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2)
        self.W = nn.Parameter(torch.randn(32 * 6 * 6, 10, 16, 8))
        if args.use_decoder:
            self.fc1 = nn.Linear(16 * 10, 512)
            self.fc2 = nn.Linear(512, 1024)
            self.fc3 = nn.Linear(1024, 784)

    def forward(self, x, labels):
        x = F.relu(self.conv1(x))

        x = self.primary_capsules(x)
        x = x.view(x.size()[0], self.W.size()[-1], -1)
        x = self.squash(x).transpose(2, 1)

        x = (self.W @ x[:, :, None, :, None]).squeeze()
        x = self.routing(x, args.routing_iterations)

        decoder_output = None

        if args.use_decoder:
            decoder_output = x * labels[:, :, None]
            decoder_output = decoder_output.view(decoder_output.size()[0], -1)
            decoder_output = F.relu(self.fc1(decoder_output))
            decoder_output = F.relu(self.fc2(decoder_output))
            decoder_output = torch.sigmoid(self.fc3(decoder_output))

        x = x.norm(p=2, dim=2)
        x = F.softmax(x, dim=-1)

        return x, decoder_output

    def squash(self, s):
        len_sqr = torch.sum(s ** 2, dim=-1, keepdim=True)
        len = len_sqr ** 0.5
        return (len_sqr / (1.0 + len_sqr)) * (s / len)

    def routing(self, u_hat, routing_iterations):
        b = Variable(torch.FloatTensor(u_hat.size()[0:3]).zero_())
        b = b.cuda() if args.cuda else b

        for r in range(routing_iterations):
            c = F.softmax(b, dim=-1)
            s = torch.sum(c[:, :, :, None] * u_hat, dim=1)
            v = self.squash(s)
            if r != routing_iterations - 1:
                b = b + (u_hat[:, :, :, None, :] @ v[:, None, :, :, None]).squeeze()

        return v

    def loss(self, output, target, decoder_output, image, is_mean=True):
        L = target * torch.pow(F.relu(0.9 - output), 2) + \
            0.5 * (1.0 - target) * torch.pow(F.relu(output - 0.1), 2)
        margin_loss = L.sum(dim=1)
        margin_loss = margin_loss.mean() if is_mean else margin_loss.sum()

        reconstruction_loss = 0
        if decoder_output is not None:
            reconstruction_loss = F.mse_loss(decoder_output, image.view(image.size()[0], -1))
            reconstruction_loss = reconstruction_loss.mean() if is_mean else reconstruction_loss.sum()

        loss = margin_loss + args.reconstruction_loss_weight * reconstruction_loss
        return loss


model = CapsNet().cuda() if args.cuda else CapsNet()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.01)

def idx_to_one_hot(vector, num_classes):
    one_hot = torch.FloatTensor(vector.size()[0], num_classes).zero_()
    one_hot = one_hot.cuda() if args.cuda else one_hot
    one_hot.scatter_(dim=1, index=vector[:, None], value=float(1))
    return one_hot


def prepare_input(data, target):
    data, target = data.cuda() if args.cuda else data, target.cuda() if args.cuda else target
    return Variable(data), Variable(idx_to_one_hot(target, 10)), target


def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        image, one_hot, target_index = prepare_input(data, target)
        optimizer.zero_grad()
        digits_output, decoder_output = model(image, one_hot)
        loss = model.loss(digits_output, one_hot, decoder_output, image, is_mean=True)
        loss.backward()
        optimizer.step()

        pred = torch.max(digits_output, dim=1)[1]
        correct += pred.data.eq(target_index).int().sum()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, batch accuracy {:.1f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(),
                       100. * correct.item() / ((args.log_interval if batch_idx != 0 else 1.) * len(target))))
            correct = 0

    lr_scheduler.step()


def test():
    model.eval()
    test_loss = 0
    correct = 0
    iterations = 0
    for data, target in test_loader:
        iterations += 1
        image, one_hot, target_index = prepare_input(data, target)

        digits_output, decoder_output = model(image, one_hot)
        loss = model.loss(digits_output, one_hot, decoder_output, image, is_mean=False)
        test_loss += loss

        pred = torch.max(digits_output, dim=1)[1]
        correct += pred.data.eq(target_index).int().sum()

    sample_idx = random.randint(0, len(decoder_output) - 1)
    generated_image = torch.cat([decoder_output[sample_idx].view(1, 28, 28).data.cpu(),
                                 data[sample_idx].view(1, 28, 28).cpu()], dim=1)
    generated_image = transforms.functional.to_pil_image(generated_image.cpu())
    generated_image.save('%s/epoch-%s-image-%s.png' % (args.generated_images, epoch, sample_idx))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.data.item(), correct, len(test_loader.dataset), (100. * float(correct) / len(test_loader.dataset))))


if __name__ == '__main__':
    if not os.path.exists(args.generated_images):
        os.mkdir(args.generated_images)
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        with torch.no_grad():
            test()
