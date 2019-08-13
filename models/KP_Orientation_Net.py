import torch.nn as nn
from torchvision import models
import torch


class CoarseRegressor(nn.Module):
    """
    Coarse Key-Point Detecor Network
    """
    def __init__(self, n1=20):
        super(CoarseRegressor, self).__init__()
        self.N1 = n1
        # VGG Convolutional Layers
        self.A1 = nn.Sequential(*list(list(models.vgg16_bn(pretrained=True).children())[0].children())[:7])
        self.A2 = nn.Sequential(*list(list(models.vgg16_bn(pretrained=True).children())[0].children())[7:14])
        self.A3 = nn.Sequential(*list(list(models.vgg16_bn(pretrained=True).children())[0].children())[14:24])
        self.A4 = nn.Sequential(*list(list(models.vgg16_bn(pretrained=True).children())[0].children())[24:34])
        self.A5 = nn.Sequential(*list(list(models.vgg16_bn(pretrained=True).children())[0].children())[34:])
        # Coarse Regressors
        self.A6 = nn.Sequential(nn.Conv2d(512, 512, 1, padding=0), nn.BatchNorm2d(512), nn.ReLU())
        self.A6to7 = nn.Sequential(nn.Conv2d(512, self.N1 + 1, 1, padding=0), nn.BatchNorm2d(self.N1 + 1), nn.ReLU())
        self.A3to7 = nn.Sequential(nn.Conv2d(256, self.N1 + 1, 1, padding=0), nn.BatchNorm2d(self.N1 + 1), nn.ReLU())
        self.A4to7 = nn.Sequential(nn.Conv2d(512, self.N1 + 1, 1, padding=0), nn.BatchNorm2d(self.N1 + 1), nn.ReLU())
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        """
        Coarse Key-Point regression forward pass
        :param x: The input tensor of shape B * 3 * 224 * 224
        :return: The predicted Heatmaps of 20 Key-Points plus the map for background. Shape = B * 21 * 56 * 56
        """
        x = self.A1(x)  # B * 64 * 112 * 112
        x = self.A2(x)  # B * 128 * 56 * 56
        x = self.A3(x)  # B * 256 * 28 * 28
        res2 = self.A4(x)  # B * 512 * 14 * 14
        return self.Up(self.Up(self.Up(self.A6to7(self.A6(self.A5(res2)))) + self.A4to7(res2)) + self.A3to7(x))


class FineRegressor(nn.Module):
    """
    Key-Point Refinement Network
    """
    def __init__(self, n2=20):
        super(FineRegressor, self).__init__()
        self.N2 = n2
        self.Normalize = nn.Softmax(dim=2)
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.L1 = nn.Sequential(nn.Conv2d(24, 64, 7), nn.BatchNorm2d(64), nn.ReLU())
        self.HR1 = nn.Sequential(nn.Conv2d(64, 64, 7), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 128, 5),
                                nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256),
                                nn.ReLU(), nn.ConvTranspose2d(256, 128, 5), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.ConvTranspose2d(128, 64, 7), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.L2 = nn.Sequential(nn.ConvTranspose2d(64, self.N2 + 1 , 7), nn.BatchNorm2d(self.N2 + 1), nn.ReLU(),
                                nn.Conv2d(self.N2 + 1, 64, 7), nn.BatchNorm2d(64), nn.ReLU())
        self.L3 = nn.Sequential(nn.Conv2d(64, 64, 7), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 128, 5),
                                nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.res2 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.L4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 5), nn.BatchNorm2d(128), nn.ReLU(),
                                 nn.ConvTranspose2d(128, 64, 7), nn.BatchNorm2d(64), nn.ReLU())
        self.L5 = nn.Sequential(nn.ConvTranspose2d(64, self.N2 + 1, 7), nn.BatchNorm2d(self.N2 + 1), nn.ReLU())
        self.pose_branch1 = nn.Sequential(nn.Conv2d(256, 128, 7), nn.BatchNorm2d(128), nn.ReLU(),
                                         nn.Conv2d(128, 64, 7), nn.BatchNorm2d(64), nn.ReLU())
        self.pose_branch2 = nn.Sequential(nn.Conv2d(64, 32, 7), nn.BatchNorm2d(32), nn.ReLU())
        self.FC = nn.Sequential(nn.Linear(2048, 256, bias=True), nn.Dropout(0.5), nn.Linear(256, 8, bias=True))

    def forward(self, x):
        """
        Key-Point Refinement forward pass
        :param x: The input tensor of shape B * 24 * 56 * 56
        :return : kp: The refined Heatmaps of 20 Key-Points of Shape = B * 20 * 56 * 56
                  pose: The predicted orientation of vehicle
        """
        x = self.L1(x)  # B * 64 * 50 * 50
        x = self.L2(self.res1(x) + self.HR1(x))  # B * 20 * 50 * 50
        joint = self.L3(x)  # B * 256 * 40 * 40
        # Key Point Estimation
        kp = self.L5(self.res2(x) + self.L4(joint))  # B * 20 * 56 * 56
        B, C, H, W = kp.shape
        kp = self.Normalize(kp.view(B, C, W * H))
        kp = kp.view(B, C, H, W)
        # Orientation Estimation
        pose = self.pose_branch1(joint)  # B * 64 * 28 * 28
        pose = self.MaxPool(pose)  # B * 64 * 14 * 14
        pose = self.pose_branch2(pose)  # B * 32 * 8 * 8
        pose = pose.view(-1, 2048)  # B * 2048
        pose = self.FC(pose)  # B * 8
        return kp, pose


class KeyPointModel(nn.Module):
    """
    End-to-End Key-Point Regression models
    """
    def __init__(self):
        super(KeyPointModel, self).__init__()
        self.coarse_estimator = CoarseRegressor()
        self.refinement = FineRegressor()

    def forward(self, x1, x2):
        """
        Key-Point Estimation forward pass
        :param x1: The input tensor of shape B * 3 * 224 * 224
        :param x2: The input tensor of shape B * 3 * 56 * 56
        :return: coarse_kp: The coarse heatmaps of size B * 21 * 56 * 56
                 fine_kp: The refined heatmaps of size B * 20 * 56 * 56
                 orientation: The predicted orientation of the vehicle of size B * 8
        """
        coarse_kp = self.coarse_estimator(x1)
        x2 = torch.cat((x2, coarse_kp), dim=1)
        fine_kp, orientation = self.refinement(x2)

        return coarse_kp, fine_kp, orientation
