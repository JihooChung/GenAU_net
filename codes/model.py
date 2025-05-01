#See required.py to see all the libraries used in this code
#-----------Model Architecture-----------
class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
                                  nn.Conv2d(ch_in, ch_out,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(ch_out, ch_out,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
                                nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out,
                                         kernel_size=3,stride=1,
                                         padding=1, bias=True),
                                nn.BatchNorm2d(ch_out),
                                nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x = self.up(x)
        return x

class GABlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, gene_info_size=7):
        super().__init__()

        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )

        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )

        self.gene = nn.Sequential(
            nn.Linear(gene_info_size, f_int),
            nn.Sigmoid()
        )

        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, gene_info):
        g1 = self.w_g(g)
        x1 = self.w_x(x)

        gene_info = self.gene(gene_info)
        gene_info = gene_info.view(-1, gene_info.size(1), 1, 1)

        psi = self.relu((g1+x1)+gene_info)
        psi = self.psi(psi)

        return psi*x

class GenAU_net(nn.Module):
    def __init__(self, n_classes=1, in_channel=3, out_channel=1):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(ch_in=in_channel, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = GABlock(f_g=512, f_l=512, f_int=256)
        self.upconv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = GABlock(f_g=256, f_l=256, f_int=128)
        self.upconv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.att3 = GABlock(f_g=128, f_l=128, f_int=64)
        self.upconv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.att2 = GABlock(f_g=64, f_l=64, f_int=32)
        self.upconv2 = ConvBlock(ch_in=128, ch_out=64)

        self.conv_1x1 = nn.Conv2d(64, out_channel,
                                  kernel_size=1, stride=1, padding=0)
    def forward(self, x, gene_info):
        # encoder
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoder + concat
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4, gene_info=gene_info)
        d5 = torch.concat((x4, d5), dim=1)
        d5 = self.upconv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3, gene_info=gene_info)
        d4 = torch.concat((x3, d4), dim=1)
        d4 = self.upconv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2, gene_info=gene_info)
        d3 = torch.concat((x2, d3), dim=1)
        d3 = self.upconv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1, gene_info=gene_info)
        d2 = torch.concat((x1, d2), dim=1)
        d2 = self.upconv2(d2)

        d1 = self.conv_1x1(d2)

        return d1

genaunet = GenAU_net(n_classes=1).to(device)
opt = torch.optim.Adamax(genaunet.parameters(), lr=1e-3)

#-----------Model Training-----------

def dice_coef_metric(inputs, target):

    if inputs.ndim == 4:
        inputs = inputs.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    intersection = (inputs * target).sum(axis=(1, 2))
    union = inputs.sum(axis=(1, 2)) + target.sum(axis=(1, 2))

    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean()

def compute_dice(model, loader, threshold=0.3):
    valloss = 0

    with torch.no_grad():

        for i_step, (data, target, gene_info) in enumerate(loader):

            data = data.to(device)
            target = target.to(device)
            target = (target > 0).float()
            gene_info = torch.tensor(gene_info, dtype=torch.float32).to(device)

            outputs = model(data, gene_info)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0
            picloss = dice_coef_metric(out_cut, target.cpu().numpy())
            valloss += picloss

    return valloss / len(loader)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs):
    print(f"[INFO] Model is initializing... {model_name}")

    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()

        losses = []
        train_dices = []

        for i_step, (data, target, gene_info) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target = target.to(device)
            target = (target > 0).float()

            gene_info = np.array(gene_info)
            gene_info = torch.tensor(gene_info, dtype=torch.float32).to(device)

            outputs = model(data, gene_info)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice = dice_coef_metric(out_cut, target.cpu().numpy())

            loss = train_loss(outputs, target)

            losses.append(loss.item())
            train_dices.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_mean_dice = compute_dice(model, val_loader)

        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_dices).mean())
        val_history.append(val_mean_iou)

        print("Epoch [%d]" % (epoch))
        print("\nMean DICE on train:", np.array(train_dices).mean(),
              "\nMean DICE on validation:", val_mean_dice)

    return loss_history, train_history, val_history

%%time
num_ep = 50

lh, th, vh = train_model("Attention UNet", genaunet, train_dataloader, val_dataloader, DiceLoss(), opt, False, num_ep)
