from .models import Classifier, save_model, ClassificationLoss
from .datasets.classification_dataset import load_data
import torch
import torchvision
import torchvision.transforms as T
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = Classifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_func = ClassificationLoss()
    # loss_func = torch.nn.CrossEntropyLoss(torch.tensor([0.05, 0.2, 0.05, 0.35, 0.35]))
    loss_func.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.92, weight_decay=1e-4)
    epochs = 10

    ##train_trans = T.Compose((T.ToPILImage(), T.ColorJitter(0.8, 0.3), T.RandomHorizontalFlip(), T.RandomCrop(32), T.ToTensor())) # 96
    ##val_trans = T.Compose((T.ToPILImage(), T.CenterCrop(size=32), T.ToTensor()))
    data = load_data("classification_data/train", shuffle=False, batch_size=128, num_workers=2, transform_pipeline="aug")
    val = load_data("classification_data/val", shuffle=False, batch_size=128, num_workers=2, transform_pipeline="default")

    # data = load_data('classification_data/train', transform_pipeline="aug")
    # val = load_data('classification_data/val', transform_pipeline="default")

    for epoch in range(epochs):
        model.train()
        count = 0
        total_loss = 0
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_func(y_pred, y.long())
            total_loss = total_loss + loss.item()
            count += 1
            loss.backward()
            optim.step()
            optim.zero_grad()
        print("Epoch: " + str(epoch) + ", Loss: " + str(total_loss/count))

        model.eval()
        count = 0
        accuracy = 0
        for image, label in val:
          image = image.to(device)
          label = label.to(device)
          pred = model(image)
          accuracy = accuracy + (pred.argmax(1) == label).float().mean().item()
          count += 1
        print("Epoch: " + str(epoch) + ", Accuracy: " + str(accuracy/count))

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)