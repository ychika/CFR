import sys
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from config import args
from load_data import DataLoader
from obj_func import objective
from model.cfrnet import CFR

print(args)

def main():
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    random.seed(args.train_seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(args.gpu if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    print('Data type: {}'.format(args.data))
    print('Dataset Seed: {}'.format(args.data_seed))

    Data = DataLoader(args=args)
    tr_loader = Data.tr_loader
    val_loader = Data.val_loader
    te_loader = Data.te_loader
    in_loader = Data.in_loader

    model = CFR(Data.in_dim, args)
    p = Data.p_tr

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Error: optimizer = " + str(args.optim) + " is not defined", file=sys.stderr)
        sys.exit(1)           

    loss_func = 'l1' # set 'ce' for dataset with binary outcome
    imbalance_func = args.imbalance_func
    reg_alpha = args.reg_alpha

    writer = SummaryWriter(log_dir="./tb_logs")
    print('START Training ...')
    for epoch in range(args.epochs):
        for ind, a, x, y in tr_loader:        
            ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)
            y_hat = model(x, a)
            loss = objective(y, y_hat, x, a, p, reg_alpha, loss_func=loss_func, imbalance_func=imbalance_func, bool_weighting=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        reg_alpha = reg_alpha * args.reg_decay ** epoch
        print("loss: " + str(loss.detach().numpy()))
        writer.add_scalar("Training/loss", loss.detach().numpy(), epoch)
        #writer.add_scalar("Validation/loss", val_loss[epoch], epoch)
        #writer.add_scalar("Test/loss", te_loss[epoch], epoch)
    writer.close()

if __name__ == '__main__':
    main()
