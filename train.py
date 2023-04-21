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

def main(idx):
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

    Data = DataLoader(args=args, idx=idx)
    tr_loader = Data.tr_loader
    val_loader = Data.val_loader ### validation data are used for MSE loss evaluation
    te_loader = Data.te_loader
    in_loader = Data.in_loader ### training + validation data, including true Y0 and Y1. validation data can be used for hyperparameter tuning based on true PEHE

    model = CFR(Data.in_dim, args)
    p = Data.p_tr # p = n1 / (n0 + n1): proportion of treated individuals in training data

    if args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print("Error: optimizer = " + str(args.optim) + " is not defined", file=sys.stderr)
        sys.exit(1)           

    loss_func = 'l2' # set 'ce' for dataset with binary outcome
    imbalance_func = args.imbalance_func
    reg_alpha = args.reg_alpha

    writer = SummaryWriter(log_dir="./tb_logs")
    print('Training started ...')
    for epoch in range(args.epochs):
        for ind, a, x, y in tr_loader:        
            ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)
            a = a.reshape((a.shape[0], 1)); y = y.reshape((y.shape[0], 1))
            y_hat = model(x, a)
            x_enc = model.encode(x)
            loss = objective(y, y_hat, x_enc, a, p, reg_alpha, loss_func=loss_func, imbalance_func=imbalance_func, bool_weighting=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        reg_alpha = reg_alpha * args.reg_decay ** epoch
        print("loss: " + str(loss.detach().numpy()))
        writer.add_scalar("Training/loss", loss.detach().numpy(), epoch)

        # evaluate mean L1/CE loss on validation data
        if epoch % 10 == 0:
            val_loss = []
            model.eval()
            with torch.no_grad():
                for ind, a, x, y in val_loader:
                    ind, a, x, y = ind.to(device), a.to(device), x.to(device), y.to(device)
                    a = a.reshape((a.shape[0], 1)); y = y.reshape((y.shape[0], 1))
                    y_hat = model(x, a)
                    if loss_func == 'l1':
                        val_loss.append(torch.abs(y_hat - y))
                    elif loss_func == 'l2':
                        val_loss.append(torch.square(y_hat - y))                        
                    elif loss_func == 'ce':
                        y_hat = 0.995 / (1.0 + torch.exp(- y_hat)) + 0.0025 # ?                        
                        val_loss.append(y * torch.log(y_hat) + (1.0 - y) * torch.log(1.0 - y_hat))
                    else:
                        print("Error: loss_func = " + str(loss_func) + " is not implemented", file=sys.stderr)
                        sys.exit(1)                           
            val_loss = torch.cat(val_loss, dim=1)
            print("val_loss (wo/ penalty terms): " + str((torch.mean(val_loss)).detach().numpy()))
            writer.add_scalar("Validation/loss (wo/ penalty terms)", (torch.mean(val_loss)).detach().numpy(), epoch) 
            model.train()
    
        # evaluate PEHE on test data
        if epoch % 10 == 0:
            pehe = []
            model.eval()
            with torch.no_grad():
                for ind, x, y0, y1 in te_loader:
                    ind, x, y0, y1 = ind.to(device), x.to(device), y0.to(device), y1.to(device)
                    y0 = y0.reshape((y0.shape[0], 1)); y1 = y1.reshape((y1.shape[0], 1))
                    y0_hat = model.predict(x, 0) ## predict Y0 by model(x, 0)
                    y1_hat = model.predict(x, 1) ## predict Y1 by model(x, 1)
                    tau_hat = y1_hat - y0_hat
                    tau = y1 - y0
                    pehe.append(torch.square(tau_hat - tau)) ## Is this correct for binary outcomes?
            pehe = torch.cat(pehe, dim=0)
            mean_sqrt_pehe = torch.sqrt(torch.mean(pehe)).detach().numpy()
            print("test_pehe: " + str(mean_sqrt_pehe))
            writer.add_scalar("Test/PEHE", mean_sqrt_pehe, epoch)
            model.train()

        # # evaluate PEHE on training + validation data
        # if epoch % 10 == 0:
        #     pehe = []
        #     model.eval()
        #     with torch.no_grad():
        #         for ind, x, y0, y1 in in_loader:
        #             ind, x, y0, y1 = ind.to(device), x.to(device), y0.to(device), y1.to(device)
        #             y0 = y0.reshape((y0.shape[0], 1)); y1 = y1.reshape((y1.shape[0], 1))
        #             y0_hat = model.predict(x, 0) ## predict Y0 by model(x, 0)
        #             y1_hat = model.predict(x, 1) ## predict Y1 by model(x, 1)
        #             tau_hat = y1_hat - y0_hat
        #             tau = y1 - y0
        #             # if epoch == 4900:
        #             #     print(y0_hat)
        #             #     print(y0)
        #             #     sys.exit()
        #             pehe.append(torch.square(tau_hat - tau)) ## Is this correct for binary outcomes?
        #     pehe = torch.cat(pehe, dim=0)
        #     mean_sqrt_pehe = torch.sqrt(torch.mean(pehe)).detach().numpy()
        #     print("in_pehe: " + str(mean_sqrt_pehe))
        #     writer.add_scalar("In/PEHE", mean_sqrt_pehe, epoch)
        #     model.train()

    print('Training ended')
    writer.close()

    return mean_sqrt_pehe

if __name__ == '__main__':
    num_of_datasets = 3
    res_sqrt_pehe = np.zeros(num_of_datasets)
    for idx in range(1, num_of_datasets + 1):
        res_sqrt_pehe[idx-1] = main(idx)

    print(str(np.mean(res_sqrt_pehe)) + " +- " + str(np.std(res_sqrt_pehe)))
