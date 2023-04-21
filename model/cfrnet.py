import torch
import torch.nn as nn
import numpy as np


class CFR(nn.Module):
    def __init__(self, in_dim, args):
        super().__init__()
        if args.imbalance_func in ['lin_disc', 'mmd2_lin']:
            enc_activation = None
        else:
            enc_activation = nn.ELU(inplace=True)
        self.encoder = MLP(
            num_layers=args.enc_num_layers,
            in_dim=in_dim,
            h_dim=args.enc_h_dim,
            out_dim=args.enc_out_dim,
            activation=enc_activation,
            dropout=args.enc_dropout,
        )

        self.bool_outhead = args.bool_outhead
        ## CFR (Shalit+; ICML2017)
        if self.bool_outhead: 
            self.outhead_Y1 = MLP(
                num_layers=args.oh_num_layers,
                in_dim=args.enc_out_dim,
                h_dim=args.oh_h_dim,
                out_dim=args.oh_out_dim,
                dropout=args.oh_dropout
            )
            self.outhead_Y0 = MLP(
                num_layers=args.oh_num_layers,
                in_dim=args.enc_out_dim,
                h_dim=args.oh_h_dim,
                out_dim=args.oh_out_dim,
                dropout=args.oh_dropout
            )

            self.params = (
                list(self.encoder.parameters())
                + list(self.outhead_Y1.parameters())
                + list(self.outhead_Y0.parameters())
            )
        ## BNN (Johansson+; ICML2016)
        else:
            self.decoder = MLP(
                num_layers=args.oh_num_layers,
                in_dim=args.enc_out_dim + 1,
                h_dim=args.oh_h_dim,
                out_dim=args.oh_out_dim, 
                dropout=args.oh_dropout
            )

            self.params = (
                list(self.encoder.parameters())
                + list(self.decoder.parameters())
            )

    def encode(self, x):
        x_enc = self.encoder(x)
        return x_enc

    def forward(self, x, a):
        x_enc = self.encoder(x)
        if self.bool_outhead:
            _t_ind = torch.where(a == 1)[0] ## a: treatment vector in torch.Tensor
            _c_ind = torch.where(a == 0)[0] 
            _ind = torch.argsort(torch.concatenate([_t_ind, _c_ind], 0))  
            #a_np = a.cpu().detach().numpy()
            #_t_ind = np.where((a_np == 1))[0]
            #_c_ind = np.where((a_np == 0))[0]
            #_ind = np.argsort(np.concatenate([_t_ind, _c_ind], 0))
            y1_hat = self.outhead_Y1(x_enc[_t_ind])
            y0_hat = self.outhead_Y0(x_enc[_c_ind])
            y_hat = torch.cat([y1_hat, y0_hat])[_ind]
        else:
            y_hat = self.decoder(torch.cat((x_enc, a.reshape((x_enc.shape[0], 1))), 1))
        return y_hat

    def predict(self, x, a_val):
        x_enc = self.encoder(x)
        if self.bool_outhead:
            if a_val == 0:
                return self.outhead_Y0(x_enc)
            else:
                return self.outhead_Y1(x_enc)
        else:
            if a_val == 0:
                return self.decoder(torch.cat((x_enc, torch.zeros(x.shape[0]).reshape((x_enc.shape[0], 1))), 1))
            else:
                return self.decoder(torch.cat((x_enc, torch.ones(x.shape[0]).reshape((x_enc.shape[0], 1))), 1))

class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        num_layers,
        h_dim,
        out_dim,
        activation=nn.ELU(inplace=True),
        dropout=0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout = dropout

        bool_nonlin = False if self.activation is None else True
        layers = []
        for i in range(num_layers - 1):
            layers.extend(
                self._layer(
                    h_dim if i > 0 else in_dim,
                    h_dim,
                    bool_nonlin,
                )
            )
        layers.extend(self._layer(h_dim, out_dim, False))

        self.sequential = nn.Sequential(*layers)

        # # init
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #         bound = 1 / np.sqrt(fan_in)
        #         nn.init.uniform_(m.bias, -bound, bound)

    def _layer(self, in_dim, out_dim, activation=True):
        if activation:
            return [
                nn.Linear(in_dim, out_dim),
                self.activation,
                nn.Dropout(self.dropout),
            ]
        else:
            return [
                nn.Linear(in_dim, out_dim),
            ]

    def forward(self, x):
        out = self.sequential(x)
        return out
