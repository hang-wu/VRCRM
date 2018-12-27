from gumbel.gumbel_multilabel import *
from torch.autograd import Variable
import torch
import torch.nn as nn

def train(max_epoch, bandit_train_loader, fgan_loader, hnet, Dnet_xy, steps_fgan, is_cuda=False, is_gumbel_hard=False):
    opt_h = torch.optim.Adam(params=hnet.parameters(), weight_decay = 0.001)
    opt_h2 = torch.optim.Adam(params=hnet.parameters(), weight_decay = 0.001)
    opt_d = torch.optim.Adam(params=Dnet_xy.parameters(), weight_decay=0.01)
    for epoch in max_epoch:
        for ele in bandit_train_loader:
            X, s_labels, s_log_prop, s_loss, y = ele

            # Convert data to Torch Variable format
            idx = Variable(torch.LongTensor(y))
            X = Variable(X.type(torch.FloatTensor), requires_grad = False)
            s_labels = Variable(s_labels.type(torch.FloatTensor))
            s_log_prop = Variable(s_log_prop.type(torch.FloatTensor))
            s_loss = Variable(s_loss.type(torch.FloatTensor))
            y = Variable(y)

            opt_h.zero_grad()
            hnet.train()
            prob = hnet(X)


            # compute expected loss
            h = (1-s_labels) * (1-prob) + s_labels * prob
            stacked = torch.stack((prob, 1-prob)) #[2, bs, nl], with idx [bs, nl]
            # idx[i,j]=1, select stack[0,i,j]
            idx = idx.view((1, idx.size(0), idx.size(1)))
            h = torch.gather(stacked,dim=0,index=1-idx).squeeze() # [bs, nl]

            prob_per_instance = torch.sum(h, dim=1)
            log_prob_per_instance = torch.log(prob_per_instance+1e-9)
            log_IS = log_prob_per_instance - s_log_prop
            loss = torch.mean(s_loss * torch.exp(log_IS))

            # update paras of hnet
            loss.backward()
            opt_h.step()
            opt_h.zero_grad()

            hnet.eval()
            hnet.train()

            # f-gan training
            # hnet.eval()
            fgan_counter = 0
            #for i in range(3):
            if steps_fgan > 0:
                for ele in fgan_loader:
                    fgan_counter += 1

                    # dataset convertion
                    X, s_labels, s_log_prop, s_loss, y = ele
                    X = Variable(X.type(torch.FloatTensor), requires_grad=False)
                    s_labels = Variable(s_labels.type(torch.FloatTensor))
                    s_log_prop = Variable(s_log_prop.type(torch.FloatTensor))
                    s_loss = Variable(s_loss.type(torch.FloatTensor))
                    y = Variable(y.type(torch.FloatTensor))
                    # y: [bs, n_labels]

                    prob = hnet(X)

                    logits = torch.cat((prob, 1-prob),dim=1)
                    G_sample = gumbel_softmax(logits=logits, temperature=1, hard=is_gumbel_hard, cuda=is_cuda) # [bs, 2, n_label]
                    G_sample = G_sample.view((G_sample.size(0), -1)) # [bs, 2*n_label]
                    G_sample = torch.cat((G_sample,X),dim=-1)
                    D_fake = Dnet_xy(G_sample)
                    transformed_y = torch.cat((s_labels,1-s_labels),dim=1) # TODO: here
                    transformed_y = transformed_y.view(transformed_y.size(0),-1)
                    transformed_y = torch.cat((transformed_y, X),dim=-1)
                    D_real = Dnet_xy(transformed_y)

                    D_loss = -(torch.mean(D_fake) - torch.mean(0.25 * D_real ** 2 + D_real))
                    D_loss.backward()
                    opt_d.step()
                    opt_d.zero_grad()
                    opt_h.zero_grad()

                    prob = hnet(X)

                    logits = torch.cat((prob, 1-prob),dim=1)
                    # generator h
                    G_sample = gumbel_softmax(logits=logits, temperature=1, hard=config.gumbel_hard, cuda=is_cuda)  # [bs, 2, n_label]
                    G_sample = G_sample.view((G_sample.size(0), -1))
                    G_sample = torch.cat((G_sample, X), dim=-1)
                    D_fake = Dnet_xy(G_sample)
                    G_loss = torch.mean(D_fake)
                    G_loss.backward()
                    opt_h.step()

                    opt_d.zero_grad()
                    opt_h.zero_grad()
                    opt_h2.zero_grad()

                    if fgan_counter > steps_fgan:
                        break

if __name__ == '__main__':
    hnet = nn.Sequential(
        nn.Linear(config.nin, config.n1),
        nn.BatchNorm1d(config.n1),
        nn.ReLU(),
        nn.Linear(config.n1, config.n2),
        nn.BatchNorm1d(config.n2),
        nn.ReLU(),
        nn.Linear(config.n2, config.n3),
        nn.BatchNorm1d(config.n3),
        nn.ReLU(),
        nn.Linear(config.n3, config.nout),
        nn.Sigmoid())

    # the net only takes in
    d_dim = config.nin + config.nout*2
    Dnet_xy = nn.Sequential(
        nn.Linear(d_dim, d_dim//2),
        nn.BatchNorm1d(d_dim//2),
        nn.ReLU(),
        nn.Linear(d_dim//2, 1)
        #nn.Sigmoid()
    )
