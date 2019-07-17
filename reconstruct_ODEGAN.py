from IPython import display

import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
from torchvision import transforms, datasets
from torchdiffeq import odeint
import matplotlib.pyplot as plt
# from torchdiffeq import odeint_adjoint as odeint

ts_size = 101//0.1
latent_size = 100
#data = ?
batch_size = 32


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            #print("bougnoul")
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



class data_iterable:
    def __init__(self, data, batch_size):


        self.current_count = 0
        self.batch_size = batch_size
        self.datasize = data.shape[0]
        self.bpe = self.datasize//self.batch_size
        # indices = np.random.choice(np.arange(self.datasize), (self.bpe, self.batch_size))

        # data = (data - np.mean(data, axis = 2))/np.std(data,axis = 2)
        self.data =data# torch.from_numpy(data).float()
        #self.target = torch.from_numpy(target).float()

    def __iter__(self):
        # rows = np.random.randint(0, self.data.shape[0]-1,size = self.batch_size)
        self.indices = np.random.choice(self.datasize, (self.bpe, self.batch_size))
        # for j, row in enumerate(rows):
        #     self.samples[j] = torch.t(self.data[row])
        #     self.targets[j] = self.target[row].float()
        return self

    def __next__(self): # Python 3: def __next__(self)
        self.current_count += 1
        if self.current_count > self.bpe:
            self.current_count = 0
            raise StopIteration
        else:
            return self.data[self.indices[self.current_count-1]]#, self.target[self.indices[self.current_count-1]]
            # return self.samples, self.targets#self.samples[self.current_count-1], self.targets[self.current_count-1]



# class ODEfunc(nn.Module):
#
#     def __init__(self, latent_dim=4, nhidden=20):
#         super(ODEfunc, self).__init__()
#         self.elu = nn.ELU(inplace=True)
#         self.fc1 = nn.Linear(latent_dim, nhidden)
#         self.fc2 = nn.Linear(nhidden, nhidden)
#         self.fc3 = nn.Linear(nhidden, latent_dim)
#         self.tanh = torch.nn.Tanh()
#
#         # self.fc1 = nn.Linear(latent_dim, latent_dim)
#         # self.fc2 = nn.Linear(latent_dim, latent_dim)
#         self.nfe = 0
#
#
#     def forward(self, t, x):
#         self.nfe += 1
#         # print(self.nfe)
#         out = self.fc1(x)
#         out = self.elu(out)
#         out = self.fc2(out)
#         out = self.elu(out)
#         out = self.fc3(out)
#         # return self.tanh(out)
#         return out

def norm(dim, mn = 10):
    # return nn.GroupNorm(min(10, dim), dim)
    # return nn.GroupNorm(min(10, dim), dim)
    return nn.GroupNorm(min(mn,dim), dim)
    # return nn.GroupNorm(min(100, dim), dim)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ConcatConv1d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        # tt = torch.ones_like(x[:, :1, :, :]) * t
        tt = torch.ones_like(x[:, :1,:]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

    # def forward(self, t, x):
    #     tt = torch.ones_like(x[:, :1, :, :]) * t
    #     ttx = torch.cat([tt, x], 1)
    #     return self._layer(ttx)
class Linear_t(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Linear_t, self).__init__()

        self._layer = nn.Linear(dim_in+1, dim_out)

    def forward(self, t, x):
        # tt = torch.ones_like(x[:, :1, :, :]) * t
        tt = torch.ones_like(x[:, :,:1]) * t
        # tt = torch.ones_like(x[:, :,:]) * t
        ttx = torch.cat([tt, x], 2)
        # print('ttx: ', ttx.size())
        return self._layer(ttx)

class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        # self.norm1 = norm(dim)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ELU(inplace=True)
        self.relu = nn.Tanh()
        self.linear1 = Linear_t(1,10)
        self.linear2 = Linear_t(10,10)
        self.linear3 = Linear_t(10,1)
        # self.linear4 = Linear_t(dim,dim)

        # self.conv1 = ConcatConv1d(dim, dim, 3, 1, 1)
        # self.norm2 = norm(dim)
        # self.conv2 = ConcatConv1d(dim, dim, 3, 1, 1)
        # self.norm3 = norm(dim)
        # self.conv3 = ConcatConv1d(dim, dim, 3, 1, 1)
        # self.norm4 = norm(dim)
        self.nfe = 0


    def forward(self, t, x):
        self.nfe += 1
        # out = self.norm1(x)
        # out = self.relu(out)
        # # out = self.relu(x)
        # out = self.conv1(t, out)
        # out = self.norm2(out)
        # out = self.relu(out)
        # out = self.conv2(t, out)
        # out = self.norm3(out)
        # out = self.relu(out)
        # out = self.conv3(t, out)
        # out = self.norm3(out)
        # out = self.relu(out)
        # out = self.norm4(out)
#-----------------
        # x = x.reshape(x.shape[0],1,-1)
        out = self.relu(x)
        out = self.linear1(t,out)
        out = self.relu(out)
        out = self.linear2(t,out)
        out = self.relu(out)
        out = self.linear3(t,out)
        # out = self.linear4(t,out)

        return out



class ODEBlock(nn.Module):

    def __init__(self, odefunc, int_time):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = int_time#torch.from_numpy(np.arange(-1,10,0.01))#torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        #out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        # out = odeint(self.odefunc, x, self.integration_time, rtol=1e-1, atol=1e-1).permute(1,0,2)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3).permute(1,0,2,3)
        # odeint(func, z0, samp_ts).permute(1,0,2)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=2, obs_dim=1, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden) # input to hidden ?
        #self.h2o = nn.Linear(nhidden, latent_dim * 2) # hidden to output ?
        self.h2o = nn.Linear(nhidden, latent_dim)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        #h = torch.tanh(self.i2h(combined))
        h = self.i2h(combined)
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)




class Discriminator(nn.Module):

    def __init__(self, input_size=1, hidden_size=10, nbatch = 16, num_layers=1):
        super(Discriminator, self).__init__()

        self.hidden_size = hidden_size
        self.nbatch = nbatch
        self.input_size = input_size
        # self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(latent_dim, nhidden)

        self.gru = nn.GRU(input_size,self.hidden_size, num_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, self.input_size)
        self.fc2 = nn.Linear(int(ts_size), self.input_size)
        #self.sigmoid = torch
    def forward(self, z):
        # combined = torch.cat((x, h), dim=1)
        out, _ = self.gru(z)
        #out = out.float()
        out = self.fc1(out)
        # out = torch.transpose(out, 1, 2)
        out = out.squeeze()
        out = self.fc2(out)
        out = torch.sigmoid(out)

        # print(out.size())
        #print(out.size())
        #print(h_.size())
        # out = torch.cat((out,h), dim = 1)
        # out =
        # out = self.fc1(z)
        # out = self.relu(out)
        # out = self.fc2(out)
        # return out, hn
        return out
#TODO : initWeights
    # def initWeights(self)

'''
    def initHidden(self):
        #self.hidden = torch.zeros(1,self.nbatch, self.hidden_size).cuda()
        return torch.zeros(self.input_size,self.nbatch, self.hidden_size)
'''


class Decoder(nn.Module):

    def __init__(self, latent_dim=2, obs_dim=1, nhidden=20):
        super(Decoder, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ELU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, nhidden)
        self.fc4 = nn.Linear(nhidden, obs_dim)
        # self.fc4 = nn.Linear(nhidden, nhidden)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=latent_dim, out_channels=int(latent_dim/4), kernel_size=3,
        #         stride=1, padding=1, bias=False
        #     ),
        #     # norm(int(latent_dim/2),int(latent_dim/2)),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.pool = nn.AdaptiveAvgPool2d((100, obs_dim))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=int(latent_dim/2), out_channels=int(latent_dim/4), kernel_size=3,
        #         stride=1, padding=1, bias=False
        #     ),
        #     norm(int(latent_dim/4),int(latent_dim/4)),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=int(latent_dim/4), out_channels=1, kernel_size=3,
        #         stride=1, padding=1, bias=False
        #     ),
        #     norm(1,1),
        #     nn.LeakyReLU(0.2, inplace=True)
        # )

        # self.norm1 = norm(dim)
        # self.norm2 = norm(dim)
        # self.norm3 = norm(dim)
        # self.fc3 = nn.Linear(nhidden, nhidden)
        # self.fc4 = nn.Linear(nhidden, nhidden)
        # self.fc5 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        # print('inpuuuuuuut size: ', z.size())
        out = z.squeeze()

        out = self.relu(out)

        # out = self.conv1(out)
        # out = self.conv2(out)
        # out = self.conv3(out)


        out = self.fc1(out)
        out = self.tanh(out)
        # out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        # out = self.pool(out)

        # # out = self.tanh(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        # # out = self.tanh(out)
        # out = self.relu(out)
        # out = self.fc4(out)
        # # out = self.tanh(out)
        # out = self.relu(out)
        # out = self.fc5(out)
        # out = self.tanh(out)
        # out = out.squeeze(dim = 3)
        return out


# Noise
def noise(size):
    n = Variable(torch.randn(size, 1, latent_size))
    if torch.cuda.is_available(): return n.cuda()
    return n





D = Discriminator(latent_size)
loss = nn.BCELoss()

num_epochs = 40


def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1. Train on Real Data
    #h = D.initHidden().cuda()

    prediction_real = D(real_data)#.view(-1,1)
    # prediction_real = prediction_real.contiguous().view(-1,1)
    fake_target = fake_data_target(real_data.size(0))

    # Calculate error and backpropagate

    error_real = loss(prediction_real, real_data_target(real_data.size(0)))

    error_real.backward()

    # 2. Train on Fake Data

    prediction_fake = D(fake_data)#.view(-1,1)
    # Calculate error and backpropagate

    error_fake = loss(prediction_fake, fake_data_target(fake_data.shape[0]))

    error_fake.backward()

    optimizer.step()
    # print('D param: ', torch.mean(D.parameters().grad.data))

    for d in D.parameters():
        print('D param: ', torch.mean(d.grad))
        # p.data.clamp_(-0.01, 0.01)


    # return d_train_loss, prediction_real, prediction_fake
    return error_real + error_fake, prediction_real, prediction_fake
    return (0, 0, 0)

def train_generator(optimizer, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    # h = D.initHidden().cuda()
    prediction = D(fake_data)#.view(-1,1)#.squeeze()
    ####Ajouter qqchose en rapport avec la somme ou la diff des data generées
    # Calculate error and backpropagate
    error_fake = loss(prediction, real_data_target(prediction.size(0)))
        # p.data.clamp_(-0.01, 0.01)
    '''
    for p in G.parameters():
        p.data.clamp_(-0.01, 0.01)
    '''
    error_fake.backward()
    # Update weights with gradients
    optimizer.step()
    for i,g in enumerate(G.parameters()):
        print('G param {0}: '.format(i), torch.mean(g.grad))

    # Return error
    return error_fake
    #return error


class ExpReplay():

    def __init__(self, memory_size = 2000, init_vec = None):
        super(ExpReplay, self).__init__()
        self.memory = init_vec
        self.size = 0
        self.max_size = memory_size

    def update(self,data):
        if self.memory is None:
            self.memory = data.clone().detach()#torch.tensor(data)#data.copy_()
        else:
            self.memory = torch.cat([self.memory, data.clone().detach()], 0)

        if self.size > self.max_size:
            self.pop(data.shape[0])

        self.size = self.memory.shape[0]

    # randomly delete parts of memory
    def pop(self, nb):
        self.memory = self.memory[np.random.choice(self.memory.shape[0], self.memory.shape[0]-nb, replace=False)]
        print("prout")

    def get_sample(self,samp_size):
        rand_samp = np.random.choice(self.memory.shape[0], samp_size, replace=False)
        # rand_samp = torch.randint(0, self.memory.shape[0], samp_size)

        return self.memory[rand_samp]

num_test_samples = 4
rand_size = 1

z_fixed = Variable(torch.randn(num_test_samples, latent_size,rand_size)).cuda()


#logger = Logger(model_name='DCGAN', data_name='MNIST')
# path =  '../datasets_ideas/gaussian_process/gaussian_process3.npy'
path =  '../datasets_ideas/gaussian_process/cos_wave.npy'
data = np.load(path).astype(np.float32)
data = data.reshape((1000,-1,1))
print(data.shape)
# data = data[:,1,:].reshape(1000,-1,1)
'''
for i in range(data.shape[0]):
    data[i] = (data[i] - np.mean(data[i], axis = 0))#/np.std(data[i],axis = 0)
'''
data_loader = data_iterable(data, batch_size)

batches_per_epoch = 1000//batch_size
data_gen = inf_generator(data_loader)
#batches_per_epoch = len(data_loader)
samp_ts = Variable(torch.from_numpy(np.arange(0, 10, 0.1)).cuda())#np.arange(-1,100,0.1)).cuda()
# ts_size = 10//0.01
ts_size = 100

#print('bpe: ', batches_per_epoch)

#rec = RecognitionRNN(latent_dim=latent_size, obs_dim=1, nhidden=25, nbatch=batch_size).cuda()
#rec = nn.GRU(1,2).cuda()
# func = [ODEBlock(ODEfunc(latent_size,100), samp_ts)]#.cuda()
# linear =
# func = [nn.Linear(rand_size,1), ODEBlock(ODEfunc(latent_size), samp_ts)]#.cuda()
func = [ODEBlock(ODEfunc(latent_size), samp_ts)]#.cuda()
dec = [Decoder(latent_size,1,20)]#.cuda() # 200 ?
ode_index = 0
# func = ODEfunc(latent_size,100).cuda()
# dec = Decoder(latent_size,1,200).cuda() # 200 ?
G = nn.Sequential(*func,*dec).cuda()

# D = Discriminator(input_size=1, hidden_size=5, nbatch = batch_size, num_layers=2).cuda()
D = Discriminator(input_size=1, hidden_size=10, nbatch = batch_size, num_layers=1).cuda()


# params = (list(func.parameters()) + list(dec.parameters()))

d_optimizer = Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))


g_optimizer = Adam(G.parameters(), lr=5e-3, betas=(0.5, 0.999))



f_nfe_meter = RunningAverageMeter()
b_nfe_meter = RunningAverageMeter()

replay = ExpReplay()

for itr in range(num_epochs * batches_per_epoch):


    n_batch = itr % batches_per_epoch
    epoch = itr // batches_per_epoch
    #real_batch,numb = data_gen.__next__()
    #print(numb)
    real_batch = data_gen.__next__()





    #if itr % 4 == 0:
    # 1. Train Discriminator

    #print('rb',real_batch.size())
    #print(real_batch[1])
    real_data = Variable(torch.tensor(real_batch).cuda())
    noise_ = noise(batch_size)

    #print('real_data: ', real_data.size())
    '''h = rec.initHidden().cuda()#.to(device)

    #for t in reversed(range(samp_trajs.size(1))):
    for t in reversed(range(noise_.size(2))):
        #(batch_size, ts_size, 1)
        #obs = noise_[:, t, :]
        obs = noise_[:,:,t]
        #obs = noise_[:, t]
        z0, h = rec.forward(obs, h)
        #out= rec.forward(obs, )
   '''
    #qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
    #epsilon = torch.randn(qz0_mean.size()).to(device)
    #epsilon between zero and 1
    #z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    #epsilon*exp(1/2log(sigma2))+mean = epsilon*sigma + mean

    #z0 = out
    z0 = Variable(torch.randn(batch_size,latent_size, rand_size)).cuda()
    print('z0: ',z0.size())
    # forward in time and solve ode for reconstructions

    # pred_z = odeint(func, z0, samp_ts).permute(1,0,2)#.permute(1, 2, 0)

    fake_data = G(z0)
    replay.update(fake_data)

    nfe_forward = G[ode_index].nfe
    G[ode_index].nfe = 0


    f_nfe_meter.update(nfe_forward)
    print('forward: ', f_nfe_meter.avg)
    #pred_x = dec(pred_z)
    #print('pred_z: ', pred_z.size())
    ##################""" CHANGE THE PERMUTE - IT'S DIRTY"""

    # fake_data = dec(pred_z)#.detach()#.permute(0,2,1)

    print('fake_size: ',fake_data.size())
    #print(fake_data.cpu().detach())


    #print(real_data.size(0))
    if torch.cuda.is_available(): real_data = real_data.cuda()
    # Generate fake data

    d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer,
                                                            real_data, torch.cat([fake_data,replay.get_sample(batch_size)],0))

    # 2. Train Generator
    # Generate fake data

    #z_ = noise(real_data.size(0))


    z0 = Variable(torch.randn(batch_size, latent_size,rand_size)).cuda()
    #print('z0: ',z0.size())
    # forward in time and solve ode for reconstructions
    fake_data = G(z0)
    # pred_z = odeint(func, z0, samp_ts).permute(1,0,2)#.permute(1, 2, 0)

    #pred_x = dec(pred_z)
    #print('pred_z: ', pred_z.size())
    ##################""" CHANGE THE PERMUTE - IT'S DIRTY"""

    # fake_data = dec(pred_z)#.detach()


    # Train G
    #for param in generator.parameters():
    #    print(param.grad)
    g_error = train_generator(g_optimizer, fake_data)
    # Log error
    #logger.log(d_error, g_error, epoch, n_batch, num_batches)
    # print('backward: ',b_nfe_meter.avg)

    # nfe_backward = G[0].nfe
    G[ode_index].nfe = 0

    # batch_time_meter.update(time.time() - end)

    f_nfe_meter.update(nfe_forward)
    # b_nfe_meter.update(nfe_backward)
    # end = time.time()

    # Display Progress

    if (n_batch) % 10 == 0:
        print(n_batch)
        print('replay size: ', replay.size)
        print('mean z0: ',z0[0].mean().cpu())
        print('d true avg decision: ',d_pred_real.mean().cpu().detach().numpy())
        print('d fake avg decision: ',d_pred_fake.mean().cpu().detach().numpy())
        print('discriminator error: ',d_error.cpu().detach().numpy())
        print('generator error: ',g_error.cpu().detach().numpy())
        display.clear_output(True)
        # pred_fixed = odeint(func, z_fixed, samp_ts).permute(1,0,2)
        # fake_fixed = dec(pred_fixed)#.detach()#.permute(0,2,1)

        fake_fixed = G(z_fixed).detach()
        # print("ZOB PROUT")
        # fake_fixed = fake_data
        fig, ax = plt.subplots(nrows=4, ncols=2)

        for i in range(2):
            for j in range(2):
                ax[i, j].plot(samp_ts.cpu().numpy(), fake_fixed[(i)*(2)+j].cpu().detach().numpy(), 'r')
                ax[i+2, j].plot(samp_ts.cpu().numpy(), real_data[(i)*(2)+j].cpu().detach().numpy(), 'g')
                #col.plot(samp_ts.cpu().detach().numpy(), fake_data[(i+1)*(j+1)-1].cpu().numpy())
        #plt.figure()
        #plt.plot(samp_ts.cpu().detach().numpy(), fake_data[0].cpu().numpy())
        #plt.show()
        plt.savefig('fig/fig'+str(epoch)+'_'+str(n_batch))
        plt.close()
        # Display Images

        ## TODO Changer le concat ici pour qqchose de fixe
        #test_images = generator(test_concat).data.cpu()
        #test_images = generator(test_noise,test_y_fill_).data.cpu()
        #test_images = generator(test_noise,test_y_label_).data.cpu()
        #print(test_y_.cpu().detach())
        #logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
        # Display status Logs
        #logger.display_status(
        #    epoch, num_epochs, n_batch, num_batches,
        #    d_error, g_error, d_pred_real, d_pred_fake
        #)
    # Model Checkpoints
    #logger.save_models(generator, discriminator, epoch)
