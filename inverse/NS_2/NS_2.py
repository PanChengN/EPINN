'''
@Project ：Enhancing PINNs for solving forward and inverse Navier-Stokes equations via adaptive loss weighting and neural architecture innovation 
@File    ：NS_4.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/8/8 上午11:04 
'''

from DNN import *
import torch
import numpy as np
import warnings
import random
from tqdm import trange

import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1234)
# seed_torch(123456)

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


class Cylinder_wake_PINN:
    # Initialize the class
    def __init__(self, x0, y0, t0, u0, v0, p0, xb, yb, tb, ub, vb, pb, x, y, t, layers, model_type):
        X0 = np.concatenate([x0, y0, t0], 1) # initial value
        Xb = np.concatenate([xb, yb, tb], 1)  # boundary values
        X = np.concatenate([x, y, t], 1)  # Internal values

        self.lowb = torch.tensor(Xb.min(0)).float().to(device)
        self.upb = torch.tensor(Xb.max(0)).float().to(device)

        self.X0 = torch.tensor(X0, requires_grad=True).float().to(device)
        self.Xb = torch.tensor(Xb, requires_grad=True).float().to(device)
        self.X = torch.tensor(X, requires_grad=True).float().to(device)

        self.x0 = torch.tensor(X0[:, 0:1], requires_grad=True).float().to(device)
        self.y0 = torch.tensor(X0[:, 1:2], requires_grad=True).float().to(device)
        self.t0 = torch.tensor(X0[:, 2:3], requires_grad=True).float().to(device)

        self.xb = torch.tensor(Xb[:, 0:1], requires_grad=True).float().to(device)
        self.yb = torch.tensor(Xb[:, 1:2], requires_grad=True).float().to(device)
        self.tb = torch.tensor(Xb[:, 2:3], requires_grad=True).float().to(device)

        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)

        self.u0 = torch.tensor(u0, requires_grad=True).float().to(device)
        self.v0 = torch.tensor(v0, requires_grad=True).float().to(device)
        self.p0 = torch.tensor(p0, requires_grad=True).float().to(device)

        self.ub = torch.tensor(ub, requires_grad=True).float().to(device)
        self.vb = torch.tensor(vb, requires_grad=True).float().to(device)
        self.pb = torch.tensor(pb, requires_grad=True).float().to(device)

        self.layers = layers
        self.model_type = model_type

        if self.model_type in ['PINN']:
            self.dnn = Net(layers).to(device)
        elif self.model_type in ['EPINN-E']:
            self.dnn = Net_attention(layers).to(device)
            self.sigma_r = torch.tensor([0.], requires_grad=True).float().to(device)
            self.sigma_b = torch.tensor([0.], requires_grad=True).float().to(device)
            self.sigma_0 = torch.tensor([0.], requires_grad=True).float().to(device)
            self.sigma_r = torch.nn.Parameter(self.sigma_r)
            self.sigma_b = torch.nn.Parameter(self.sigma_b)
            self.sigma_0 = torch.nn.Parameter(self.sigma_0)
            self.optimizer_lam = torch.optim.Adam([self.sigma_r] + [self.sigma_b]+ [self.sigma_0], lr=1e-3)
        elif self.model_type in ['EPINN-L']:
            self.dnn = Net_attention(layers).to(device)
            self.sigma_0 = torch.tensor([1.], requires_grad=True).float().to(device)
            self.sigma_b = torch.tensor([1.], requires_grad=True).float().to(device)
            self.sigma_r = torch.tensor([1.], requires_grad=True).float().to(device)
            self.sigma_0 = torch.nn.Parameter(self.sigma_0)
            self.sigma_b = torch.nn.Parameter(self.sigma_b)
            self.sigma_r = torch.nn.Parameter(self.sigma_r)
            self.optimizer_lam = torch.optim.Adam([self.sigma_r] + [self.sigma_b]+ [self.sigma_0], lr=1e-3)
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        self.optimizer_Adam = torch.optim.Adam(list(self.dnn.parameters()) + [self.lambda_1] + [self.lambda_2], lr=1e-3)

        self.iter = 0
        self.loss_history = []
        self.lambda_history = []
        self.w_history = []

    def d(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def normalize(self, X):
        lowb = torch.tensor(self.lowb, dtype=torch.float32).to(device)
        upb = torch.tensor(self.upb, dtype=torch.float32).to(device)
        return 2.0 * (X - lowb) / (upb - lowb) - 1.0

    def net_NS(self, x, y, t):
        X_norm = self.normalize(torch.cat([x, y, t], dim=1))
        u_v_p = self.dnn(X_norm)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        return u, v, p

    def net_f_NS(self, x, y, t):

        u, v, p = self.net_NS(x, y, t)

        u_t = self.d(u, t)
        u_x = self.d(u, x)
        u_y = self.d(u, y)
        u_xx = self.d(u_x, x)
        u_yy = self.d(u_y, y)

        v_t = self.d(v, t)
        v_x = self.d(v, x)
        v_y = self.d(v, y)
        v_xx = self.d(v_x, x)
        v_yy = self.d(v_y, y)

        p_x = self.d(p, x)
        p_y = self.d(p, y)

        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)
        f_e = u_x + v_y

        return u, v, p, f_u, f_v, f_e

    def epoch_train(self):
        u_ini_pred, v_ini_pred, p_ini_pred = self.net_NS(self.x0, self.y0, self.t0)
        u_boundary_pred, v_boundary_pred, p_boundary_pred = self.net_NS(self.xb, self.yb, self.tb)
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred, f_e_pred = self.net_f_NS(self.x, self.y, self.t)

        loss_i = torch.mean(torch.square(self.u0 - u_ini_pred)) + torch.mean(
            torch.square(self.v0 - v_ini_pred)) + torch.mean(torch.square(self.p0 - p_ini_pred))
        loss_b = torch.mean(torch.square(self.ub - u_boundary_pred)) + torch.mean(
            torch.square(self.vb - v_boundary_pred)) + torch.mean(torch.square(self.pb - p_boundary_pred))
        loss_r = torch.mean(torch.square(f_u_pred) + torch.square(f_v_pred) + torch.square(f_e_pred))

        return loss_i, loss_b, loss_r


    def train(self, nIter):
        print(f"model: {self.model_type}, layer: {self.layers}")
        self.dnn.train()
        pbar = trange(nIter, ncols=180)
        for it in pbar:
            u_0_pred, v_0_pred, p_0_pred = self.net_NS(self.x0, self.y0, self.t0)
            u_b_pred, v_b_pred, p_b_pred = self.net_NS(self.xb, self.yb, self.tb)
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred, f_e_pred = self.net_f_NS(self.x, self.y, self.t)

            loss_0 = torch.mean(torch.square(self.u0 - u_0_pred)) + torch.mean(
                torch.square(self.v0 - v_0_pred)) + torch.mean(torch.square(self.p0 - p_0_pred))
            loss_b = torch.mean(torch.square(self.ub - u_b_pred)) + torch.mean(
                torch.square(self.vb - v_b_pred)) + torch.mean(torch.square(self.pb - p_b_pred))
            loss_r = torch.mean(torch.square(f_u_pred)) + torch.mean(torch.square(f_v_pred)) + torch.mean(
                torch.square(f_e_pred))

            self.optimizer_Adam.zero_grad()
            if self.model_type in ['PINN']:
                loss = loss_0 + loss_b + loss_r
                loss.backward()
                self.w_history.append([1., 1., 1.])
            elif self.model_type in ['EPINN-E']:
                self.optimizer_lam.zero_grad()
                loss = torch.exp(-self.sigma_0) * loss_0 + torch.exp(-self.sigma_b) * loss_b + torch.exp(
                    -self.sigma_r) * loss_r + self.sigma_0 + self.sigma_b + self.sigma_r
                loss.backward()
                self.optimizer_lam.step()
                self.w_history.append(
                    [torch.exp(-self.sigma_0).item(), torch.exp(-self.sigma_b).item(), torch.exp(-self.sigma_r).item()])
            elif self.model_type in ['EPINN-L']:
                self.optimizer_lam.zero_grad()
                loss = 1. / (self.sigma_0 ** 2) * loss_0 + 1. / (self.sigma_b ** 2) * loss_b + 1. / (
                            self.sigma_r ** 2) * loss_r + torch.log(
                    self.sigma_0 ** 2) + torch.log(self.sigma_b ** 2) + torch.log(self.sigma_r ** 2)
                loss.backward()
                self.optimizer_lam.step()
                self.w_history.append(
                    [1. / (self.sigma_0 ** 2).item(), 1. / (self.sigma_b ** 2).item(), 1. / (self.sigma_r ** 2).item()])

            self.optimizer_Adam.step()
            true_loss = loss_r + loss_b + loss_0
            self.loss_history.append([loss_0.item(), loss_b.item(), loss_r.item()])
            self.lambda_history.append([self.lambda_1.item(), self.lambda_2.item()])

            if it % 100 == 0:
                if self.model_type in ['PINN']:
                    pbar.set_postfix({
                        'Loss': '{0:.3e}'.format(true_loss.item()),
                        'Loss_0': '{0:.3e}'.format(loss_0.item()),
                        'Loss_b': '{0:.3e}'.format(loss_b.item()),
                        'Loss_r': '{0:.3e}'.format(loss_r.item()),
                        'w_0': '{0:.2f}'.format(1.),
                        'w_b': '{0:.2f}'.format(1.),
                        'w_r': '{0:.2f}'.format(1.),
                        'lam_1': '{0:.5f}'.format(self.lambda_1.item()),
                        'lam_2': '{0:.5f}'.format(self.lambda_2.item())
                    })
                elif self.model_type in ['EPINN-E']:
                    pbar.set_postfix({
                        'Loss': '{0:.3e}'.format(true_loss.item()),
                        'Loss_0': '{0:.3e}'.format(loss_0.item()),
                        'Loss_b': '{0:.3e}'.format(loss_b.item()),
                        'Loss_r': '{0:.3e}'.format(loss_r.item()),
                        'w_0': '{0:.2f}'.format(torch.exp(-self.sigma_0).item()),
                        'w_b': '{0:.2f}'.format(torch.exp(-self.sigma_b).item()),
                        'w_r': '{0:.2f}'.format(torch.exp(-self.sigma_r).item()),
                        'lam_1': '{0:.5f}'.format(self.lambda_1.item()),
                        'lam_2': '{0:.5f}'.format(self.lambda_2.item())
                    })
                elif self.model_type in ['EPINN-L']:
                    pbar.set_postfix({
                        'Loss': '{0:.3e}'.format(true_loss.item()),
                        'Loss_0': '{0:.3e}'.format(loss_0.item()),
                        'Loss_b': '{0:.3e}'.format(loss_b.item()),
                        'Loss_r': '{0:.3e}'.format(loss_r.item()),
                        'w_0': '{0:.2f}'.format(1. / (self.sigma_0 ** 2).item()),
                        'w_b': '{0:.2f}'.format(1. / (self.sigma_b ** 2).item()),
                        'w_r': '{0:.2f}'.format(1. / (self.sigma_r ** 2).item()),
                        'lam_1': '{0:.5f}'.format(self.lambda_1.item()),
                        'lam_2': '{0:.5f}'.format(self.lambda_2.item())
                    })

    def predict(self, x_star, y_star, t_star):
        self.dnn.eval()
        x = torch.tensor(x_star, requires_grad=True).float().to(device)
        y = torch.tensor(y_star, requires_grad=True).float().to(device)
        t = torch.tensor(t_star, requires_grad=True).float().to(device)
        with torch.no_grad():
            u, v, p = self.net_NS(x,y,t)
        return u, v, p

def plot_heatmap(x, y, values, title, save_path):
    """
    Create a heat map
    """
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    zi = griddata((x, y), values, (xi[None, :], yi[:, None]), method='cubic')

    plt.figure(figsize=(8, 6))
    plt.contourf(xi, yi, zi, levels=100, cmap='jet')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    data = scipy.io.loadmat('./Data/cylinder_nektar_wake.mat')
    U_star = data['U_star']  # N x 2 x T
    # velocity nx2xt
    P_star = data['p_star']  # N x T
    # pressure nxt
    t_star = data['t']  # T x 1
    # time t
    X_star = data['X_star']  # N x 2
    # location nx2
    # n number of datapoint, t time steps
    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    data1 = np.concatenate([x, y, t, u, v, p], 1)

    # data range
    data_domain = data1[:, :][data1[:, 2] <= 7]

    # data at initial time
    data_t0 = data_domain[:, :][data_domain[:, 2] == 0]

    # Boundary conditions in the y direction
    data_y1 = data_domain[:, :][data_domain[:, 0] == 1]
    data_y8 = data_domain[:, :][data_domain[:, 0] == 8]

    # # Boundary conditions in the x direction
    data_x = data_domain[:, :][data_domain[:, 1] == -2]
    data_x2 = data_domain[:, :][data_domain[:, 1] == 2]

    data_sup_b_train = np.concatenate([data_y1, data_y8, data_x, data_x2], 0)

    # Number of Residual points
    N_train=10000
    idx = np.random.choice(data_domain.shape[0], N_train, replace=False)

    # extract the training data for the Residual
    x_train = data_domain[idx, 0].reshape(-1, 1)
    y_train = data_domain[idx, 1].reshape(-1, 1)
    t_train = data_domain[idx, 2].reshape(-1, 1)

    # Training data for initial conditions
    x0_train = data_t0[:, 0].reshape(-1, 1)
    y0_train = data_t0[:, 1].reshape(-1, 1)
    t0_train = data_t0[:, 2].reshape(-1, 1)
    u0_train = data_t0[:, 3].reshape(-1, 1)
    v0_train = data_t0[:, 4].reshape(-1, 1)
    p0_train = data_t0[:, 5].reshape(-1, 1)

    # Training data for boundary conditions
    xb_train = data_sup_b_train[:, 0].reshape(-1, 1)
    yb_train = data_sup_b_train[:, 1].reshape(-1, 1)
    tb_train = data_sup_b_train[:, 2].reshape(-1, 1)
    ub_train = data_sup_b_train[:, 3].reshape(-1, 1)
    vb_train = data_sup_b_train[:, 4].reshape(-1, 1)
    pb_train = data_sup_b_train[:, 5].reshape(-1, 1)

    # depth = 7
    # widths = 100
    depth = 5
    widths = 50
    layers = [3] + [widths] * depth + [3]
    layer_str = f'{depth}x{widths}'
    epoch = 10000
    model_list = ['PINN', 'EPINN-E', 'EPINN-L']
    for model_type in model_list:

        model = Cylinder_wake_PINN(x0_train, y0_train, t0_train, u0_train, v0_train, p0_train,
                         xb_train, yb_train, tb_train, ub_train, vb_train, pb_train,
                         x_train, y_train, t_train, layers, model_type)

        model.train(epoch)

        # 下面都是在不同时刻的预测值
        # Test model performance at 1 second
        snap1 = np.array([10])
        x_star1 = X_star[:, 0:1]
        y_star1 = X_star[:, 1:2]
        t_star1 = TT[:, snap1]

        u_star1 = U_star[:, 0, snap1]
        v_star1 = U_star[:, 1, snap1]
        p_star1 = P_star[:, snap1]

        # Prediction
        u_pred1, v_pred1, p_pred1 = model.predict(x_star1, y_star1, t_star1)
        u_pred1 = u_pred1.detach().cpu().numpy()
        v_pred1 = v_pred1.detach().cpu().numpy()
        p_pred1 = p_pred1.detach().cpu().numpy()

        # Data format conversion
        lambda_1_value = model.lambda_1.detach().cpu().numpy()
        lambda_2_value = model.lambda_2.detach().cpu().numpy()

        # Error
        error_u1 = np.linalg.norm(u_star1 - u_pred1, 2) / np.linalg.norm(u_star1, 2)
        error_v1 = np.linalg.norm(v_star1 - v_pred1, 2) / np.linalg.norm(v_star1, 2)
        error_p1 = np.linalg.norm(p_star1 - p_pred1, 2) / np.linalg.norm(p_star1, 2)
        error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
        error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100

        print('performance at 1 second:\n')
        print("l1={}".format(lambda_1_value.item()))
        print("l2={}".format(lambda_2_value.item()))
        print('Error u1: %e' % error_u1)
        print('Error v1: %e' % error_v1)
        print('Error p1: %e' % error_p1)
        print('Error l1: %.4f%%' % (error_lambda_1))
        print('Error l2: %.4f%%' % (error_lambda_2))

        # Calculate the absolute error
        error_u1 = np.abs(u_star1 - u_pred1)
        error_v1 = np.abs(v_star1 - v_pred1)
        error_p1 = np.abs(p_star1 - p_pred1)

        # Plot loss image
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_history, label=['$\mathcal{L}_i$', '$\mathcal{L}_b$', '$\mathcal{L}_r$'])
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./pic/{model_type}_{layer_str}_loss_history.pdf')
        # plt.show()

        # plot the parameter changes
        plt.figure(figsize=(10, 6))
        plt.plot(model.lambda_history, label=['$\lambda_1$', '$\lambda_2$'])
        plt.xlabel('Epoch')
        plt.ylabel('$\lambda$ values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./pic/{model_type}_{layer_str}_lambda_history.pdf')
        # plt.show()

        # plot the weight change image
        plt.figure(figsize=(10, 6))
        plt.plot(model.w_history, label=['$\lambda_i$', '$\lambda_b$', '$\lambda_r$'])
        plt.xlabel('Epoch')
        plt.ylabel('Weight values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./pic/{model_type}_{layer_str}_w_history.pdf')
        # plt.show()

        # Heatmap of exact solution, predicted solution, error
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), u_star1.flatten(), 'Exact u', f'./pic/{model_type}_{layer_str}_exact_u.pdf')
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), u_pred1.flatten(), 'Predicted u',
                     f'./pic/{model_type}_{layer_str}_predicted_u.pdf')
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), error_u1.flatten(), 'Absolute Error u',
                     f'./pic/{model_type}_{layer_str}_error_u.pdf')

        plot_heatmap(x_star1.flatten(), y_star1.flatten(), v_star1.flatten(), 'Exact v', f'./pic/{model_type}_{layer_str}_exact_v.pdf')
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), v_pred1.flatten(), 'Predicted v',
                     f'./pic/{model_type}_{layer_str}_predicted_v.pdf')
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), error_v1.flatten(), 'Absolute Error v',
                     f'./pic/{model_type}_{layer_str}_error_v.pdf')

        plot_heatmap(x_star1.flatten(), y_star1.flatten(), p_star1.flatten(), 'Exact p', f'./pic/{model_type}_{layer_str}_exact_p.pdf')
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), p_pred1.flatten(), 'Predicted p',
                     f'./pic/{model_type}_{layer_str}_predicted_p.pdf')
        plot_heatmap(x_star1.flatten(), y_star1.flatten(), error_p1.flatten(), 'Absolute Error p',
                     f'./pic/{model_type}_{layer_str}_error_p.pdf')

