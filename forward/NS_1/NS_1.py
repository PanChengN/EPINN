'''
@Project ：Enhancing PINNs for solving forward and inverse Navier-Stokes equations via adaptive loss weighting and neural architecture innovation 
@File    ：NS_1.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/8/6 上午9:58 
'''
from DNN import Net, Net_attention
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import trange

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

seed_torch(1234)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

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

class VPNSFnet:
    def __init__(self, xb, yb, ub, vb, pb, x, y, layers, model_type):
        Xb = np.concatenate([xb, yb], 1)
        X = np.concatenate([x, y], 1)

        self.lowb = Xb.min(0)
        self.upb = Xb.max(0)

        self.Xb = Xb
        self.X = X

        # Convert the first and second columns of Xb, respectively, into tensors for gradient computation
        self.xb = torch.tensor(Xb[:, 0:1], requires_grad=True).float().to(device)
        self.yb = torch.tensor(Xb[:, 1:2], requires_grad=True).float().to(device)
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        # Convert boundary conditions ub, vb into tensors that require gradient calculations
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
            self.sigma_r = torch.nn.Parameter(self.sigma_r)
            self.sigma_b = torch.nn.Parameter(self.sigma_b)
            self.optimizer_lam = torch.optim.Adam([self.sigma_r] + [self.sigma_b], lr=1e-3)
        elif self.model_type in ['EPINN-L']:
            self.dnn = Net_attention(layers).to(device)
            self.sigma_b = torch.tensor([1.], requires_grad=True).float().to(device)
            self.sigma_r = torch.tensor([1.], requires_grad=True).float().to(device)
            self.sigma_b = torch.nn.Parameter(self.sigma_b)
            self.sigma_r = torch.nn.Parameter(self.sigma_r)
            self.optimizer_lam = torch.optim.Adam([self.sigma_r] + [self.sigma_b], lr=1e-3)

        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(), lr=1e-3)


        self.loss_history = []
        self.w_history = []

        self.iter=0

    def d(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

    def normalize(self, X):
        lowb = torch.tensor(self.lowb, dtype=torch.float32).to(device)
        upb = torch.tensor(self.upb, dtype=torch.float32).to(device)
        return 2.0 * (X - lowb) / (upb - lowb) - 1.0

    def net_NS(self, x, y):
        X_norm = self.normalize(torch.cat([x, y], dim=1))
        u_v_p = self.dnn(X_norm)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]
        return u, v, p

    def net_f_NS(self, x, y):
        X_norm = self.normalize(torch.cat([x, y], dim=1))
        u_v_p = self.dnn(X_norm)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        u_x = self.d(u, x)
        u_y = self.d(u, y)
        u_xx = self.d(u_x, x)
        u_yy = self.d(u_y, y)

        v_x = self.d(v, x)
        v_y = self.d(v, y)
        v_xx = self.d(v_x, x)
        v_yy = self.d(v_y, y)

        p_x = self.d(p, x)
        p_y = self.d(p, y)

        f_u = (u * u_x + v * u_y) + p_x - (1.0 / 40) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y - (1.0 / 40) * (v_xx + v_yy)
        f_e = u_x + v_y
        return u, v, p, f_u, f_v, f_e

    def Adam_train(self, nIter=5000):
        print(f"model: {self.model_type}, layer: {self.layers}")
        self.dnn.train()
        pbar = trange(nIter, ncols=180)
        for it in pbar:
            u_b_pred, v_b_pred, p_b_pred = self.net_NS(self.xb, self.yb)
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred, f_e_pred = self.net_f_NS(self.x, self.y)

            loss_b = torch.mean(torch.square(self.ub - u_b_pred)) + torch.mean(
                torch.square(self.vb - v_b_pred)) + torch.mean(torch.square(self.pb - p_b_pred))
            loss_r = torch.mean(torch.square(f_u_pred)) + torch.mean(torch.square(f_v_pred)) + torch.mean(
                torch.square(f_e_pred))

            self.optimizer_Adam.zero_grad()
            if self.model_type in ['PINN']:
                loss = loss_b + loss_r
                loss.backward()
                self.w_history.append([1.,1.])
            elif self.model_type in ['EPINN-E']:
                self.optimizer_lam.zero_grad()
                loss = torch.exp(-self.sigma_r) * loss_r + torch.exp(
                    -self.sigma_b) * loss_b + self.sigma_r + self.sigma_b
                loss.backward()
                self.optimizer_lam.step()
                self.w_history.append([torch.exp(-self.sigma_b).item(), torch.exp(-self.sigma_r).item()])
            elif self.model_type in ['EPINN-L']:
                self.optimizer_lam.zero_grad()
                loss = 1. / (self.sigma_b ** 2) * loss_b + 1. / (self.sigma_r ** 2) * loss_r + torch.log(
                    self.sigma_b ** 2) + torch.log(self.sigma_r ** 2)
                loss.backward()
                self.optimizer_lam.step()
                self.w_history.append([1. / (self.sigma_b ** 2).item(), 1. / (self.sigma_r ** 2).item()])
            self.optimizer_Adam.step()
            true_loss = loss_r+loss_b
            self.loss_history.append([loss_b.item(), loss_r.item()])

            if it % 100 == 0:
                if self.model_type in ['PINN']:
                    pbar.set_postfix({
                        'Loss': '{0:.3e}'.format(true_loss.item()),
                        'Loss_b': '{0:.3e}'.format(loss_b.item()),
                        'Loss_r': '{0:.3e}'.format(loss_r.item()),
                        'w_b': '{0:.2f}'.format(1.),
                        'w_r': '{0:.2f}'.format(1.)
                    })
                elif self.model_type in ['EPINN-E']:
                    pbar.set_postfix({
                        'Loss': '{0:.3e}'.format(true_loss.item()),
                        'Loss_b': '{0:.3e}'.format(loss_b.item()),
                        'Loss_r': '{0:.3e}'.format(loss_r.item()),
                        'w_b': '{0:.2f}'.format(torch.exp(-self.sigma_b).item()),
                        'w_r': '{0:.2f}'.format(torch.exp(-self.sigma_r).item())
                    })
                elif self.model_type in ['EPINN-L']:
                    pbar.set_postfix({
                        'Loss': '{0:.3e}'.format(true_loss.item()),
                        'Loss_b': '{0:.3e}'.format(loss_b.item()),
                        'Loss_r': '{0:.3e}'.format(loss_r.item()),
                        'w_b': '{0:.2f}'.format(1. / (self.sigma_b ** 2).item()),
                        'w_r': '{0:.2f}'.format(1. / (self.sigma_r ** 2).item())
                    })
            self.iter+=1
        print("Trainning done!")

    def predict(self, x_star, y_star):
        self.dnn.eval()
        x = torch.tensor(x_star, requires_grad=True).float().to(device)
        y = torch.tensor(y_star, requires_grad=True).float().to(device)
        with torch.no_grad():
            u, v, p = self.net_NS(x, y)
        return u, v, p


if __name__ == "__main__":
    model_list = ['PINN', 'EPINN-E', 'EPINN-L']
    for model_type in model_list:
        train_epoch=10000
        depth=7
        widths=20
        layers = [2] + [widths] * depth + [3]
        layer_str = f'{depth}x{widths}'
        N_train = 2601

        Re = 40
        lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

        x = np.linspace(-0.5, 1.0, 101)
        y = np.linspace(-0.5, 1.5, 101)
        yb1 = np.array([-0.5] * 100)
        yb2 = np.array([1] * 100)
        xb1 = np.array([-0.5] * 100)
        xb2 = np.array([1.5] * 100)
        y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
        x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)
        xb_train = x_train1.reshape(x_train1.shape[0], 1)
        yb_train = y_train1.reshape(y_train1.shape[0], 1)

        ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
        vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)
        pb_train = 0.5 * (1 - np.exp(2 * lam * xb_train))

        x_train = (np.random.rand(N_train, 1) - 1 / 3) * 3 / 2
        y_train = (np.random.rand(N_train, 1) - 1 / 4) * 2

        model = VPNSFnet(xb_train, yb_train, ub_train, vb_train, pb_train,
                         x_train, y_train, layers, model_type)

        model.Adam_train(train_epoch)

        np.random.seed(1234)
        num_test = 30000

        # Prepare predictive data. It is also used to find exact solutions
        x_star = (np.random.rand(num_test, 1) - 1 / 3) * 3 / 2
        y_star = (np.random.rand(num_test, 1) - 1 / 4) * 2

        # The exact value of the predicted dataset for uvp
        u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
        v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
        p_star = 0.5 * (1 - np.exp(2 * lam * x_star))

        # Model prediction/evaluation
        u_pred, v_pred, p_pred = model.predict(x_star, y_star)
        u_pred = u_pred.detach().cpu().numpy()
        v_pred = v_pred.detach().cpu().numpy()
        p_pred = p_pred.detach().cpu().numpy()

        error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
        error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
        error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
        print('Error u: %e' % error_u)
        print('Error v: %e' % error_v)
        print('Error p: %e' % error_p)

        # Calculate the absolute error
        error_u_abs = np.abs(u_star - u_pred)
        error_v_abs = np.abs(v_star - v_pred)
        error_p_abs = np.abs(p_star - p_pred)

        # Heatmap of exact solution, predicted solution, error
        plot_heatmap(x_star.flatten(), y_star.flatten(), u_star.flatten(), 'Exact u', f'./pic/{model_type}_{layer_str}_exact_u.pdf')
        plot_heatmap(x_star.flatten(), y_star.flatten(), u_pred.flatten(), 'Predicted u',
                     f'./pic/{model_type}_{layer_str}_predicted_u.pdf')
        plot_heatmap(x_star.flatten(), y_star.flatten(), error_u_abs.flatten(), 'Absolute Error u',
                     f'./pic/{model_type}_{layer_str}_error_u.pdf')

        plot_heatmap(x_star.flatten(), y_star.flatten(), v_star.flatten(), 'Exact v', f'./pic/{model_type}_{layer_str}_exact_v.pdf')
        plot_heatmap(x_star.flatten(), y_star.flatten(), v_pred.flatten(), 'Predicted v',
                     f'./pic/{model_type}_{layer_str}_predicted_v.pdf')
        plot_heatmap(x_star.flatten(), y_star.flatten(), error_v_abs.flatten(), 'Absolute Error v',
                     f'./pic/{model_type}_{layer_str}_error_v.pdf')

        plot_heatmap(x_star.flatten(), y_star.flatten(), p_star.flatten(), 'Exact p', f'./pic/{model_type}_{layer_str}_exact_p.pdf')
        plot_heatmap(x_star.flatten(), y_star.flatten(), p_pred.flatten(), 'Predicted p',
                     f'./pic/{model_type}_{layer_str}_predicted_p.pdf')
        plot_heatmap(x_star.flatten(), y_star.flatten(), error_p_abs.flatten(), 'Absolute Error p',
                     f'./pic/{model_type}_{layer_str}_error_p.pdf')

        # Plot loss image
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_history, label=['$\mathcal{L}_b$', '$\mathcal{L}_r$'])
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./pic/{model_type}_{layer_str}_loss_history.pdf', dpi=300)
        # plt.show()

        # plot the weight change image
        plt.figure(figsize=(10, 6))
        plt.plot(model.w_history, label=['$\lambda_b$', '$\lambda_r$'])
        plt.xlabel('Epoch')
        plt.ylabel('Weight values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./pic/{model_type}_{layer_str}_w_history.pdf', dpi=300)
        # plt.show()
