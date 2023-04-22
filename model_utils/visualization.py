import config
from matplotlib import pyplot as plt
import numpy as np
import random


class Visualization():
    @staticmethod
    def tr_plot(tr_data, start_epoch, ):
        if type(tr_data) == dict:
            history = tr_data
        else:
            history = tr_data.history
        # Plot the training and validation data
        tacc = history['r_square']
        tloss = history['loss']
        vacc = history['val_r_square']
        vloss = history['val_loss']
        Epoch_count = len(tacc) + start_epoch
        Epochs = []
        for i in range (start_epoch, Epoch_count):
            Epochs.append(i + 1)   
        index_loss = np.argmin(vloss) 
        # This is the epoch with the lowest validation loss
        val_lowest = vloss[index_loss]
        index_acc = np.argmax(vacc)
        acc_highest = vacc[index_acc]
        plt.style.use('fivethirtyeight')
        sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
        vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)
        fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(20,8))
        axes[0].plot(Epochs, tloss, 'r', label='Training loss (RMSE)')
        axes[0].plot(Epochs, vloss,'g',label='Validation loss (RMSE)')
        axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_ylim(bottom=min(min(tloss), min(vloss)) - 0.025, top=max(max(tloss), max(vloss)) + 0.025)
        axes[0].legend()
        axes[1].plot(Epochs, tacc,'r',label='Training Accuracy (R^2)')
        axes[1].plot(Epochs, vacc,'g',label='Validation Accuracy (R^2)')
        axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c= 'blue', label=vc_label)
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(bottom=max(min(min(tacc), min(vacc)), -1) - 0.025, top=max(max(tacc), max(vacc)) + 0.025)
        axes[1].legend()
        plt.show()

    @staticmethod
    def plot_pred_true(y_pred, y_test, select_dim, linewidths=[0.2, 3], compare_idx=None):

        def smooth(y, mean_y, buffer=100):
            z, N = [], buffer
            for i in range(len(y)):
                if i <= N//2:
                    z.append(sum(y[:(2*i + 1)]/(2*i + 1)))
                else:
                    z.append(sum(y[i - N//2:i + 1 + N//2])/(N + 1))
            y = [item - mean_y for item in y]
            z = [item - mean_y for item in z]
            return y, z

        for (i, dim) in enumerate(select_dim):
            f, ax= plt.subplots(figsize=(20, 10))
            x = [i for i in range(len(y_pred[:, i]))]
            y = y_pred[:, i]
            t = y_test[:, i]

            mean_y = np.mean(y)
            y, z1 = smooth(y, mean_y=mean_y)
            t, z2 = smooth(t, mean_y=mean_y)
            ax.plot(x, y, color="orange", linewidth=linewidths[0], alpha=0.7, label="Prediction")
            ax.plot(x, t, color="#1f77b4", linewidth=linewidths[0], alpha=0.7, label="Ground Truth")
            ax.plot(x, z1, color="red", linewidth=linewidths[1], alpha=0.7, label="Prediction (Smoothed)")
            ax.plot(x, z2, color="green", linewidth=linewidths[1], alpha=0.7, label="Ground Truth (Smoothed)")
            if compare_idx:
                plt.axvline(x=compare_idx, linestyle="--", linewidth=linewidths[1], color='black', label='Multi-step Start')
            ax.legend()
            ax.grid(True)

            ax.fill_between(x, y, alpha=0.25, color="lightpink")
            ax.fill_between(x, t, alpha=0.25, color="lightblue")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value of Dimension -> " + str(dim))
            plt.show()

    @staticmethod
    def plot_pred_true_partraw(y_pred, y_test, select_dim, ratio=0.05):

        for (i, dim) in enumerate(select_dim):
            f, ax= plt.subplots(figsize=(20, 10))
            x = [i for i in range(len(y_pred[:, i]))]
            select_len = int(ratio * len(y_pred[:, i]))
            start_idx = int(random.random() * (len(y_pred[:, i]) - select_len))
            x = x[start_idx:start_idx + select_len]
            y = y_pred[start_idx:start_idx + select_len, i]
            t = y_test[start_idx:start_idx + select_len, i]
            mean_y = np.mean(y)
            y = [item - mean_y for item in y]
            t = [item - mean_y for item in t]

            ax.plot(x, y, color="orange", linewidth=1, label="Prediction")
            ax.plot(x, t, color="#1f77b4", linewidth=1, label="Ground Truth")
            ax.legend()
            ax.grid(True)

            ax.fill_between(x, y, alpha=0.5, color="lightpink")
            ax.fill_between(x, t, alpha=0.5, color="lightblue")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value of Dimension -> " + str(dim))
            plt.show()

