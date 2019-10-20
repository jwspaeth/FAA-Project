
import os
import time
import resource
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
from tensorflow.keras import Model

from lib.classes.dataset_classes.SubjectDataset import SubjectDataset

class BabyTrainingCallback(Callback):
    '''
    BabyTrainingCallback: Custom callback for recording data during training. Built for version-3 code base

    Types of figures to create:
        • png and eps for every figure
        • Filter response for every week
            •Necessary data:
                •Full dataset
                •Filter response for each week
        • Dual figure of coefficient vs time and predicted label vs target label
            •Necessary data:
                •Filters and bias
                •Loss
                •Target label
                •Predicted label
                •Feature names and their indices relative to filters

    Saving cycle:
        • Checkpoint the model every ? epochs, with its epoch number and loss value stored
        • Save figures at the end of training only

    Wishlist:
        •Automatically pull a zoom of this best responses? That's too much to ask I think
    '''

    def __init__(self, master_config):
        self.data_config = master_config.Core_Config.Data_Config
        self.save_config = master_config.Core_Config.Save_Config
        self.model_config = master_config.Core_Config.Model_Config

        self.child_name = master_config.child_name

        self.iteration_parameters = master_config.iteration_parameters

        self.subject = SubjectDataset(master_config=master_config)

        self.epoch_count = 0
        self.recent_epoch_logs = 0
        self.epoch_start_time = 0

    def on_train_begin(self, logs):
        print("Training begin!", flush=True)

    def on_epoch_begin(self, epoch, logs):

        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs):

        print("Log keys: {}\n".format(logs.keys()), flush=True)

        self.recent_epoch_logs = self._create_log(logs)
        if self.epoch_count % self.save_config.Output.checkpoint_trigger == 0:
            self._checkpoint()

        self.epoch_count += 1

        total_epoch_time = time.time() - self.epoch_start_time
        print("Total epoch time: {} secs".format(total_epoch_time), flush=True)

    def on_train_end(self, logs):

        print("Training finished!", flush=True)

        #print("Input: {}\n\n".format(self.recent_epoch_logs["input"]), flush=True)
        #print("Input 0: {}\n\n".format(self.recent_epoch_logs["split_input_0"]), flush=True)
        #print("Input 1: {}\n\n".format(self.recent_epoch_logs["split_input_1"]), flush=True)

        self._checkpoint()

        ### Save out both types of figures we're currently using
        print("Saving out figures...", flush=True)

        ### Save dual fig as png and eps
        self._create_dual_figs()

        ### Save filter responses as png and eps
        self._collect_filter_response_figs()

        ### Save trajectories as png and eps
        #self._collect_trajectory_figs()

        print("Done saving figures!", flush=True)

    def on_test_batch_end(self, batch, logs):

        self.recent_epoch_logs = self._create_log(logs)

        self.on_train_end(logs)

    def _checkpoint(self):
        print("Checkpointing...", flush=True)
        self._save_model()
        self._create_statistics_file()
        print("Finished checkpointing!", flush=True)

    def _save_model(self):
        print("Saving model...", flush=True)
        save_folder_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name
        save_folder_path += "/model_checkpoints/"

        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        self.model.save_weights(save_folder_path + "model_weights_epoch_{}.h5".format(self.epoch_count))
        print("Model saved!", flush=True)

    def _create_statistics_file(self):
        
        ### Create statistics representative of training
        ### This includes printing total loss and all separate regularization losses
        print("Saving statistics...", flush=True)
        save_file_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name
        save_file_path += "/statistics.txt"

        total_loss = self.recent_epoch_logs["loss"]
        convolutional_l2_loss = self.recent_epoch_logs["convolutional_l2_loss"]
        convolutional_activity_loss = self.recent_epoch_logs["convolutional_activity_loss"]
        convolutional_dot_loss = self.recent_epoch_logs["convolutional_dot_loss"]
        core_loss = self.recent_epoch_logs["core_loss"]
        #core_loss = total_loss - convolutional_l2_loss - convolutional_activity_loss - convolutional_dot_loss

        with open(save_file_path, "w") as f:
            f.write("Experiment parameters: {}\n".format(self.iteration_parameters))
            f.write("Epoch count: {}\n".format(self.epoch_count))
            f.write("Total loss: {}\n".format(round(total_loss, 5)))
            f.write("Core loss: {}\n".format(round(core_loss, 5)))
            f.write("L2 loss: {}\n".format(round(convolutional_l2_loss, 5)))
            f.write("Activity loss: {}\n".format(round(convolutional_activity_loss, 5)))
            f.write("Dot loss: {}\n".format(round(convolutional_dot_loss, 5)))

        print("Statistics saved!", flush=True)

    def _get_all_feature_xyz(self):
        feature_names = copy.deepcopy(self.data_config.feature_names)
        raw_names = [name.replace("_x", "") for name in feature_names]
        raw_names = [name.replace("_y", "") for name in raw_names]
        raw_names = [name.replace("_z", "") for name in raw_names]

        unique_names = list(set(raw_names))

        feature_xyz_dict = {}
        for unique_name in unique_names:

            xyz_dict = self._get_single_feature_xyz(unique_name, feature_names)
            if xyz_dict:
                feature_xyz_dict[unique_name] = copy.deepcopy(xyz_dict)

        return feature_xyz_dict

    def _get_single_feature_xyz(self, unique_name, feature_names):
        
        x_index = -1
        y_index = -1
        z_index = -1
        for i, feature_name in enumerate(feature_names):
            if unique_name in feature_name:
                if "_x" in feature_name:
                    x_index = i
                elif "_y" in feature_name:
                    y_index = i
                elif "_z" in feature_name:
                    z_index = i

        xyz_dict = {}
        if x_index != -1:
            xyz_dict["x"] = x_index

        if y_index != -1:
            xyz_dict["y"] = y_index

        if z_index != -1:
            xyz_dict["z"] = z_index

        if xyz_dict:
            return xyz_dict
        else:
            return


    def _collect_trajectory_figs(self):

        print("Saving trajectory figs...", flush=True)
        
        ### Get dictionary of all features and their xyz components
        feature_xyz_dict = self._get_all_feature_xyz()

        ### If it's null, return
        if not feature_xyz_dict:
            return

        ### Loop through available xyz dicts
        for feature_name, xyz_dict in feature_xyz_dict.items():

            for i in range(self.model_config.Convolution.n_filters):

                for j in range(self.subject.get_n_weeks()):

                    ### Check if xyz dict has x and z
                    if "x" in xyz_dict.keys() and "z" in xyz_dict.keys():
                        self._create_xvz_trajectory_fig(filter_index=i, week_index=j, feature_name=feature_name,
                            x_index=xyz_dict["x"], z_index=xyz_dict["z"])

                    ### Check if xyz dict has x and y
                    if "x" in xyz_dict.keys() and "z" in xyz_dict.keys():
                        self._create_xvy_trajectory_fig(filter_index=i, week_index=j, feature_name=feature_name,
                            x_index=xyz_dict["x"], y_index=xyz_dict["y"])

        print("Done saving trajectory figs!", flush=True)

    def _create_xvz_trajectory_fig(self, filter_index, week_index, x_index, z_index, feature_name):

        ### Generate features
        subject_factory = self.subject.subject_factory_wrapper(self.data_config, self.model_config)
        subject_generator = subject_factory()
        features, labels = next(subject_generator)

        ### Create figure
        fig, axs = plt.subplots(1)
        fig.set_figwidth(self.save_config.Callback.figwidth)
        fig.set_figheight(self.save_config.Callback.figheight)
        fig.tight_layout()
        fig.suptitle("Week "+str(week_index+1)+" Trajectory Response: X vs Z")

        ### Get individual points
        x_points = features[0, week_index, :, x_index]
        z_points = features[0, week_index, :, z_index]

        ### Create combined points and segments
        points = np.array([x_points, z_points]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        ### Get convolutional output from logs
        conv_output = self.recent_epoch_logs["convolutional_output"]

        ### Define colormap function
        norm = plt.Normalize(conv_output[0,week_index,:,filter_index].min(), conv_output[0,week_index,:,filter_index].max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(conv_output[0,week_index,:,filter_index])
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
        fig.colorbar(line, ax=axs)

        ### Set margins
        margin = .3
        half_x_points = x_points.max() - (x_points.min()+x_points.max())/2
        axs.set_xlim(x_points.min()-half_x_points*margin, x_points.max()+half_x_points*margin)
        half_z_points = z_points.max() - (z_points.min()+z_points.max())/2
        axs.set_ylim(z_points.min()-half_z_points*margin, z_points.max() + half_z_points*margin)

        ### Set labels
        axs.set_xlabel("Z Position")
        axs.set_ylabel("X Position")

        ### Save filter response fig as png and eps
        filter_response_folder_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name + "/trajectory-responses/"
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "{}/".format(feature_name)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "x-vs-z/".format(filter_index)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "filter-{}/".format(filter_index)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "week-{}/".format(week_index+1)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        single_response_png_path = filter_response_folder_path + "week-{}.png".format(week_index+1)
        fig.savefig(single_response_png_path, dpi=200, format="png", bbox_inches="tight")
        single_response_eps_path = filter_response_folder_path + "week-{}.eps".format(week_index+1)
        fig.savefig(single_response_eps_path, dpi=200, format="eps", bbox_inches="tight")

        plt.close(fig)

    def _create_xvy_trajectory_fig(self, filter_index, week_index, x_index, y_index, feature_name):

        ### Generate features
        subject_factory = self.subject.subject_factory_wrapper(self.data_config, self.model_config)
        subject_generator = subject_factory()
        features, labels = next(subject_generator)

        ### Create figure
        fig, axs = plt.subplots(1)
        fig.set_figwidth(self.save_config.Callback.figwidth)
        fig.set_figheight(self.save_config.Callback.figheight)
        fig.tight_layout()
        fig.suptitle("Week "+str(week_index+1)+" Trajectory Response: X vs Y")

        ### Get individual points
        x_points = features[0, week_index, :, x_index]
        y_points = features[0, week_index, :, y_index]

        ### Create combined points and segments
        points = np.array([x_points, y_points]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        ### Get convolutional output from logs
        conv_output = self.recent_epoch_logs["convolutional_output"]

        ### Define colormap function
        norm = plt.Normalize(conv_output[0,week_index,:,filter_index].min(), conv_output[0,week_index,:,filter_index].max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(conv_output[0,week_index,:,filter_index])
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
        fig.colorbar(line, ax=axs)

        ### Set margins
        margin = .3
        half_x_points = x_points.max() - (x_points.min()+x_points.max())/2
        axs.set_xlim(x_points.min()-half_x_points*margin, x_points.max()+half_x_points*margin)
        half_y_points = y_points.max() - (y_points.min()+y_points.max())/2
        axs.set_ylim(y_points.min()-half_y_points*margin, y_points.max()+half_y_points*margin)

        ### Set labels
        axs.set_xlabel("Y Position")
        axs.set_ylabel("X Position")

        ### Save filter response fig as png and eps
        filter_response_folder_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name + "/trajectory-responses/"
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "{}/".format(feature_name)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "x-vs-y/".format(filter_index)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "filter-{}/".format(filter_index)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "week-{}/".format(week_index+1)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        single_response_png_path = filter_response_folder_path + "week-{}.png".format(week_index+1)
        fig.savefig(single_response_png_path, dpi=200, format="png", bbox_inches="tight")
        single_response_eps_path = filter_response_folder_path + "week-{}.eps".format(week_index+1)
        fig.savefig(single_response_eps_path, dpi=200, format="eps", bbox_inches="tight")

        plt.close(fig)

    def _collect_filter_response_figs(self):

        print("Saving filter responses...", flush=True)
        
        for i in range(self.model_config.Convolution.n_filters):

            for j in range(self.subject.get_n_weeks()):

                self._create_filter_response_fig(filter_index=i, week_index=j)

        print("Done saving filter responses!", flush=True)

    def _create_filter_response_fig(self, filter_index, week_index):

        ### Generate features
        subject_generator = self.subject.get_generator()
        features, labels = next(subject_generator)

        ### Get convolutional output from logs
        conv_output = self.recent_epoch_logs["convolutional_output"]

        ### Create figure
        fig, axs = plt.subplots(1)
        fig.set_figwidth(self.save_config.Callback.figwidth)
        fig.set_figheight(self.save_config.Callback.figheight)
        fig.tight_layout()
        fig.suptitle("Week "+str(week_index+1)+" Filter Response")

        ### Get colormap cycle
        color_map_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ### Create scaling variables for figure
        big_max = np.max(features[0,week_index,:,0])
        if np.max(features[0,week_index,:,1]) > big_max:
            big_max = np.max(features[0,week_index,:,1])
        elif np.max(features[0,week_index,:,2]) > big_max:
            big_max = np.max(features[0,week_index,:,2])

        ### Adjust axis settings
        for i in range(features.shape[3]):
            axs.plot(features[0,week_index,:,i], color=color_map_cycle[i])

        ### Start plotting on the axes
        filter_linestyle = "--"
        filter_height = 1.5
        filter_mag = .5
        axs.plot(conv_output[0,week_index,:,filter_index]*big_max*filter_mag+big_max*filter_height, color=color_map_cycle[3])
        axs.plot(np.full(shape=conv_output[0,week_index,:,filter_index].shape, fill_value=0+big_max*filter_height),
                 linestyle=filter_linestyle,
                 color=color_map_cycle[3])
        axs.plot(np.full(shape=conv_output[0,week_index,:,filter_index].shape, fill_value=big_max*filter_mag + big_max*filter_height),
                 linestyle=filter_linestyle,
                 color=color_map_cycle[3])

        ### Set up axis limits
        axs.set_ylim(top=big_max*filter_mag+big_max*2)
        axs.set_ylabel("centimeters")
        this_yticks = axs.get_yticks()
        axs.set_yticklabels(np.around(this_yticks*100, decimals=0))
        axs.set_xlabel("time (sec)")

        ### left_x and right_x represent data bounds in seconds
        left_x = 0
        right_x = 300

        ### Create padding
        padding_distance = .013 * ((right_x/60*3000)-(left_x/60*3000))
        axs.set_xlim(left=(left_x/60*3000)-padding_distance, right=(right_x/60*3000)+padding_distance)

        ### Set the legend
        fig = self._set_legend_filter_response_fig(fig, self.data_config.feature_names, left_x=left_x, right_x=right_x)

        ###### Make correct folder
        ### Save filter response fig as png and eps
        filter_response_folder_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name + "/filter-responses/"
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "filter-{}/".format(filter_index)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        filter_response_folder_path += "week-{}/".format(week_index+1)
        if not os.path.exists(filter_response_folder_path):
            os.mkdir(filter_response_folder_path)

        single_response_png_path = filter_response_folder_path + "week-{}.png".format(week_index+1)
        fig.savefig(single_response_png_path, dpi=200, format="png", bbox_inches="tight")
        single_response_eps_path = filter_response_folder_path + "week-{}.eps".format(week_index+1)
        fig.savefig(single_response_eps_path, dpi=200, format="eps", bbox_inches="tight")

        plt.close(fig)

    def _set_legend_filter_response_fig(self, figure, feature_names, left_x=0, right_x=300):
        
        ### Get colormap cycle
        color_map_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax_list = figure.axes
        
        handles = []
        for i in range(len(ax_list)):
            handles[:] = []
            for j in range(len(feature_names)):
                line = mlines.Line2D([], [], color=color_map_cycle[j], markersize=15, label=feature_names[j])
                handles.append(line)
            ax_list[i].legend(handles=handles, loc="upper right", prop={"size":10})
            ### Set x tick locations
            minx = left_x/60*3000
            maxx = right_x/60*3000
            ax_list[i].set_xticks(np.arange(minx, maxx+1, (maxx-minx)/10))
            ### Set x tick labels
            maxx_seconds = right_x
            num_labels = np.arange(left_x, right_x+1, (right_x-left_x)/10)
            num_labels = np.around(num_labels, decimals=2)
            ax_list[i].set_xticklabels(map(str, num_labels))
        
        return figure

    def _create_dual_figs(self):

        print("Saving dual figs...", flush=True)

        ### Get feature names for figure labels
        feature_names = self.data_config.feature_names

        ### Get loss value, rounded to 5 decimal places
        loss = round(self.recent_epoch_logs["loss"], 5)

        ### Get convolutional filter and bias
        conv_filter = self.model.get_weights()[0]
        bias = self.model.get_weights()[1]

        ### Get n_filters
        n_filters = conv_filter.shape[2]

        fig_list = []

        for i in range(n_filters):

            ### Create blank figure
            fig, axes = plt.subplots(1, 2)
            fig.set_figwidth(self.save_config.Callback.figwidth)
            fig.set_figheight(self.save_config.Callback.figheight)
            fig.tight_layout()
            plt.subplots_adjust(wspace=.2)

            ### Insert text into figure
            fig.text(x=.1, y=1.2, s="Epoch: "+str(self.epoch_count), fontsize=13)
            fig.text(x=.1, y=1.05, s="Loss: "+str(np.round(loss, decimals=5)), fontsize=13)
            fig.text(x=.25, y=1.05, s="Bias: "+str(np.round(bias[0], decimals=3)), fontsize=13)

            ### Deal with axes
            if len(axes.shape) == 1:
                axes = np.expand_dims(axes, axis=0)

            ### Generate features
            subject_generator = self.subject.get_generator()
            features, labels = next(subject_generator)

            ### Get predicted labels
            predicted_labels = self.recent_epoch_logs["predicted_labels"]

            # Loop for every filter
            axes = self._modify_axes(axes, i, conv_filter, predicted_labels, labels)

            # Handle this method
            height_loc_list = [1.6, 1.45]
            fig = self._set_legend_dual_fig(fig, height_loc_list, feature_names)
            fig_list.append(fig)
        
        dual_fig_folder_path = "results/" + self.save_config.Output.batch_name + "/" + self.child_name + "/dual-figs/"
        if not os.path.exists(dual_fig_folder_path):
            os.mkdir(dual_fig_folder_path)

        for i in range(len(fig_list)):
            current_filter_dual_fig = ""
            current_filter_dual_fig += dual_fig_folder_path
            current_filter_dual_fig += "filter-{}/".format(i)
            if not os.path.exists(current_filter_dual_fig):
                os.mkdir(current_filter_dual_fig)

            dual_fig_png_path = current_filter_dual_fig + "dual-fig.png"
            dual_fig_eps_path = current_filter_dual_fig + "dual-fig.eps"
            fig_list[i].savefig(dual_fig_png_path, dpi=200, format="png", bbox_inches="tight")
            fig_list[i].savefig(dual_fig_eps_path, dpi=200, format="eps", bbox_inches="tight")

            plt.close(fig_list[i])

        print("Done saving dual figs!", flush=True)

    def _modify_axes(self, axes, filter_index, conv_filter, predicted_labels, labels):

        #### Below this is mostly old code, with some replaced variable names. Haven't touched it, because it's a mess

        mask_week_vector = self.subject.valid_mask
        color_map_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        min_label = 0-(conv_filter[:,0,0].shape[0]/2)
        max_label = 0+(conv_filter[:,0,0].shape[0]/2)
        num_labels = np.arange(min_label, max_label, 1)

        # Plot the filter axis
        for j in range(conv_filter.shape[1]):
            axes[0, 0].plot(conv_filter[:,j,filter_index], color=color_map_cycle[j])
        
        ########################
        ### Set x tick locations
        minx = 0
        maxx = conv_filter[:,0,0].shape[0]
        axes[0, 0].set_xticks(np.arange(minx, maxx+1, maxx/10))
    
        ### Set x tick labels
        maxx_seconds = (maxx/2) / 3000 * 60
        num_labels = np.arange(minx-maxx_seconds, maxx_seconds+1, maxx_seconds*2/10)
        num_labels = np.around(num_labels, decimals=2)
        axes[0, 0].set_xticklabels(map(str, num_labels))
        ########################
        
        # Plot the rate axis
        mask_predicted_labels = np.ma.masked_where(mask_week_vector == 0, predicted_labels[0,:,filter_index])
        
        axes[0, 1].plot(mask_predicted_labels, color=color_map_cycle[0])
        axes[0, 1].plot(labels[0,:,filter_index], color=color_map_cycle[1])
        
        num_labels = np.arange(-1, mask_predicted_labels.shape[0]+1, 2)
        axes[0, 1].set_xticklabels(map(str, num_labels))

        return axes


    def _set_legend_dual_fig(self, figure, height_loc_list, feature_names):

        color_map_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        rate_names = ["Predicted Labels", "Actual Labels"]
        
        ax_list = figure.axes
        
        handles = []
        for i in range(len(feature_names)):
            line = mlines.Line2D([], [], color=color_map_cycle[i], markersize=15, label=feature_names[i])
            handles.append(line)
        ax_list[0].legend(handles=handles, loc="upper right", bbox_to_anchor=(1, height_loc_list[0]), prop={"size":10})
        ax_list[0].set_xlabel("Time (sec)")
        ax_list[0].set_ylabel("Coefficient")
        
        handles[:] = []
        for i in range(len(rate_names)):
            line = mlines.Line2D([], [], color=color_map_cycle[i], markersize=15, label=rate_names[i])
            handles.append(line)
        ax_list[1].legend(handles=handles, loc="upper right", bbox_to_anchor=(1, height_loc_list[1]), prop={"size": 10})
        ax_list[1].set_xlabel("Week")
        ax_list[1].set_ylabel("Rate")
        
        return figure
        ### Old code ends here

    def _create_log(self, logs):

        logs["core_loss"] = logs["loss"]
        logs["core_loss"] -= logs["convolutional_activity_loss"]
        logs["core_loss"] -= logs["convolutional_dot_loss"]
        logs["core_loss"] -= logs["convolutional_l2_loss"]

        return logs





