import numpy as np
import tensorflow as tf
from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns

from model.model import RedShiftClassificationModel
from utils import MMapSequence, DataSetSampler


class MetricsEvaluator():
    def __init__(self, model, rs_num_bins=180, rs_max_val=0.4, fold_name='', dataset_loc='', is_plot=False):
        self.model = model
        self.compile_model()
        self.dataset_loc = dataset_loc
        self.fold_name = fold_name
        self.dataset_sampler = DataSetSampler(
            cube=np.load('{}/cube.npy'.format(self.dataset_loc), mmap_mode='r'),
            labels=np.load('{}/labels.npy'.format(self.dataset_loc), mmap_mode='r')
        )
        self.test_sequence = MMapSequence(
            cube=self.dataset_sampler.cube,
            labels=self.dataset_sampler.labels,
            idxes=self.dataset_sampler.test_indices,
            is_training=False
        )
        self.is_plot = is_plot
        self.rs_max_val = rs_max_val
        self.rs_num_bins = rs_num_bins

    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            metrics=['sparse_categorical_accuracy'],
            loss='sparse_categorical_accuracy'
        )

    def execute(self, weights_list):
        # Model 1, Model 2, Model 3 , Model 4, Model 5, Model 6
        # For each model, do it for all folds
        #   M1F1, M2F1, M3F
        # Execute your operation here!
        # self.model.load_weights(weights_list)
        residuals = []
        for j in range(len(self.test_sequence) - 1):
            y_preds = None
            for weight in weights_list:
                self.model.load_weights(weight)
                if y_preds is None:
                    y_preds = self.get_prediction(self.model.predict_on_batch(self.test_sequence[j][0]))
                else:
                    y_preds += self.get_prediction(self.model.predict_on_batch(self.test_sequence[j][0]))
            y_preds = y_preds / len(weights_list)
            y_truth = self.test_sequence[j][1]
            this_batch_residual = self.batch_residuals(y_preds, y_truth)
            residuals.append(this_batch_residual)

        residuals = np.array(residuals).flatten()
        if self.is_plot:
            self.plot_normalized_residuals(residuals)
        else:
            prediction_bias = np.average(residuals)
            sigma_mad = self.sigma_mad(residuals)
            n_outliers = self.eta_outliers(residuals, sigma_mad)
            print('Metrics for this fold({}) are:'.format(self.fold_name))
            fold_metrics_str = {
                'Fold ': [self.fold_name],
                'pred_bias': ['{:.5f}'.format(prediction_bias)],
                'mad': ['{:.5f}'.format(sigma_mad)],
                'eta_outliers': ['{:.5f}'.format(n_outliers)]
            }
            fold_metrics = {
                'Fold': self.fold_name,
                'pred_bias': prediction_bias,
                'mad': sigma_mad,
                'eta_outliers': n_outliers
            }
            print(tabulate(fold_metrics_str, headers='keys', tablefmt='pretty'))
            return fold_metrics

    def batch_residuals(self, y_pred, y_true):
        return (y_pred - y_true) / (y_true + 1)

    def get_prediction(self, y_pred):
        step = self.rs_max_val / self.rs_num_bins
        bins = np.arange(0, self.rs_max_val, step) + (step / 2)
        y_prediction = tf.reduce_sum(tf.multiply(y_pred, bins), axis=1)
        return y_prediction

    def sigma_mad(self, residuals):
        median = np.median(residuals)
        return 1.4826 * np.median(np.abs(residuals - median))

    def eta_outliers(self, residuals, dev_mad):
        num_outliers = np.sum(np.greater(np.abs(residuals), dev_mad * 5))
        return num_outliers / residuals.shape[0]

    def plot_normalized_residuals(self, residuals):
        sns.displot(residuals, kde=True)
        fig = plt.gcf()
        plt.xlabel('Delta Z')
        plt.ylabel('Relative Frequency')
        plt.savefig('residuals-dist-plot-{}.png'.format(self.fold_name))

    def average_crps(self):
        pass


class MetricsAggregator():

    def execute(self, fold1, fold2, fold3, fold4, fold5):
        # Execute your operation here!
        folds = [fold1, fold2, fold3, fold4, fold5]
        pred_bias = []
        mad = []
        n_outliers = []
        for fold in folds:
            pred_bias.append(fold['pred_bias'])
            mad.append(fold['mad'])
            n_outliers.append(fold['eta_outliers'])
        folds.append({
            'Fold': 'Avg',
            'pred_bias': np.mean(pred_bias),
            'mad': np.mean(mad),
            'eta_outliers': np.mean(n_outliers)
        })
        folds_dict = {
            'Fold': [],
            'pred_bias': [],
            'eta_outliers': [],
            'mad': []
        }
        for fold in folds:
            folds_dict['Fold'].append(fold['Fold'])
            folds_dict['pred_bias'].append('{:.5f}'.format(fold['pred_bias']))
            folds_dict['mad'].append('{:.5f}'.format(fold['mad']))
            folds_dict['eta_outliers'].append('{:.5f}'.format(fold['eta_outliers']))

        print(tabulate(folds_dict, headers='keys', tablefmt='pretty'))



def get_fold_weights_path():
    import glob
    NUM_FOLDS = 5
    models_weights = sorted(glob.glob('/home/umesh/redshift/results-glorot-uniform-*'))
    folds_weights_dict = {}
    for weights_dir in models_weights:
        for j in range(1, NUM_FOLDS+1):
            last_weight_should_be = 30
            weights_path = []
            weights_path_list = folds_weights_dict.get(f'fold_{j}', [])
            while len(weights_path) == 0:
                weights_path = glob.glob(weights_dir + '/' + f'99-percent-of-dataset/train_fold_{j}/weights.{last_weight_should_be}.hdf5')
                last_weight_should_be -= 1
            weights_path_list.append(weights_path[0])
            folds_weights_dict[f'fold_{j}'] = weights_path_list
    return folds_weights_dict



def main():
    fold_weights = get_fold_weights_path()
    folds = []
    for j in range(1, 6):
        model = RedShiftClassificationModel(
            (64, 64, 5),
            redenning_shape=(1,)
        )
        mme = MetricsEvaluator(
            model=model,
            dataset_loc='/home/umesh/redshift/dataset',
            fold_name=f'Fold {j}: All Data',
            is_plot=False
        )
        folds.append(mme.execute(fold_weights.get(f'fold_{j}')))

    aggregator = MetricsAggregator()
    aggregator.execute(*folds)


if __name__ == '__main__':
    print(get_fold_weights_path())

