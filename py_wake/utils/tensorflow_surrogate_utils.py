import os
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pickle
from pathlib import Path
import warnings


def extra_data_pkl2json(path):  # pragma: no cover

    file_list = [x[0] for x in os.walk(path)]
    file_list.pop(0)
    for f in file_list:
        model = tf.keras.models.load_model(os.path.join(f, 'model.h5'))
        with open(os.path.join(f, 'save_dic.pkl'), 'rb') as g:
            save_dic = pickle.load(g)
        save_dic["wind_speed_cut_in"] = 4.0
        save_dic["wind_speed_cut_out"] = 25.0

        save_dic['wohler_exponent'] = {'Blade': 10, 'Tower': 4, 'Power': None}[os.path.basename(f)[:5]]
        scaler_attrs = ["feature_range", "copy", "n_features_in_", "n_samples_seen_",
                        "scale_", "min_", "data_min_", "data_max_", "data_range_"]

        def fmt(v):
            return v if isinstance(v, (int, float)) else list(v)

        for n in ['input', 'output']:
            scaler = save_dic.pop(f'{n}_scaler')
            save_dic[f'{n}_scalers'] = {'operation': {k: fmt(getattr(scaler, k)) for k in scaler_attrs}}
        with open(os.path.join(f, 'extra_data.json'), 'w') as fid:
            json.dump(save_dic, fid, indent=4)
        # model.save(f'{f}/model_set_operation.tf')


class TensorflowSurrogate():

    def __init__(self, path, set_name):

        # Load extra data.
        path = Path(path)
        with open(path / 'extra_data.json') as fid:
            extra_data = json.load(fid)

        self.input_channel_names = extra_data['input_channel_names']
        self.output_channel_name = extra_data['output_channel_name']
        self.wind_speed_cut_in = extra_data['wind_speed_cut_in']
        self.wind_speed_cut_out = extra_data['wind_speed_cut_out']
        if 'wohler_exponent' in extra_data:
            self.wohler_exponent = extra_data['wohler_exponent']

        # Create the MinMaxScaler scaler objects.
        def json2scaler(d):
            scaler = MinMaxScaler()
            for k, v in d.items():
                setattr(scaler, k, v)
            return scaler

        self.input_scaler = json2scaler(extra_data['input_scalers'][set_name])
        self.output_scaler = json2scaler(extra_data['output_scalers'][set_name])

        self.model = tf.keras.models.load_model(path / f'model_set_{set_name}.h5')

    def predict_output(self, x, bounds='warn'):
        """
        Predict the response of a model.

        Parameters
        ----------
        x : array_like
            2D array of input on which evaluate the model, shape=(#samples,"input_vars)


        Returns
        -------
        output : numpy.ndarray
            Model output, optionally scaled through output_scaler.
            2D array, where each row is a different sample, and each column a
            different output.

        Raises
        ------
        Warning: if some points are outside of the boundary.

        """

        # Scale the input.
        x_scaled = self.input_scaler.transform(x)
        assert bounds in ['warn', 'ignore']
        if bounds == 'warn':
            if x_scaled.min() < self.input_scaler.feature_range[0]:
                for i, k in enumerate(self.input_channel_names):
                    min_v = x[:, i].min()
                    if min_v < self.input_scaler.data_min_[i]:
                        mi, ma = self.input_scaler.data_min_[i], self.input_scaler.data_max_[i]
                        warnings.warn(f"Input, {k}, with value, {min_v} outside range {mi}-{ma}")
            if x_scaled.max() > self.input_scaler.feature_range[1]:
                for i, k in enumerate(self.input_channel_names):
                    max_v = x[:, i].max()
                    if max_v > self.input_scaler.data_max_[i]:
                        mi, ma = self.input_scaler.data_min_[i], self.input_scaler.data_max_[i]
                        warnings.warn(f"Input, {k}, with value, {max_v} outside range {mi}-{ma}")

        return self.output_scaler.inverse_transform(self.model.predict(x_scaled))

    @property
    def input_space(self):
        i_s = self.input_scaler
        return {k: (mi, ma) for k, mi, ma in zip(self.input_channel_names, i_s.data_min_, i_s.data_max_)}


if __name__ == '__main__':  # pragma: no cover
    extra_data_pkl2json(r'C:\mmpe\programming\python\Topfarm\PyWake\py_wake\examples\data\dtu10mw\surrogates')
