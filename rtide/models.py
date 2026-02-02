import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Layer, Concatenate
from tensorflow.keras.models import Model, Sequential

from .utils import AddLayer


@tf.keras.utils.register_keras_serializable(package="rtide")
class SineLayer(Layer):
    """SIREN-style sine layer.

    Implements: y = sin(w0 * (Wx + b)) with SIREN-recommended initialization.

    Parameters
    ----------
    units : int
        Number of hidden units.
    w0 : float
        Frequency factor (omega_0) used inside the sine nonlinearity.
    is_first : bool
        Whether this is the first layer (uses different init range).
    l1_strength, l2_strength : float
        Optional kernel regularization.
    """

    def __init__(
        self,
        units: int,
        w0: float = 1.0,
        is_first: bool = False,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        self.l1_strength = float(l1_strength)
        self.l2_strength = float(l2_strength)
        self._dense = None

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if self.is_first:
            # SIREN paper: U(-1/in_dim, 1/in_dim)
            scale = 1.0 / max(1, in_dim)
        else:
            # SIREN paper: U(-sqrt(6/in_dim)/w0, sqrt(6/in_dim)/w0)
            scale = (6.0 / max(1, in_dim)) ** 0.5 / max(1e-8, self.w0)

        initializer = tf.keras.initializers.RandomUniform(minval=-scale, maxval=scale)
        self._dense = Dense(
            self.units,
            kernel_initializer=initializer,
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_strength, l2=self.l2_strength),
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.sin(self.w0 * self._dense(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "w0": self.w0,
                "is_first": self.is_first,
                "l1_strength": self.l1_strength,
                "l2_strength": self.l2_strength,
            }
        )
        return config
    


@tf.keras.utils.register_keras_serializable(package="rtide")
class SineActivation(Layer):
    """Sine activation layer for SIREN networks."""
    
    def __init__(self, omega_0=30.0, **kwargs):
        super(SineActivation, self).__init__(**kwargs)
        self.omega_0 = omega_0

    def call(self, inputs):
        return tf.sin(self.omega_0 * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"omega_0": self.omega_0})
        return config


@tf.keras.utils.register_keras_serializable(package="rtide")
class TrendLayer(Layer):
    """Layer for computing linear or quadratic trends.
    
    This layer takes normalized time values (0 to 1) and computes trend components
    that are added to the model output. The trend is learned during training.
    
    Parameters
    ----------
    trend_type : str
        Either 'linear' or 'quadratic'. Determines the polynomial order of the trend.
    n_outputs : int
        Number of output channels (1 for elevation, 2 for currents u/v).
    """
    
    def __init__(self, trend_type='linear', n_outputs=1, initial_coeffs = None, **kwargs):
        super(TrendLayer, self).__init__(**kwargs)
        self.trend_type = trend_type
        self.n_outputs = n_outputs
        self.initial_coeffs = initial_coeffs  # NEW parameter
        
        if trend_type not in ['linear', 'quadratic']:
            raise ValueError(f"trend_type must be 'linear' or 'quadratic', got {trend_type}")
    
    def build(self, input_shape):
        # input_shape: (batch_size, 1) for time values
        if self.trend_type == 'linear':
            # y = a*t + b
            slope_init = self.initial_coeffs.get('slope', 0.0)
            self.trend_weights = self.add_weight(
                initializer=tf.keras.initializers.Constant(slope_init),
                shape=(self.n_outputs,),
                trainable=True  # Still trainable - backprop refines it!
            )
            bias_init = self.initial_coeffs.get('intercept', 0.0)
            self.trend_bias = self.add_weight(
                name='linear_bias',
                shape=(self.n_outputs,),
                initializer=tf.keras.initializers.Constant(bias_init),
                trainable=True
            )
        else:  # quadratic
            # y = a*t^2 + b*t + c
            a_init = self.initial_coeffs.get('a', 0.0)
            self.trend_weights_quad = self.add_weight(
                name='quadratic_coef',
                shape=(self.n_outputs,),
                initializer=tf.keras.initializers.Constant(a_init),
                trainable=True
            )
            b_init = self.initial_coeffs.get('b', 0.0)
            self.trend_weights_lin = self.add_weight(
                name='linear_coef',
                shape=(self.n_outputs,),
                initializer=tf.keras.initializers.Constant(b_init),
                trainable=True
            )
            c_init = self.initial_coeffs.get('c', 0.0)
            self.trend_bias = self.add_weight(
                name='constant_bias',
                shape=(self.n_outputs,),
                initializer=tf.keras.initializers.Constant(c_init),
                trainable=True
            )
        
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs: (batch_size, 1) normalized time values
        t = inputs
        
        if self.trend_type == 'linear':
            # Broadcast multiplication for multi-output
            trend = t * self.trend_weights + self.trend_bias
        else:  # quadratic
            trend = (t * t) * self.trend_weights_quad + t * self.trend_weights_lin + self.trend_bias
        
        return trend
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'trend_type': self.trend_type,
            'n_outputs': self.n_outputs
        })
        return config


def siren_initializer(scale=1.0):
    """Weight initialization for SIREN layers."""
    def initializer(shape, dtype=None):
        return tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=dtype)
    return initializer


def build_response_model(
    input_dims: int,
    n_outputs: int,
    hidden_nodes: int,
    depth: int,
    l1_strength: float = 0.0,
    l2_strength: float = 0.0,
    trend: str = None,
    trend_initial_coeffs: dict = None,
):
    """Build the RTide 'response method' network.

    Parameters
    ----------
    input_dims : int
        Number of input features (forcing functions).
    n_outputs : int
        Number of outputs (1 for elevation, 2 for currents).
    hidden_nodes : int
        Number of hidden units in each layer.
    depth : int
        Number of hidden layers.
    l1_strength, l2_strength : float
        Regularization strengths.
    trend : str or None
        Trend type: None (no trend), 'linear', or 'quadratic'.
        When specified, a separate trend component is estimated during training.

    Returns
    -------
    model : keras.Model
        Compiled model (functional API if trend is used, Sequential otherwise).
    """
    if n_outputs < 1:
        raise ValueError("n_outputs must be >= 1")

    # If no trend, use legacy Sequential model
    if trend is None:
        model = Sequential(name="rtide_response")

        model.add(
            Dense(
                hidden_nodes,
                input_dim=input_dims,
                activation="tanh",
                name="nonlinear_layer1",
                kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
            )
        )

        # Hidden stack
        if depth > 1:
            for layer_number in range(2, depth + 1):
                units = input_dims if layer_number < depth else input_dims * n_outputs
                model.add(
                    Dense(
                        units,
                        activation="tanh",
                        name=f"nonlinear_layer_{layer_number}",
                        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
                    )
                )
        else:
            # depth==1 special-case for multi-output
            if n_outputs > 1:
                model.add(
                    Dense(
                        input_dims * n_outputs,
                        activation="tanh",
                        name="nonlinear_layer_out",
                        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
                    )
                )

        if n_outputs == 1:
            model.add(AddLayer(name="add_layer", n_outputs=1))
        else:
            model.add(Dense(
                2,
                activation=None,
                name='output_layer'
            ))

        return model
    
    # With trend: use Functional API
    # Input: [forcing_features, normalized_time]
    forcing_input = Input(shape=(input_dims,), name="forcing_input")
    time_input = Input(shape=(1,), name="time_input")
    
    # Build the same network structure using Functional API
    x = Dense(
        hidden_nodes,
        activation="tanh",
        name="nonlinear_layer1",
        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
    )(forcing_input)
    
    if depth > 1:
        for layer_number in range(2, depth + 1):
            units = input_dims if layer_number < depth else input_dims * n_outputs
            x = Dense(
                units,
                activation="tanh",
                name=f"nonlinear_layer_{layer_number}",
                kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
            )(x)
    else:
        if n_outputs > 1:
            x = Dense(
                input_dims * n_outputs,
                activation="tanh",
                name="nonlinear_layer_out",
                kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
            )(x)
    
    # Final prediction layer
    if n_outputs == 1:
        tide_output = AddLayer(name="add_layer", n_outputs=1)(x)
    else:
        tide_output = Dense(2, activation=None, name='tide_layer')(x)
    
    # Trend component
    trend_output = TrendLayer(trend_type=trend, n_outputs=n_outputs, initial_coeffs = trend_initial_coeffs, name='trend_layer')(time_input)
    
    # Combine tide + trend
    if n_outputs == 1:
        # For scalar output, simple addition
        output = tf.keras.layers.Add(name='output')([tide_output, trend_output])
    else:
        # For vector output, element-wise addition
        output = tf.keras.layers.Add(name='output')([tide_output, trend_output])
    
    model = Model(inputs=[forcing_input, time_input], outputs=output, name="rtide_response_with_trend")
    return model


def build_siren_model(
    input_dims: int,
    n_outputs: int,
    hidden_nodes: int,
    depth: int,
    siren_w0: float = 30.0,
    siren_w0_initial: float = 30.0,
    l1_strength: float = 0.0,
    l2_strength: float = 0.0,
    trend: str = None,
    trend_initial_coeffs: dict = None,
):
    """Build a SIREN network (sinusoidal activations + SIREN init).

    Parameters
    ----------
    input_dims : int
        Number of input features (forcing functions).
    n_outputs : int
        Number of outputs (1 for elevation, 2 for currents).
    hidden_nodes : int
        Number of hidden units in each SIREN layer.
    depth : int
        Number of SIREN layers (must be >= 1).
    siren_w0 : float
        Frequency factor for hidden layers.
    siren_w0_initial : float
        Frequency factor for first layer (typically same as siren_w0).
    l1_strength, l2_strength : float
        Regularization strengths.
    trend : str or None
        Trend type: None (no trend), 'linear', or 'quadratic'.

    Returns
    -------
    model : keras.Model
        SIREN model with optional trend estimation.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1 for SIREN")

    # If no trend, use legacy implementation
    if trend is None:
        inputs = Input(shape=(input_dims,), name="inputs")
        x = SineLayer(
            hidden_nodes,
            w0=siren_w0,
            is_first=True,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            name="sine_1",
        )(inputs)

        for i in range(2, depth + 1):
            x = SineLayer(
                hidden_nodes,
                w0=siren_w0,
                is_first=False,
                l1_strength=l1_strength,
                l2_strength=l2_strength,
                name=f"sine_{i}",
            )(x)

        # Final linear layer
        outputs = Dense(
            n_outputs,
            activation="linear",
            name="output",
            kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
        )(x)

        return Model(inputs=inputs, outputs=outputs, name="rtide_siren")
    
    # With trend: dual-input model
    forcing_input = Input(shape=(input_dims,), name="forcing_input")
    time_input = Input(shape=(1,), name="time_input")
    
    # SIREN layers for forcing
    x = SineLayer(
        hidden_nodes,
        w0=siren_w0,
        is_first=True,
        l1_strength=l1_strength,
        l2_strength=l2_strength,
        name="sine_1",
    )(forcing_input)

    for i in range(2, depth + 1):
        x = SineLayer(
            hidden_nodes,
            w0=siren_w0,
            is_first=False,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            name=f"sine_{i}",
        )(x)

    # Tide component
    tide_output = Dense(
        n_outputs,
        activation="linear",
        name="tide_output",
        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
    )(x)
    
    # Trend component
    trend_output = TrendLayer(trend_type=trend, n_outputs=n_outputs, initial_coeffs = trend_initial_coeffs, name='trend_layer')(time_input)
    
    # Combine
    output = tf.keras.layers.Add(name='output')([tide_output, trend_output])
    
    return Model(inputs=[forcing_input, time_input], outputs=output, name="rtide_siren_with_trend")


def build_siren_model_mod(
    input_dims: int,
    n_outputs: int,
    hidden_nodes: int,
    depth: int,
    siren_w0: float = 30.0,
    l1_strength: float = 0.0,
    l2_strength: float = 0.0,
    trend: str = None,
):
    """Build a SIREN network using Sequential API with custom activation layers.

    Note: This variant does not support trend estimation (trend parameter is ignored).
    Use build_siren_model() for trend estimation support.

    Parameters
    ----------
    input_dims : int
        Number of input features.
    n_outputs : int
        Number of outputs.
    hidden_nodes : int
        Number of hidden units.
    depth : int
        Number of SIREN layers (must be >= 1).
    siren_w0 : float
        Frequency factor.
    l1_strength, l2_strength : float
        Regularization strengths.
    trend : str or None
        Ignored in this variant (use build_siren_model for trend support).
    """
    if depth < 1:
        raise ValueError("depth must be >= 1 for SIREN")
    
    if trend is not None:
        raise ValueError(
            "build_siren_model_mod does not support trend estimation. "
            "Use build_siren_model() with trend parameter instead."
        )

    scale = 1 / input_dims
    
    model = Sequential(name="rtide_siren")
    model.add(Dense(
        hidden_nodes,
        input_dim=input_dims,
        activation=None,
        kernel_initializer=siren_initializer(scale=scale),
        kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
        name='siren_layer1'
    ))
    model.add(SineActivation(omega_0=siren_w0))
    
    for layer_number in range(2, depth + 1):
        model.add(Dense(
            hidden_nodes,
            activation=None,
            kernel_initializer=siren_initializer(scale=scale),
            kernel_regularizer=regularizers.l1_l2(l1=l1_strength, l2=l2_strength),
            name=f'siren_layer_{layer_number}'
        ))
        model.add(SineActivation(omega_0=siren_w0))

    model.add(Dense(
        n_outputs,
        activation=None,
        name="output",
        kernel_initializer=siren_initializer(scale=scale),
    ))

    return model


def build_model(
    architecture: str,
    input_dims: int,
    n_outputs: int,
    hidden_nodes: int,
    depth: int,
    l1_strength: float = 0.0,
    l2_strength: float = 0.0,
    siren_w0: float = 30.0,
    trend: str = None,
    trend_initial_coeffs: dict = None,
):
    """Factory for RTide model architectures.
    
    Parameters
    ----------
    architecture : str
        Model architecture: 'response'/'rtide'/'tanh' or 'siren'/'sine'.
    input_dims : int
        Number of input features.
    n_outputs : int
        Number of outputs (1 for elevation, 2 for currents).
    hidden_nodes : int
        Number of hidden units per layer.
    depth : int
        Number of layers.
    l1_strength, l2_strength : float
        Regularization strengths.
    siren_w0 : float
        SIREN frequency parameter (only used for SIREN architecture).
    trend : str or None
        Trend estimation: None (default, no trend), 'linear', or 'quadratic'.
        When specified, the model will simultaneously learn both tidal dynamics
        and a polynomial trend component during training.
    
    Returns
    -------
    model : keras.Model
        Compiled Keras model ready for training.
        
    Notes
    -----
    When trend is specified:
    - Model expects dual inputs: [forcing_features, normalized_time]
    - The normalized_time should be scaled 0-1 over the training period
    - Trend coefficients are learned during training alongside tidal parameters
    - Legacy models (trend=None) remain fully compatible
    """
    arch = (architecture or "response").lower().strip()

    if arch in {"standard", "response", "rtide", "tanh"}:
        return build_response_model(
            input_dims=input_dims,
            n_outputs=n_outputs,
            hidden_nodes=hidden_nodes,
            depth=depth,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            trend=trend,
            trend_initial_coeffs=trend_initial_coeffs,
        )

    if arch in {"siren", "sine"}:
        return build_siren_model(
            input_dims=input_dims,
            n_outputs=n_outputs,
            hidden_nodes=hidden_nodes,
            depth=depth,
            siren_w0=siren_w0,
            l1_strength=l1_strength,
            l2_strength=l2_strength,
            trend=trend,
            trend_initial_coeffs=trend_initial_coeffs,
        )

    raise ValueError(
        f"Unknown architecture '{architecture}'. Expected one of: response, siren."
    )


def get_custom_objects():
    """Custom objects needed to `load_model()` saved RTide models."""
    return {
        "AddLayer": AddLayer,
        "SineActivation": SineActivation,
        "SineLayer": SineLayer,
        "TrendLayer": TrendLayer,
    }