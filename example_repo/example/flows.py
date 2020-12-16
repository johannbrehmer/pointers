import logging
from nflows import distributions, flows, transforms
from nflows.nn import nets
import nflows.utils.torchutils as flowutils

logger = logging.getLogger(__name__)


def make_scalar_flow(
    dim,
    flow_steps=5,
    transform_type="rq",
    linear_transform="none",
    bins=10,
    tail_bound=10.0,
    hidden_features=64,
    num_transform_blocks=3,
    use_batch_norm=False,
    dropout_prob=0.0,
):
    logger.info(
        f"Creating flow for {dim}-dimensional unstructured data, using {flow_steps} blocks of {transform_type} transforms, "
        f"each with {num_transform_blocks} transform blocks and {hidden_features} hidden units, interlaced with {linear_transform} "
        f"linear transforms"
    )

    base_dist = distributions.StandardNormal((dim,))

    transform = []
    for i in range(flow_steps):
        if linear_transform != "none":
            transform.append(_make_scalar_linear_transform(linear_transform, dim))
        transform.append(
            _make_scalar_base_transform(
                i,
                dim,
                transform_type,
                bins,
                tail_bound,
                hidden_features,
                num_transform_blocks,
                use_batch_norm,
                dropout_prob=dropout_prob,
            )
        )
    if linear_transform != "none":
        transform.append(_make_scalar_linear_transform(linear_transform, dim))

    transform = transforms.CompositeTransform(transform)
    flow = flows.Flow(transform, base_dist)

    return flow


def make_image_flow(
    chw,
    levels=7,
    steps_per_level=3,
    transform_type="rq",
    bins=4,
    tail_bound=3.0,
    hidden_channels=96,
    act_norm=True,
    batch_norm=False,
    dropout_prob=0.0,
    alpha=0.05,
    num_bits=8,
    preprocessing="glow",
    residual_blocks=3,
):
    c, h, w = chw
    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    # Base density
    base_dist = distributions.StandardNormal((c * h * w,))
    logger.debug(f"Base density: standard normal in {c * h * w} dimensions")

    # Preprocessing: Inputs to the model in [0, 2 ** num_bits]
    if preprocessing == "glow":
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits), shift=-0.5)
    elif preprocessing == "realnvp":
        preprocess_transform = transforms.CompositeTransform(
            [
                # Map to [0,1]
                transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits)),
                # Map into unconstrained space as done in RealNVP
                transforms.AffineScalarTransform(shift=alpha, scale=(1 - alpha)),
                transforms.Logit(),
            ]
        )
    elif preprocessing == "realnvp_2alpha":
        preprocess_transform = transforms.CompositeTransform(
            [
                transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits)),
                transforms.AffineScalarTransform(shift=alpha, scale=(1 - 2.0 * alpha)),
                transforms.Logit(),
            ]
        )
    else:
        raise RuntimeError("Unknown preprocessing type: {}".format(preprocessing))

    logger.debug(f"{preprocessing} preprocessing")

    # Multi-scale transform
    logger.debug("Input: c, h, w = %s, %s, %s", c, h, w)
    mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
    for level, level_hidden_channels in zip(range(levels), hidden_channels):
        logger.debug("Level %s", level)
        squeeze_transform = transforms.SqueezeTransform()
        c, h, w = squeeze_transform.get_output_shape(c, h, w)
        logger.debug("  c, h, w = %s, %s, %s", c, h, w)
        transform_level = [squeeze_transform]
        logger.debug("  SqueezeTransform()")

        for _ in range(steps_per_level):
            transform_level.append(
                _make_image_base_transform(
                    c,
                    level_hidden_channels,
                    act_norm,
                    transform_type,
                    residual_blocks,
                    batch_norm,
                    dropout_prob,
                    tail_bound,
                    bins,
                )
            )

        transform_level.append(transforms.OneByOneConvolution(c))  # End each level with a linear transformation
        logger.debug("  OneByOneConvolution(%s)", c)
        transform_level = transforms.CompositeTransform(transform_level)

        new_shape = mct.add_transform(transform_level, (c, h, w))
        if new_shape:  # If not last layer
            c, h, w = new_shape
            logger.debug("  new_shape = %s, %s, %s", c, h, w)

    # Full transform and flow
    transform = transforms.CompositeTransform([preprocess_transform, mct])
    flow = flows.Flow(transform, base_dist)

    return flow


def _make_image_base_transform(
    num_channels,
    hidden_channels,
    actnorm,
    transform_type,
    num_res_blocks,
    resnet_batchnorm,
    dropout_prob,
    tail_bound,
    num_bins,
    apply_unconditional_transform=False,
    min_bin_width=0.001,
    min_bin_height=0.001,
    min_derivative=0.001,
):
    def convnet_factory(in_channels, out_channels):
        net = nets.ConvResidualNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_res_blocks,
            use_batch_norm=resnet_batchnorm,
            dropout_probability=dropout_prob,
        )
        return net

    mask = flowutils.create_mid_split_binary_mask(num_channels)

    if transform_type == "cubic":
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=convnet_factory,
            tails="linear",
            tail_bound=tail_bound,
            num_bins=num_bins,
            apply_unconditional_transform=apply_unconditional_transform,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
        )
    elif transform_type == "quadratic":
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=convnet_factory,
            tails="linear",
            tail_bound=tail_bound,
            num_bins=num_bins,
            apply_unconditional_transform=apply_unconditional_transform,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
        )
    elif transform_type == "rq":
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=convnet_factory,
            tails="linear",
            tail_bound=tail_bound,
            num_bins=num_bins,
            apply_unconditional_transform=apply_unconditional_transform,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
    elif transform_type == "affine":
        coupling_layer = transforms.AffineCouplingTransform(mask=mask, transform_net_create_fn=convnet_factory)
    elif transform_type == "additive":
        coupling_layer = transforms.AdditiveCouplingTransform(mask=mask, transform_net_create_fn=convnet_factory)
    else:
        raise RuntimeError("Unknown transform type")

    step_transforms = []
    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))
    step_transforms.extend([transforms.OneByOneConvolution(num_channels), coupling_layer])
    transform = transforms.CompositeTransform(step_transforms)

    logger.debug(f"  Block with {transform_type} coupling layers")

    return transform


class _TweakedUniform(distributions.uniform.BoxUniform):
    """ Thin wrapper around the BoxUniform distribution in the nflows package """

    def log_prob(self, value, context):
        return flowutils.sum_except_batch(super().log_prob(value))

    def sample(self, num_samples, context):
        return super().sample((num_samples,))


def _make_scalar_base_transform(
    i,
    dim,
    transform="rq",
    bins=10,
    tail_bound=10.0,
    hidden_features=64,
    num_transform_blocks=2,
    use_batch_norm=False,
    dropout_prob=0.0,
):
    """ Creates the main transformation block for our flow """

    def transform_net_factory(in_features, out_features):
        return nets.ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            num_blocks=num_transform_blocks,
            dropout_probability=dropout_prob,
            use_batch_norm=use_batch_norm,
        )

    if transform == "affine":
        return transforms.AffineCouplingTransform(
            mask=flowutils.create_alternating_binary_mask(features=dim, even=(i % 2 == 0)),
            transform_net_create_fn=transform_net_factory,
            unconditional_transform=None,
        )
    elif transform == "rq":
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=flowutils.create_alternating_binary_mask(features=dim, even=(i % 2 == 0)),
            transform_net_create_fn=transform_net_factory,
            num_bins=bins,
            apply_unconditional_transform=False,
            tail_bound=tail_bound,
            tails="linear",
        )
    else:
        raise ValueError(transform)


def _make_scalar_linear_transform(transform, features):
    if transform == "permutation":
        return transforms.RandomPermutation(features=features)
    elif transform == "lu":
        return transforms.CompositeTransform(
            [transforms.RandomPermutation(features=features), transforms.LULinear(features, identity_init=True)]
        )
    elif transform == "svd":
        return transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=features),
                transforms.SVDLinear(features, num_householder=10, identity_init=True),
            ]
        )
    else:
        raise ValueError
