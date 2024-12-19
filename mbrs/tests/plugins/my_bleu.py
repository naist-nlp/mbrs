from mbrs.metrics import MetricBLEU, register


@register("my_bleu")
class MetricMyBLEU(MetricBLEU):
    """My customized metric class."""
