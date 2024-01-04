from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def get_sr_metrics(metrics_type):
    if metrics_type == 'psnr':
        metrics = PeakSignalNoiseRatio(data_range=1.0, base=10.0, 
                                        reduction='elementwise_mean', dim=(2,3))
    elif metrics_type == 'ssim':
        metrics = StructuralSimilarityIndexMeasure(gaussian_kernel=True, sigma=1.5, 
                                                    kernel_size=11, 
                                                    reduction='elementwise_mean', 
                                                    data_range=1.0, k1=0.01, k2=0.03, 
                                                    return_full_image=False, 
                                                    return_contrast_sensitivity=False)
    return metrics