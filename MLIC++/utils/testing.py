import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import compute_metrics
from utils.utils import *

import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import copy


def test_one_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            rec = torch2img(out_net['x_hat'])
            img = torch2img(d)
            p, m = compute_metrics(rec, img)
            psnr.update(p)
            ms_ssim.update(m)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            img.save(os.path.join(save_dir, '%03d_gt.png' % i))

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {loss.avg:.4f} | "
            f"MS-SSIM loss: {ms_ssim_loss.avg:.6f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {ms_ssim.avg:.6f}"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss.avg

def compress_one_image(model, x, stream_path, H, W, img_name):
    with torch.no_grad():
        out = model.compress(x)

    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])

    size = filesize(output)
    bpp = float(size) * 8 / (H * W)
    return bpp, out["cost_time"], out["y"], out["uncompressed_bpp"]


def decompress_one_image(model, stream_path, img_name, y):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)

    with torch.no_grad():
        out = model.decompress(strings, shape, y)

    x_hat = out["x_hat"]
    y_hat = out["y_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    cost_time = out["cost_time"]
    return x_hat, y_hat, cost_time, out["decompressed_bpp"]



def test_model(test_dataloader, net, logger_test, save_dir, epoch, gpu_id):
    net.eval()
    device = next(net.parameters()).device

    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_bpp = AverageMeter()
    avg_enc_time = AverageMeter()
    avg_dec_time = AverageMeter()

    with torch.no_grad():
        for i, img in enumerate(test_dataloader):
            img = img.to(device)
            B, C, H, W = img.shape
            img_name = f"{H}x{W}"
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            # warmup GPU
            if i == 0:
                bpp, enc_time, _, _ = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            # avoid resolution leakage
            net.update_resolutions(16, 16)
            bpp, enc_time, y, uncompressed_bpp = compress_one_image(model=net, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(i))
            # avoid resolution leakage
            net.update_resolutions(16, 16)
            x_hat, y_hat, dec_time, decompressed_bpp = decompress_one_image(model=net, stream_path=save_dir, img_name=str(i), y=y)
            rec = torch2img(x_hat)
            img = torch2img(img)
            img.save(os.path.join(save_dir, f'{img_name}_gt.png'))
            rec.save(os.path.join(save_dir, f'{img_name}_rec.png'))
            p, m = compute_metrics(rec, img)
            avg_psnr.update(p)
            avg_ms_ssim.update(m)
            avg_bpp.update(bpp)
            avg_enc_time.update(enc_time)
            avg_dec_time.update(dec_time)
            logger_test.info(
                f"{img_name} | "
                f"Bpp loss: {bpp:.2f} | "
                f"PSNR: {p:.4f} | "
                f"MS-SSIM: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding Latency: {dec_time:.4f}"
            )

            #(ADDED)
            
            # Initialize lists to store channel-wise PSNR and MS-SSIM differences
            channel_masked_psnr = []
            channel_ms_ssim_prime = []
            psnr_diff_list = []
            num_channels = y_hat.shape[1]
            keep_top_psnr = []
            disgard_top_psnr = []

            #print(mse_matrix)
            # Evaluate the importance of each channel
            for ch in range(num_channels):
                
                # Set the channel to zero in y_hat
                y_hat_zeroed = y_hat.clone()  # Assuming x_hat is y_hat from decompress function
                y_hat_zeroed[:, ch, :, :] = 0

                # avoid resolution leakage
                net.update_resolutions(16, 16)
                rec_prime = net.g_s(y_hat_zeroed)
                
                rec_prime = rec_prime[:,:,0:H,0:W]
                rec_prime = torch2img(rec_prime)

                # Calculate PSNR and MS-SSIM
                psnr_prime, ms_ssim_prime = compute_metrics(rec_prime, img)

                # Calculate the difference
                psnr_diff = p - psnr_prime
                # ms_ssim_diff = m - ms_ssim_prime

                channel_masked_psnr.append(psnr_prime)
                channel_ms_ssim_prime.append(ms_ssim_prime)
                psnr_diff_list.append(psnr_diff)

                if psnr_diff >= 5:
                    rec_prime.save(os.path.join(save_dir, f'{gpu_id}_{img_name}_rec_channel_{ch}_psnr_{psnr_diff:.1f}.png'))

            top_16 = np.argsort(psnr_diff_list)[::-1][:16]
            top_16_bpp_before_channel = np.argsort(uncompressed_bpp)[::-1][:16]
            top_16_bpp_after_channel = np.argsort(decompressed_bpp)[::-1][:16]

            print("top 16 channels found.")
            print("Reconstruct image process started.")

            y_hat_only_top = y_hat.clone()
            y_hat_without_top = y_hat.clone()

            for ch in range(num_channels):
                if ch in top_16:
                    y_hat_without_top[:, ch, :, :] = 0
                else:
                    y_hat_only_top[:, ch, :, :] = 0
            
            # reconstruct images from modified y_hat
            net.update_resolutions(16, 16)
            rec_only_top = net.g_s(y_hat_only_top)
            rec_only_top = rec_only_top[:,:,0:H,0:W]
            rec_only_top = torch2img(rec_only_top)

            net.update_resolutions(16, 16)
            rec_without_top = net.g_s(y_hat_without_top)
            rec_without_top = rec_without_top[:,:,0:H,0:W]
            rec_without_top = torch2img(rec_without_top)

            # calculate PSNR and MS-SSIM
            psnr_only_top, ms_ssim_only_top = compute_metrics(rec_only_top, img)
            psnr_without_top, ms_ssim_without_top = compute_metrics(rec_without_top, img)

            rec_only_top.save(os.path.join(save_dir, f'{gpu_id}_{img_name}_only_top16_psnr{psnr_only_top:.2f}.png'))
            rec_without_top.save(os.path.join(save_dir, f'{gpu_id}_{img_name}_without_top16_psnr{psnr_without_top:.2f}.png'))

            logger_test.info(
                f"{img_name} | "
                f"Top 16 channels idx: {top_16} | "
                f"Top 16 bpp_before channels idx: {top_16_bpp_before_channel} | "
                f"Top 16 bpp_after channels idx: {top_16_bpp_after_channel} | "
                f"bpp_before: {uncompressed_bpp}"
                f"bpp_after: {decompressed_bpp} | "
                f"psnr_diff: {psnr_diff_list} | "
                f"PSNR only top: {psnr_only_top:.4f} | "
                f"PSNR without top: {psnr_without_top:.4f}"
            )

            print(f"Image[{i}] finished")
    logger_test.info(
        f"Epoch:[{epoch}] | "
        f"Avg Bpp: {avg_bpp.avg:.4f} | "
        f"Avg PSNR: {avg_psnr.avg:.4f} | "
        f"Avg MS-SSIM: {avg_ms_ssim.avg:.4f} | "
        f"Avg Encoding Latency:: {avg_enc_time.avg:.4f} | "
        f"Avg decoding Latency:: {avg_dec_time.avg:.4f}"
    )
