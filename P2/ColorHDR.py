import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import cv2
from skimage import color


def calcular_luminancia(imagenrgb, eps=1e-6):
    rgb_pos = np.maximum(imagenrgb.astype(np.float64), 0.0)
    Y = 0.177 * rgb_pos[..., 0] + 0.813 * rgb_pos[..., 1] + 0.011 * rgb_pos[..., 2]
    return np.maximum(Y, eps)


def normalizar_rango(imagen, p_low=0.5, p_high=99.5, gamma=2.2):
    bajo, alto = np.percentile(imagen, [p_low, p_high])
    img_norm = np.clip((imagen - bajo) / (alto - bajo + 1e-12), 0.0, 1.0)

    if gamma is not None and gamma != 1.0:
        img_norm = img_norm ** (1.0 / gamma)
    return img_norm


def H_homomorfico(forma, gamma_H=1.5, gamma_L=0.5, D0=30.0, c=1.0):
    P, Q = forma
    v, u = np.mgrid[0:P, 0:Q]
    D2 = (u - Q / 2.0) ** 2 + (v - P / 2.0) ** 2
    return (gamma_H - gamma_L) * (1.0 - np.exp(-c * D2 / (D0 ** 2))) + gamma_L


def filtro_homomorfico(f, gamma_H=1.5, gamma_L=0.5, D0=30.0, c=1.0, eps=1e-6):
    N, M = f.shape
    z = np.log(np.maximum(f, 0.0) + eps)
    z_pad = np.pad(z, ((0, N), (0, M)), mode="symmetric")
    H = H_homomorfico(z_pad.shape, gamma_H, gamma_L, D0, c)
    S = H * fftshift(fft2(z_pad))
    s = ifft2(ifftshift(S)).real[:N, :M]
    return np.exp(s)


def metodo_durand(f, sigma_s=16.0, sigma_r=0.4, contraste_objetivo=5.0, eps=1e-6):
    log_I = np.log10(np.maximum(f, 0.0) + eps).astype(np.float32)

    base = cv2.bilateralFilter(log_I, d=-1, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    detalle = log_I - base

    rango_base = float(base.max() - base.min())
    factor_comp = np.log10(contraste_objetivo) / (rango_base + 1e-12)

    offset = -float(base.max()) * factor_comp
    log_out = base * factor_comp + detalle + offset
    return (10.0 ** log_out).astype(np.float64)


def hdr_por_canal_rgb(rgb, metodo_fn, **kwargs):
    canales = [metodo_fn(rgb[..., c], **kwargs) for c in range(3)]
    rgb_comp = np.stack(canales, axis=-1)
    return normalizar_rango(rgb_comp, gamma=2.2)


def reconstruir_color_beta(rgb_in, Y_in, Y_out, beta=1.0, eps=1e-6):
    rgb_pos = np.maximum(rgb_in.astype(np.float64), 0.0)
    ratio = rgb_pos / (Y_in[..., None] + eps)
    rgb_raw = Y_out[..., None] * (ratio ** beta)

    fuera_rango = np.any((rgb_raw < 0.0) | (rgb_raw > 1.0), axis=-1)
    pct_clip = 100.0 * np.mean(fuera_rango)

    return np.clip(rgb_raw, 0.0, 1.0), pct_clip


def reconstruir_color_hsv(rgb_in, Y_in, Y_out, alpha=1.0, eps=1e-6):
    rgb_base, _ = reconstruir_color_beta(rgb_in, Y_in, Y_out, beta=1.0, eps=eps)
    hsv = color.rgb2hsv(rgb_base)
    S_raw = alpha * hsv[..., 1]
    pct_clip = 100.0 * np.mean(S_raw > 1.0)

    hsv[..., 1] = np.clip(S_raw, 0.0, 1.0)
    return color.hsv2rgb(hsv), pct_clip


def metricas_color(rgb_ref_hdr, rgb_out, s_min=0.05, v_min=0.05):
    ref_pos = np.maximum(rgb_ref_hdr.astype(np.float64), 0.0)
    ref_scaled = ref_pos / (ref_pos.max() + 1e-12)

    hsv_in = color.rgb2hsv(ref_scaled)
    hsv_out = color.rgb2hsv(np.clip(rgb_out, 0.0, 1.0))

    mask = (hsv_in[..., 1] > s_min) & (hsv_out[..., 1] > s_min) & (hsv_out[..., 2] > v_min)

    h_in_deg = hsv_in[..., 0] * 360.0
    h_out_deg = hsv_out[..., 0] * 360.0
    diff = np.abs(h_out_deg - h_in_deg)
    delta_h = np.minimum(diff, 360.0 - diff)

    lab_out = color.rgb2lab(np.clip(rgb_out, 0.0, 1.0))
    chroma_out = np.hypot(lab_out[..., 1], lab_out[..., 2])

    return {
        "S_in_media": float(np.mean(hsv_in[..., 1])),
        "S_out_media": float(np.mean(hsv_out[..., 1])),
        "C_out_media": float(np.mean(chroma_out)),
        "delta_H_media_deg": float(np.mean(delta_h[mask])) if np.any(mask) else 0.0,
        "delta_H_p95_deg": float(np.percentile(delta_h[mask], 95)) if np.any(mask) else 0.0,
    }


def comparar_imagenes(imagenes: list, titulos: list, titulo_general: str, filas: int = 1, cols: int = 3):
    fig, axes = plt.subplots(filas, cols, figsize=(5 * cols, 4.5 * filas))
    axes = np.atleast_1d(axes).ravel()

    for ax, img, titulo in zip(axes, imagenes, titulos):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(titulo)
        ax.axis("off")

    fig.suptitle(titulo_general, fontsize=13)
    plt.tight_layout()


#==========================================================Parte 0 (implementacion necesaria)================================================================================================

ruta_imagen = "archivos_t4/belgium.hdr"

params_homo = dict(gamma_H=1.5, gamma_L=0.5, D0=30.0, c=1.0)
params_durand = dict(sigma_s=16.0, sigma_r=0.4, contraste_objetivo=5.0)

betas = [0.6, 1.0, 1.4]
alphas = [0.6, 1.0, 1.4]

imagen_bgr = cv2.imread(ruta_imagen, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
imagen_orig_vis = normalizar_rango(np.maximum(imagen_rgb, 0.0), gamma=2.2)

Y = calcular_luminancia(imagen_rgb)

Y_homo_raw = filtro_homomorfico(Y, **params_homo)
Y_homo = normalizar_rango(Y_homo_raw, gamma=2.2)

Y_dur_raw = metodo_durand(Y, **params_durand)
Y_dur = normalizar_rango(Y_dur_raw, gamma=2.2)

rgb_homo_indep = hdr_por_canal_rgb(imagen_rgb, filtro_homomorfico, **params_homo)
rgb_dur_indep = hdr_por_canal_rgb(imagen_rgb, metodo_durand, **params_durand)


#==========================================================Parte 1================================================================================================

imgs_homo_beta, titulos_homo_beta = [], []
imgs_dur_beta, titulos_dur_beta = [], []

for b in betas:
    img_h, clip_h = reconstruir_color_beta(imagen_rgb, Y, Y_homo, beta=b)
    met_h = metricas_color(imagen_rgb, img_h)
    imgs_homo_beta.append(img_h)
    titulos_homo_beta.append(f"Homomorfico (beta={b}) | clip={clip_h:.2f}%")
    print(f"[Homomorfico | beta={b}] clip={clip_h:.2f}% | {met_h}")

    img_d, clip_d = reconstruir_color_beta(imagen_rgb, Y, Y_dur, beta=b)
    met_d = metricas_color(imagen_rgb, img_d)
    imgs_dur_beta.append(img_d)
    titulos_dur_beta.append(f"Durand (beta={b}) | clip={clip_d:.2f}%")
    print(f"[Durand | beta={b}] clip={clip_d:.2f}% | {met_d}")

comparar_imagenes(
    imgs_homo_beta, titulos_homo_beta,
    "Estrategia 1 (razon de cromaticidad) - Filtro Homomorfico",
    filas=1, cols=3
)
comparar_imagenes(
    imgs_dur_beta, titulos_dur_beta,
    "Estrategia 1 (razon de cromaticidad) - Metodo de Durand",
    filas=1, cols=3
)


#==========================================================Parte 2================================================================================================

imgs_homo_hsv, titulos_homo_hsv = [], []
imgs_dur_hsv, titulos_dur_hsv = [], []

for a in alphas:
    img_h, clip_h = reconstruir_color_hsv(imagen_rgb, Y, Y_homo, alpha=a)
    met_h = metricas_color(imagen_rgb, img_h)
    imgs_homo_hsv.append(img_h)
    titulos_homo_hsv.append(f"Homomorfico HSV (alpha={a}) | clip={clip_h:.2f}%")
    print(f"[Homomorfico | HSV alpha={a}] clip={clip_h:.2f}% | {met_h}")

    img_d, clip_d = reconstruir_color_hsv(imagen_rgb, Y, Y_dur, alpha=a)
    met_d = metricas_color(imagen_rgb, img_d)
    imgs_dur_hsv.append(img_d)
    titulos_dur_hsv.append(f"Durand HSV (alpha={a}) | clip={clip_d:.2f}%")
    print(f"[Durand | HSV alpha={a}] clip={clip_d:.2f}% | {met_d}")

comparar_imagenes(
    imgs_homo_hsv, titulos_homo_hsv,
    "Estrategia 2 (ajuste de saturacion HSV) - Filtro Homomorfico",
    filas=1, cols=3
)
comparar_imagenes(
    imgs_dur_hsv, titulos_dur_hsv,
    "Estrategia 2 (ajuste de saturacion HSV) - Metodo de Durand",
    filas=1, cols=3
)


#==========================================================Parte 3================================================================================================

print("[Homomorfico | RGB indep]", metricas_color(imagen_rgb, rgb_homo_indep))
print("[Durand | RGB indep]", metricas_color(imagen_rgb, rgb_dur_indep))

comparar_imagenes(
    [imagen_orig_vis, Y_homo, rgb_homo_indep, imgs_homo_beta[0], imgs_homo_beta[1], imgs_homo_hsv[0]],
    [
        "Original (gamma=2.2)", "Luminancia Homomorfico", "Homomorfico RGB indep.",
        f"Homomorfico beta={betas[0]}", f"Homomorfico beta={betas[1]}", f"Homomorfico HSV alpha={alphas[0]}"
    ],
    "Comparacion principal - Filtro Homomorfico",
    filas=2, cols=3
)
comparar_imagenes(
    [imagen_orig_vis, Y_dur, rgb_dur_indep, imgs_dur_beta[0], imgs_dur_beta[1], imgs_dur_hsv[0]],
    [
        "Original (gamma=2.2)", "Luminancia Durand", "Durand RGB indep.",
        f"Durand beta={betas[0]}", f"Durand beta={betas[1]}", f"Durand HSV alpha={alphas[0]}"
    ],
    "Comparacion principal - Metodo de Durand",
    filas=2, cols=3
)


recorte_color = (200, 450, 50, 300)
recorte_borde = (150, 400, 320, 570)

for nombre_rec, (y1, y2, x1, x2) in [("Colores intensos", recorte_color), ("Transicion fuerte", recorte_borde)]:
    comparar_imagenes(
        [
            rgb_homo_indep[y1:y2, x1:x2],
            imgs_homo_beta[0][y1:y2, x1:x2],
            imgs_homo_hsv[0][y1:y2, x1:x2],
            rgb_dur_indep[y1:y2, x1:x2],
            imgs_dur_beta[0][y1:y2, x1:x2],
            imgs_dur_hsv[0][y1:y2, x1:x2],
        ],
        [
            "Homo RGB indep.", f"Homo beta={betas[0]}", f"Homo HSV alpha={alphas[0]}",
            "Durand RGB indep.", f"Durand beta={betas[0]}", f"Durand HSV alpha={alphas[0]}",
        ],
        f"Recorte ampliado: {nombre_rec} [{y1}:{y2}, {x1}:{x2}]",
        filas=2, cols=3
    )

plt.show()